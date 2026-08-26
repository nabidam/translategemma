"""Data stage: one frozen test set, one train pool, nested training subsets.

Order matters and is not negotiable:

  1. The 500-row in-domain test set is carved out FIRST, with
     build_test_set.py's strategy (stratified, diversity-maximised, then a
     document-level holdout plus a near-duplicate purge of the remainder). Its
     leftover `train_pool.csv` is the only pool the training subsets may draw
     from, so no training row shares a document with a test row.
  2. Subsets are NESTED: 5k ⊂ 10k ⊂ 50k ⊂ 100k, built from one document order.
     A volume-to-volume delta is therefore data *added*, not data swapped, which
     is the only way the volume curve means anything.
  3. Each subset is split into train/validation with split_dataset.py's
     document-level strategy. No test split is produced here — the test sets are
     the frozen files in `test_sets`.

Both repository scripts are invoked as subprocesses against generated configs
rather than imported, because both are written as CLI entry points with global
state (build_test_set.py seeds a module-level RNG in main()).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BASE_TESTSET_CONFIG,
    BASE_TRAINING_CONFIG,
    PROJECT_ROOT,
    SweepConfig,
    deep_merge,
    load_yaml,
    volume_label,
    write_yaml,
)
from .scheduler import logger

# The SFT record shape train.py and split_dataset.py expect. Column *names* come
# from config.yaml's data section, so they are read rather than hard-coded.
SFT_COLUMNS = ("id", "domain", "source", "target", "source_lang", "target_lang")


def _training_data_columns() -> dict[str, str]:
    data_cfg = load_yaml(BASE_TRAINING_CONFIG)["data"]
    return {
        "id": data_cfg["id_column"],
        "domain": data_cfg["domain_column"],
        "source": data_cfg["source_column"],
        "target": data_cfg["target_column"],
        "source_lang": data_cfg["source_lang_column"],
        "target_lang": data_cfg["target_lang_column"],
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running: %s", " ".join(command))
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        raise RuntimeError(
            f"{command[1]} failed with exit code {completed.returncode}. Log: {log_path}\n" + "\n".join(tail)
        )


def normalize_corpus(config: SweepConfig, force: bool = False) -> Path:
    """Write the corpus as the id/en/fa/domain CSV build_test_set.py requires."""
    output = config.work_dir / "corpus_normalized.csv"
    if output.exists() and not force:
        logger.info("Reusing normalized corpus %s", output)
        return output
    source = config.resolve(config.corpus["csv_path"])
    columns = config.corpus["columns"]
    logger.info("Reading corpus %s", source)
    frame = pd.read_csv(source, dtype=str)
    missing = [name for name in columns.values() if name not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing corpus columns {missing}; found {list(frame.columns)}")
    normalized = pd.DataFrame(
        {
            "id": frame[columns["id"]],
            "en": frame[columns["source"]].fillna("").str.strip(),
            "fa": frame[columns["target"]].fillna("").str.strip(),
            # Stripped: a trailing space in one row's domain label would
            # otherwise become a separate domain in every quota and slice table.
            "domain": frame[columns["domain"]].fillna("unknown").astype(str).str.strip(),
        }
    )
    before = len(normalized)
    normalized = normalized[(normalized["en"] != "") & (normalized["fa"] != "")]
    normalized = normalized.dropna(subset=["id"]).drop_duplicates(subset=["id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    logger.info(
        "Normalized corpus: %d rows (%d dropped as blank/duplicate-id), %d domains -> %s",
        len(normalized), before - len(normalized), normalized["domain"].nunique(), output,
    )
    return output


def _testset_spec(config: SweepConfig, corpus_csv: Path, corpus_rows: int) -> dict:
    """Identity of the in-domain test set: everything that changes which rows it holds."""
    settings = config.data["in_domain_test"]
    return {
        "size": int(settings["size"]),
        "exclude_domains": sorted(config.excluded_test_domains),
        "overrides": settings.get("overrides") or {},
        "id_separator": config.corpus["id_separator"],
        "corpus": {"path": str(corpus_csv), "rows": corpus_rows},
    }


def _excluded_domain_labels(config: SweepConfig, corpus: pd.DataFrame) -> list[str]:
    """Resolve configured exclusions to the corpus's own labels."""
    requested = config.excluded_test_domains
    if not requested:
        return []
    labels = [str(value) for value in corpus["domain"].dropna().unique()]
    by_fold = {_fold(label): label for label in labels}
    resolved, missing = [], []
    for name in requested:
        if name in labels:
            resolved.append(name)
        elif match := by_fold.get(_fold(name)):
            logger.warning(
                "data.in_domain_test.exclude_domains entry %r matched the corpus label %r "
                "(case/whitespace only).", name, match,
            )
            resolved.append(match)
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            f"data.in_domain_test.exclude_domains names {missing}, which are not corpus domains. "
            f"Available: {sorted(labels)}"
        )
    return resolved


def build_in_domain_test_set(config: SweepConfig, corpus_csv: Path, force: bool = False) -> tuple[Path, Path]:
    """Return (test_csv, train_pool_csv), running build_test_set.py when needed.

    Domains in data.in_domain_test.exclude_domains are withheld from the builder
    and appended to the train pool afterwards. That exists for a specific,
    common shape: a domain small enough that the document-level holdout consumes
    all of it, leaving it with no training rows at all. Keeping it out of the
    test set is the only way to keep it trainable — its quality is then measured
    on NTREX/FLORES and by the other domains' in-domain rows, not by its own.
    """
    settings = config.data["in_domain_test"]
    if settings["mode"] == "existing":
        return _use_existing_test_set(config, corpus_csv, force)

    test_path, pool_path = config.in_domain_test_path, config.train_pool_csv
    corpus = pd.read_csv(corpus_csv, dtype=str)
    spec = _testset_spec(config, corpus_csv, len(corpus))
    spec_path = config.testset_build_dir / "testset_spec.json"
    if test_path.exists() and pool_path.exists() and not force:
        stored = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else None
        if stored == spec:
            logger.info("Reusing in-domain test set %s and train pool %s", test_path, pool_path)
            return test_path, pool_path
        logger.warning(
            "Rebuilding the in-domain test set: it was built from a different specification "
            "(size, excluded domains, builder overrides or corpus changed)."
        )

    excluded = _excluded_domain_labels(config, corpus)
    builder_input = corpus_csv
    if excluded:
        kept = corpus[~corpus["domain"].isin(excluded)]
        builder_input = config.work_dir / "corpus_for_testset.csv"
        kept.to_csv(builder_input, index=False)
        logger.info(
            "Withholding %d rows in domain(s) %s from the test-set builder; they stay fully available "
            "for training.", len(corpus) - len(kept), excluded,
        )

    # Forced, not configurable: the sweep's contract is 500 test rows, no dev
    # split, and a train pool written where the subset builder looks for it.
    overrides = deep_merge(
        settings.get("overrides") or {},
        {
            "input": {"csv_path": str(builder_input), "id_separator": config.corpus["id_separator"],
                      "source_lang_col": "en", "target_lang_col": "fa"},
            "selection": {"total_size": int(settings["size"])},
            "splits": {"dev_test_split": False},
            "output": {
                "dir": str(config.testset_build_dir),
                "test_file": "test.csv",
                "train_pool_file": "train_pool.csv",
            },
            # Kept beside this sweep's artefacts: the cache is keyed by a hash of
            # the corpus text, so sharing testset_output/embeddings.npy with an
            # unrelated run only invites a silent, expensive re-encode.
            "embeddings": {"cache_path": str(config.testset_build_dir / "embeddings.npy")},
        },
    )
    derived = write_yaml(
        config.derived_config_dir / "testset_config.yaml",
        deep_merge(load_yaml(BASE_TESTSET_CONFIG), overrides),
    )
    logger.info("Building the %d-row in-domain test set (build_test_set.py)", settings["size"])
    _run(
        [sys.executable, "build_test_set.py", "--config", str(derived)],
        config.output_dir / "logs" / "build_test_set.log",
    )
    if not test_path.exists() or not pool_path.exists():
        raise RuntimeError(f"build_test_set.py did not produce {test_path} and {pool_path}")
    if excluded:
        _restore_excluded_domains(config, corpus, excluded, test_path, pool_path)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return test_path, pool_path


def _restore_excluded_domains(
    config: SweepConfig, corpus: pd.DataFrame, excluded: list[str], test_path: Path, pool_path: Path
) -> None:
    """Append the withheld domains' rows to the train pool.

    The document-level guarantee is preserved explicitly: a document that also
    contributed a test row (possible when one document carries rows of more than
    one domain) is dropped again. What is NOT re-run for these rows is the
    embedding near-duplicate purge, which needs the builder's embedding matrix;
    they contributed no test rows, so only a cross-domain duplicate could leak,
    and the document check above already covers the same-document case.
    """
    separator = config.corpus["id_separator"]
    pool = pd.read_csv(pool_path, dtype=str)
    test = pd.read_csv(test_path, dtype=str)
    held_documents = set(test["document_id"].astype(str)) if "document_id" in test.columns else {
        str(value).split(separator)[0] for value in test["id"]
    }
    extra = corpus[corpus["domain"].isin(excluded)].copy()
    extra["document_id"] = extra["id"].astype(str).str.split(separator).str[0]
    overlapping = extra["document_id"].isin(held_documents)
    if int(overlapping.sum()):
        logger.warning(
            "Dropping %d withheld row(s) whose document also contributed a test row.",
            int(overlapping.sum()),
        )
    extra = extra[~overlapping]
    combined = pd.concat([pool, extra[[column for column in pool.columns if column in extra.columns]]],
                         ignore_index=True)
    combined = combined.drop_duplicates(subset=["id"])
    combined.to_csv(pool_path, index=False)
    logger.info(
        "Train pool: %d rows after restoring %d withheld row(s) in %s. Domains now: %s",
        len(combined), len(extra), excluded, combined["domain"].value_counts().to_dict(),
    )


def _use_existing_test_set(config: SweepConfig, corpus_csv: Path, force: bool) -> tuple[Path, Path]:
    """Quarantine every document that the supplied test set touches.

    Without this, a hand-picked test file drawn from the same corpus leaks its
    sibling chunks into training and every score above the base model is
    partly a memorisation score.
    """
    test_path = config.in_domain_test_path
    pool_path = config.work_dir / "in_domain_testset" / "train_pool.csv"
    if pool_path.exists() and not force:
        logger.info("Reusing train pool %s", pool_path)
        return test_path, pool_path
    separator = config.corpus["id_separator"]
    test_frame = pd.read_csv(test_path, dtype=str)
    id_column = (config.data["in_domain_test"].get("id_column") or "id")
    if id_column not in test_frame.columns:
        raise ValueError(f"{test_path} has no {id_column!r} column to derive document ids from")
    held = {str(value).split(separator)[0] for value in test_frame[id_column]}
    corpus = pd.read_csv(corpus_csv, dtype=str)
    corpus["document_id"] = corpus["id"].astype(str).str.split(separator).str[0]
    pool = corpus[~corpus["document_id"].isin(held)]
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(pool_path, index=False)
    logger.info(
        "Train pool from existing test set: %d rows kept, %d quarantined across %d held-out documents -> %s",
        len(pool), len(corpus) - len(pool), len(held), pool_path,
    )
    return test_path, pool_path


def prepare_test_sets(config: SweepConfig, force: bool = False) -> dict[str, Path]:
    """Normalize every enabled test set to one CSV shape under the work dir.

    External sets (NTREX, FLORES) carry no domain column and sometimes no id
    column; both are required for the slice tables and for the strict id join
    the benchmark performs between a dataset and a candidate's output.
    """
    directory = config.work_dir / "testsets"
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for test_set in config.test_sets:
        output = directory / f"{test_set['id']}.csv"
        spec_path = directory / f"{test_set['id']}.spec.json"
        paths[test_set["id"]] = output
        source = config.test_set_path(test_set)
        if not source.exists():
            raise FileNotFoundError(
                f"Test set {test_set['id']!r} not found at {source}. Set test_sets[].enabled: false "
                "to exclude it, or correct its path."
            )
        # Content-addressed, not existence-checked. The in-domain source is
        # rebuilt whenever its own specification changes, and an external file
        # can be corrected in place; either way the prepared copy that the
        # benchmark actually scores must not be the previous one.
        spec = {
            "source": str(source),
            "sha256": _file_sha256(source),
            "columns": test_set["columns"],
            "default_domain": test_set.get("default_domain"),
            "max_examples": test_set.get("max_examples"),
        }
        if output.exists() and not force:
            stored = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else None
            if stored == spec:
                logger.info("Reusing prepared test set %s", output)
                continue
            logger.warning(
                "Rebuilding prepared test set %s: %s changed since it was prepared.",
                output, source,
            )
        columns = test_set["columns"]
        frame = _read_any(source)
        for field in ("source", "reference"):
            if columns[field] not in frame.columns:
                raise ValueError(
                    f"Test set {test_set['id']} is missing column {columns[field]!r}; found {list(frame.columns)}"
                )
        prepared = pd.DataFrame(
            {
                "en": frame[columns["source"]].astype(str).str.strip(),
                "fa": frame[columns["reference"]].astype(str).str.strip(),
            }
        )
        if columns["id"] in frame.columns:
            prepared.insert(0, "id", frame[columns["id"]].astype(str))
        else:
            logger.warning(
                "Test set %s has no %r column; using row-ordinal ids (%s-00001, ...).",
                test_set["id"], columns["id"], test_set["id"],
            )
            prepared.insert(0, "id", [f"{test_set['id']}-{index + 1:05d}" for index in range(len(prepared))])
        domain_column = columns.get("domain")
        if domain_column and domain_column in frame.columns:
            prepared["domain"] = frame[domain_column].fillna(test_set.get("default_domain", test_set["id"]))
        else:
            prepared["domain"] = test_set.get("default_domain", test_set["id"])
        prepared = prepared[(prepared["en"] != "") & (prepared["fa"] != "")]
        if prepared["id"].duplicated().any():
            raise ValueError(f"Test set {test_set['id']} has duplicate ids after preparation")
        if maximum := test_set.get("max_examples"):
            prepared = prepared.head(int(maximum))
        prepared.to_csv(output, index=False)
        spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Prepared test set %s: %d rows -> %s", test_set["id"], len(prepared), output)
    return paths


def _read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl", dtype=str)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", dtype=str)
    raise ValueError(f"Unsupported test set format {path.suffix!r}: {path}")


ALL_DOMAINS = "__all__"
EMPTY = pd.DataFrame()


def _domain_orders(pool: pd.DataFrame, seed: int, stratified: bool, unit: str = "row"):
    """One fixed, shuffled draw order per domain (or one global order).

    Fixed is the whole point: every volume takes a longer prefix of the same
    order, which is what makes the subsets nested. `stratified=False` collapses
    all domains into one bucket, i.e. a purely random draw.

    unit="row" shuffles the domain's ROWS, so a quota is filled from as many
    documents as the pool offers — the composition is exact and each subset is
    as document-diverse as the data allows. unit="document" shuffles documents
    instead and takes them whole, which keeps a subset's rows contiguous inside
    their source documents; with few, very large documents that makes a small
    subset come from only a handful of them.
    """
    rng = np.random.default_rng(seed)
    orders: dict[str, Any] = {}
    groups = pool.groupby("domain", sort=True) if stratified else [(ALL_DOMAINS, pool)]
    for domain, group in groups:
        if unit == "row":
            positions = group.index.to_numpy().copy()
            rng.shuffle(positions)
            orders[str(domain)] = positions
        else:
            documents = group["document_id"].drop_duplicates().to_numpy()
            rng.shuffle(documents)
            orders[str(domain)] = [str(document) for document in documents]
    return orders


def _fold(value: str) -> str:
    """Case- and whitespace-insensitive key for matching a domain label."""
    return " ".join(str(value).split()).casefold()


def _match_domains(requested: list[str], pool: pd.DataFrame, config: SweepConfig) -> dict[str, str]:
    """Map each configured domain name to the label the train pool actually uses.

    Exact matches win. A name that differs only in case or surrounding
    whitespace is matched and logged, because "Computer Science" versus
    "computer science " is a typo in the config, not a different domain.
    Anything still unmatched is an error -- and the message distinguishes the two
    causes, which need different fixes: a wrong label, or a domain that exists in
    the corpus but was entirely quarantined into the test-set holdout.
    """
    pool_labels = [str(value) for value in pool["domain"].dropna().unique()]
    by_fold: dict[str, list[str]] = {}
    for label in pool_labels:
        by_fold.setdefault(_fold(label), []).append(label)

    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    for name in requested:
        if name in pool_labels:
            mapping[name] = name
            continue
        candidates = by_fold.get(_fold(name), [])
        if len(candidates) == 1:
            logger.warning(
                "data.composition domain %r matched the train pool label %r (case/whitespace only). "
                "Fix the config to match exactly.", name, candidates[0],
            )
            mapping[name] = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                f"data.composition domain {name!r} is ambiguous: the train pool has {candidates}. "
                "Use the exact label."
            )
        else:
            unmatched.append(name)
    if not unmatched:
        return mapping

    corpus_labels: list[str] = []
    corpus_csv = config.work_dir / "corpus_normalized.csv"
    if corpus_csv.exists():
        corpus_labels = [
            str(value)
            for value in pd.read_csv(corpus_csv, usecols=["domain"], dtype=str)["domain"].dropna().unique()
        ]
    held_out = [name for name in unmatched if any(_fold(name) == _fold(label) for label in corpus_labels)]
    if held_out:
        raise ValueError(
            f"Domain(s) {held_out} exist in the corpus but have no rows left in the train pool: every "
            "document of theirs went into the in-domain test set's document-level holdout. Lower "
            "data.in_domain_test.overrides.selection.min_per_domain (or raise max_test_documents) and "
            f"rebuild the test set with --force, or drop them from data.composition. Train pool labels: "
            f"{sorted(pool_labels)}"
        )
    raise ValueError(
        f"data.composition names domain(s) {unmatched} that do not exist. Train pool labels: "
        f"{sorted(pool_labels)}"
        + (f"; corpus labels: {sorted(corpus_labels)}" if corpus_labels else "")
        + ". Labels are compared exactly (case and inner whitespace included)."
    )


def _resolve_shares(config: SweepConfig, pool: pd.DataFrame, volume: int) -> dict[str, float]:
    """Target domain shares for one volume, keyed by the pool's own labels."""
    composition = config.composition
    shares = config.composition_shares(volume)
    if composition["mode"] == "random" and not shares:
        return {ALL_DOMAINS: 1.0}
    if shares:
        mapping = _match_domains(list(shares), pool, config)
        return {mapping[name]: float(share) for name, share in shares.items()}
    counts = pool["domain"].value_counts(normalize=True)
    return {str(domain): float(share) for domain, share in counts.items()}


def _quotas(shares: dict[str, float], volume: int, available: dict[str, int], on_shortfall: str) -> dict[str, int]:
    """Row quota per domain: shares scaled to `volume`, then made feasible.

    Largest-remainder rounding, so the quotas sum to exactly `volume` instead of
    volume +/- the number of domains. A domain that cannot fill its quota is an
    error by default: silently shifting its rows elsewhere would change what the
    cell measures without saying so.
    """
    exact = {domain: shares.get(domain, 0.0) * volume for domain in shares}
    quotas = {domain: int(value) for domain, value in exact.items()}
    remainder = volume - sum(quotas.values())
    ranked = sorted(exact.items(), key=lambda item: (-(item[1] - int(item[1])), item[0]))
    for domain, _ in ranked[:remainder]:
        quotas[domain] += 1

    short = {domain: quota for domain, quota in quotas.items() if quota > available.get(domain, 0)}
    if not short:
        return quotas
    detail = ", ".join(
        f"{domain}: need {quotas[domain]}, have {available.get(domain, 0)}" for domain in sorted(short)
    )
    if on_shortfall == "error":
        raise ValueError(
            f"Domain composition is not satisfiable at volume {volume} ({detail}). Lower the volume, "
            "change data.composition.domain_shares, or set data.composition.on_shortfall: redistribute."
        )
    logger.warning("Volume %d: %s. Redistributing the deficit over the domains with room.", volume, detail)
    deficit = 0
    for domain in short:
        deficit += quotas[domain] - available.get(domain, 0)
        quotas[domain] = available.get(domain, 0)
    # Every domain in the pool becomes eligible, not only the ones the requested
    # shares named: a share of 0 means "not wanted", but redistribution is the
    # explicit instruction to fill the gap from wherever rows exist.
    for domain in available:
        quotas.setdefault(domain, 0)
    room = {domain: available.get(domain, 0) - quota for domain, quota in quotas.items()}
    while deficit > 0:
        open_domains = sorted((domain for domain, value in room.items() if value > 0),
                             key=lambda name: (-room[name], name))
        if not open_domains:
            raise ValueError(f"Train pool cannot supply {volume} rows: {deficit} rows short after redistribution.")
        for domain in open_domains:
            if deficit == 0:
                break
            quotas[domain] += 1
            room[domain] -= 1
            deficit -= 1
    return quotas


def _document_groups(pool: pd.DataFrame, stratified: bool) -> dict[tuple[str, str], pd.DataFrame]:
    """Pool rows keyed by (domain bucket, document).

    Keyed by BOTH, not by document alone. A document here can carry rows of more
    than one domain -- the corpus labels domains per row, and this corpus has very
    few, very large documents -- so a per-domain quota filled with whole
    documents would drag another domain's rows in with it and the realized mix
    would not be the configured one.
    """
    if not stratified:
        return {
            (ALL_DOMAINS, str(document)): group
            for document, group in pool.groupby("document_id", sort=False)
        }
    return {
        (str(domain), str(document)): group
        for (domain, document), group in pool.groupby(["domain", "document_id"], sort=False)
    }


def _take_rows(
    documents: list[str],
    groups: dict[tuple[str, str], pd.DataFrame],
    bucket: str,
    target: int,
) -> list[pd.DataFrame]:
    """Prefix of `documents` holding exactly `target` rows of `bucket`."""
    picked: list[pd.DataFrame] = []
    rows = 0
    for document in documents:
        group = groups.get((bucket, document))
        if group is None or group.empty:
            continue
        if rows + len(group) > target:
            # Truncating one document keeps the cell at exactly `target` rows.
            # Deterministic (sorted by id), and harmless for nesting: volumes are
            # supersets of one another, not disjoint folds.
            picked.append(group.sort_values("id").head(target - rows))
            return picked
        picked.append(group)
        rows += len(group)
        if rows == target:
            return picked
    return picked


def _select_subset(
    config: SweepConfig,
    pool: pd.DataFrame,
    orders: dict[str, Any],
    groups: dict[tuple[str, str], pd.DataFrame] | None,
    volume: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill each domain's row quota from that domain's own fixed draw order."""
    unit = config.composition["unit"]
    shares = _resolve_shares(config, pool, volume)
    if unit == "row":
        available = {bucket: int(len(order)) for bucket, order in orders.items()}
    else:
        available = {
            bucket: int(sum(len(groups.get((bucket, document), EMPTY)) for document in documents))
            for bucket, documents in orders.items()
        }
    quotas = _quotas(shares, volume, available, config.composition["on_shortfall"])
    frames: list[pd.DataFrame] = []
    for bucket, quota in sorted(quotas.items()):
        if quota <= 0:
            continue
        if bucket not in orders:
            raise ValueError(f"No rows for domain {bucket!r} in the train pool")
        if unit == "row":
            frames.append(pool.loc[orders[bucket][:quota]])
        else:
            frames.extend(_take_rows(orders[bucket], groups, bucket, quota))
    return pd.concat(frames, ignore_index=True), quotas


def _subset_spec(config: SweepConfig, pool_csv: Path, pool_rows: int, volume: int) -> dict:
    """Identity of a subset: everything that changes which rows it holds."""
    return {
        # Bumped when the selector's behaviour changes, so subsets built by an
        # earlier version are rebuilt instead of silently reused. v2: per-domain
        # quotas take only that domain's rows from a mixed-domain document.
        # v3: draw unit is configurable and defaults to rows.
        "builder_version": 3,
        "volume": volume,
        "seed": int(config.sweep["seed"]),
        "validation_ratio": float(config.data["validation_ratio"]),
        "composition_mode": config.composition["mode"],
        "draw_unit": config.composition["unit"],
        "shares": config.composition_shares(volume),
        "pool": {"path": str(pool_csv), "rows": pool_rows},
    }


def _subset_is_current(paths: dict[str, Path], spec: dict, validation_ratio: float) -> bool:
    """Whether an existing subset was built from this exact specification.

    Existence alone is not enough: editing data.composition (or the seed, or the
    validation ratio) after a first run would otherwise reuse the old rows
    silently, and every later number would describe a mix nobody configured.
    """
    if not paths["train"].exists():
        return False
    if validation_ratio > 0 and not paths["validation"].exists():
        return False
    spec_path = paths["train"].parent / "subset_spec.json"
    if not spec_path.exists():
        return False
    try:
        return json.loads(spec_path.read_text(encoding="utf-8")) == spec
    except json.JSONDecodeError:
        return False


def build_subsets(config: SweepConfig, pool_csv: Path, force: bool = False) -> dict[int, dict[str, Path]]:
    """Write nested train/validation splits for every configured volume."""
    columns = _training_data_columns()
    separator = config.corpus["id_separator"]
    pool = pd.read_csv(pool_csv, dtype=str)
    if "document_id" not in pool.columns:
        pool["document_id"] = pool["id"].astype(str).str.split(separator).str[0]
    pool = pool.dropna(subset=["en", "fa"])
    largest = max(config.volumes)
    if len(pool) < largest:
        raise ValueError(
            f"Train pool has {len(pool)} rows but sweep.volumes asks for {largest}. Lower the largest "
            f"volume, or reduce the holdout cost (data.in_domain_test.overrides.selection.max_test_documents)."
        )

    composition = config.composition
    unit = composition["unit"]
    stratified = composition["mode"] != "random" or bool(composition["per_volume"])
    # Positional index, so a row order can address rows through pool.loc.
    pool = pool.reset_index(drop=True)
    orders = _domain_orders(pool, int(config.sweep["seed"]), stratified, unit)
    if composition["per_volume"]:
        logger.warning(
            "data.composition.per_volume is set, so the volumes target different compositions and are NOT "
            "nested. A volume-to-volume delta then mixes 'more data' with 'different data'."
        )
    logger.info(
        "Train pool: %d rows, %d documents, composition mode '%s', draw unit '%s'",
        len(pool), pool["document_id"].nunique(), composition["mode"], unit,
    )
    groups = _document_groups(pool, stratified) if unit == "document" else None
    mixed = int((pool.groupby("document_id")["domain"].nunique() > 1).sum())
    if mixed:
        logger.info(
            "%d document(s) carry rows of more than one domain; per-domain quotas take only the rows of "
            "the domain they are filling, so a document may contribute to two of them.", mixed,
        )

    results: dict[int, dict[str, Path]] = {}
    manifest: dict[str, dict] = {}
    for volume in config.volumes:
        paths = config.subset_split_paths(volume)
        results[volume] = paths
        spec = _subset_spec(config, pool_csv, len(pool), volume)
        if not force and _subset_is_current(paths, spec, float(config.data["validation_ratio"])):
            logger.info("Reusing subset %s", paths["train"].parent)
            continue
        if paths["train"].exists() and not force:
            logger.warning(
                "Rebuilding subset %s: it was built from a different specification (composition, seed, "
                "validation ratio or train pool changed).", paths["train"].parent,
            )
        subset, quotas = _select_subset(config, pool, orders, groups, volume)
        realized = subset["domain"].value_counts(normalize=True).round(4).to_dict()
        subset_jsonl = _write_sft_jsonl(subset, config, columns, config.subset_dir(volume) / "subset.jsonl")
        manifest[volume_label(volume)] = {
            "requested_rows": volume,
            "rows": len(subset),
            "documents": int(subset["document_id"].nunique()),
            "composition_mode": composition["mode"],
            "draw_unit": unit,
            "target_row_quotas": quotas,
            "realized_domain_shares": realized,
            "domain_rows": subset["domain"].value_counts().to_dict(),
            "subset_path": str(subset_jsonl),
        }
        _split_subset(config, subset_jsonl, volume)
        (config.subset_dir(volume) / "subset_spec.json").write_text(
            json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "Volume %s: %d rows, %d documents, domains %s -> %s",
            volume_label(volume), len(subset), subset["document_id"].nunique(), realized, paths["train"],
        )

    if manifest:
        path = config.work_dir / "subsets" / "subset_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def _write_sft_jsonl(frame: pd.DataFrame, config: SweepConfig, columns: dict[str, str], path: Path) -> Path:
    """Write rows in the SFT record shape, with per-row language codes."""
    records = pd.DataFrame(
        {
            columns["id"]: frame["id"].astype(str),
            columns["domain"]: frame["domain"].astype(str),
            columns["source"]: frame["en"].astype(str),
            columns["target"]: frame["fa"].astype(str),
            columns["source_lang"]: config.corpus["source_lang"],
            columns["target_lang"]: config.corpus["target_lang"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    records.to_json(path, orient="records", lines=True, force_ascii=False)
    return path


def _split_subset(config: SweepConfig, subset_jsonl: Path, volume: int) -> None:
    ratio = float(config.data["validation_ratio"])
    overrides = {
        "splitting": {
            "input_dataset_path": str(subset_jsonl),
            "output_dir": str(config.subset_dir(volume)),
            "train_ratio": 1.0 - ratio,
            "validation_ratio": ratio,
            "test_ratio": 0.0,
            "seed": int(config.sweep["seed"]),
            "group_id_delimiter": config.corpus["id_separator"],
        }
    }
    derived = write_yaml(
        config.derived_config_dir / f"split_{volume_label(volume)}.yaml",
        deep_merge(load_yaml(BASE_TRAINING_CONFIG), overrides),
    )
    _run(
        [sys.executable, "split_dataset.py", "--config", str(derived)],
        config.output_dir / "logs" / f"split_{volume_label(volume)}.log",
    )


def run(config: SweepConfig, force: bool = False) -> dict:
    """Execute the whole data stage and return a summary of what it produced."""
    corpus_csv = normalize_corpus(config, force)
    test_path, pool_path = build_in_domain_test_set(config, corpus_csv, force)
    test_sets = prepare_test_sets(config, force)
    subsets = build_subsets(config, pool_path, force)
    summary = {
        "corpus_csv": str(corpus_csv),
        "in_domain_test": str(test_path),
        "train_pool": str(pool_path),
        "test_sets": {key: str(value) for key, value in test_sets.items()},
        "subsets": {volume_label(volume): {name: str(path) for name, path in paths.items()}
                    for volume, paths in subsets.items()},
    }
    path = config.output_dir / "data_stage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Data stage complete: %s", path)
    return summary
