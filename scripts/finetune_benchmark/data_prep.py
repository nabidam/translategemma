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

import json
import subprocess
import sys
from pathlib import Path

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


def build_in_domain_test_set(config: SweepConfig, corpus_csv: Path, force: bool = False) -> tuple[Path, Path]:
    """Return (test_csv, train_pool_csv), running build_test_set.py when needed."""
    settings = config.data["in_domain_test"]
    if settings["mode"] == "existing":
        return _use_existing_test_set(config, corpus_csv, force)

    test_path, pool_path = config.in_domain_test_path, config.train_pool_csv
    if test_path.exists() and pool_path.exists() and not force:
        logger.info("Reusing in-domain test set %s and train pool %s", test_path, pool_path)
        return test_path, pool_path

    # Forced, not configurable: the sweep's contract is 500 test rows, no dev
    # split, and a train pool written where the subset builder looks for it.
    overrides = deep_merge(
        settings.get("overrides") or {},
        {
            "input": {"csv_path": str(corpus_csv), "id_separator": config.corpus["id_separator"],
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
    return test_path, pool_path


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
        paths[test_set["id"]] = output
        if output.exists() and not force:
            logger.info("Reusing prepared test set %s", output)
            continue
        source = config.test_set_path(test_set)
        if not source.exists():
            raise FileNotFoundError(
                f"Test set {test_set['id']!r} not found at {source}. Set test_sets[].enabled: false "
                "to exclude it, or correct its path."
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


def _domain_orders(pool: pd.DataFrame, seed: int, stratified: bool) -> dict[str, list[str]]:
    """One fixed, shuffled document order per domain (or one global order).

    Fixed is the whole point: every volume takes a longer prefix of the same
    order, which is what makes the subsets nested. `stratified=False` collapses
    all domains into one bucket, i.e. a purely random draw.
    """
    rng = np.random.default_rng(seed)
    orders: dict[str, list[str]] = {}
    groups = pool.groupby("domain", sort=True) if stratified else [(ALL_DOMAINS, pool)]
    for domain, group in groups:
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


def _take_rows(documents: list[str], by_document: dict[str, pd.DataFrame], target: int) -> list[pd.DataFrame]:
    """Prefix of `documents` holding exactly `target` rows (last one truncated)."""
    picked: list[pd.DataFrame] = []
    rows = 0
    for document in documents:
        group = by_document[document]
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
    orders: dict[str, list[str]],
    by_document: dict[str, pd.DataFrame],
    volume: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    shares = _resolve_shares(config, pool, volume)
    available = {
        domain: int(sum(len(by_document[document]) for document in documents))
        for domain, documents in orders.items()
    }
    quotas = _quotas(shares, volume, available, config.composition["on_shortfall"])
    frames: list[pd.DataFrame] = []
    for domain, quota in sorted(quotas.items()):
        if quota <= 0:
            continue
        if domain not in orders:
            raise ValueError(f"No documents for domain {domain!r} in the train pool")
        frames.extend(_take_rows(orders[domain], by_document, quota))
    return pd.concat(frames, ignore_index=True), quotas


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
    stratified = composition["mode"] != "random" or bool(composition["per_volume"])
    orders = _domain_orders(pool, int(config.sweep["seed"]), stratified)
    if composition["per_volume"]:
        logger.warning(
            "data.composition.per_volume is set, so the volumes target different compositions and are NOT "
            "nested. A volume-to-volume delta then mixes 'more data' with 'different data'."
        )
    logger.info(
        "Train pool: %d rows, %d documents, composition mode '%s'",
        len(pool), pool["document_id"].nunique(), composition["mode"],
    )
    by_document = {str(document): group for document, group in pool.groupby("document_id", sort=False)}

    results: dict[int, dict[str, Path]] = {}
    manifest: dict[str, dict] = {}
    for volume in config.volumes:
        paths = config.subset_split_paths(volume)
        results[volume] = paths
        if paths["train"].exists() and not force and (
            paths["validation"].exists() or config.data["validation_ratio"] == 0
        ):
            logger.info("Reusing subset %s", paths["train"].parent)
            continue
        subset, quotas = _select_subset(config, pool, orders, by_document, volume)
        realized = subset["domain"].value_counts(normalize=True).round(4).to_dict()
        subset_jsonl = _write_sft_jsonl(subset, config, columns, config.subset_dir(volume) / "subset.jsonl")
        manifest[volume_label(volume)] = {
            "requested_rows": volume,
            "rows": len(subset),
            "documents": int(subset["document_id"].nunique()),
            "composition_mode": composition["mode"],
            "target_row_quotas": quotas,
            "realized_domain_shares": realized,
            "domain_rows": subset["domain"].value_counts().to_dict(),
            "subset_path": str(subset_jsonl),
        }
        _split_subset(config, subset_jsonl, volume)
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
