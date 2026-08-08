#!/usr/bin/env python3
"""
build_test_set.py
=================
Builds a stratified, diversity-maximized, contamination-safe MT test set
from preprocessed_translations.csv (columns: id, en, fa, domain).

Pipeline stages (each configurable via config.yaml):
  1. Load + validate
  2. Feature annotation (length buckets, math, numbers/units, acronyms,
     mixed script, rare-term novelty score)
  3. Deduplication (exact -> MinHash near-dup -> embedding near-dup)
  4. Embeddings (LaBSE via sentence-transformers, TF-IDF fallback)
  5. Quota allocation (domain x length-bucket, proportional/flattened)
 5b. Candidate-document restriction (`selection.max_test_documents`, see below)
  6. Per-cell k-center greedy (farthest-point) selection
  7. Hard-phenomena minimum-quota top-up
  8. Document-level holdout + embedding near-dup purge of the train pool
  9. Optional stratified dev/test split
 10. Gold subset flagging for human verification
 11. Report + frozen MANIFEST with sha256 hashes

--------------------------------------------------------------------------
`selection.max_test_documents`  (int or null, default: null = disabled)
--------------------------------------------------------------------------
WHY IT EXISTS
  Stage 8 holds out data at the DOCUMENT level: every document that
  contributes even one test/dev row has ALL of its sibling chunks removed
  from the train pool (training on siblings would leak author style,
  terminology, and topic into evaluation). Meanwhile the k-center
  diversity selector naturally spreads its picks across as many documents
  as possible. When the corpus has FEW, LARGE documents (a LOW
  document-count / HIGH chunks-per-document ratio — e.g. theses with
  ~200 chunks each), these two behaviors interact badly: 1,000 selected
  rows can touch hundreds of documents and quarantine 80%+ of the corpus.

WHAT IT DOES
  When set, stage 5b pre-selects at most `max_test_documents` documents
  (stratified across domains, preferring documents whose chunks cover all
  length buckets and carry hard-phenomenon flags, so downstream quotas
  remain satisfiable). Test/dev selection then runs ONLY inside those
  documents; every other document stays fully available for training.
  The train-pool cost of the test set becomes bounded and predictable:
      rows quarantined <= max_test_documents * avg_chunks_per_document.

WHEN TO SET IT / WHEN TO LEAVE IT OFF
  * SET it (e.g. 25-50) when chunks-per-document is HIGH (few fat
    documents, ratio >~ 50): the holdout cost dominates and you must cap
    it to keep the train pool useful. Trade-off: test rows become more
    correlated (shared authors/topics), so the effective sample size is
    lower than the nominal row count — an accepted cost here.
  * LEAVE it null when chunks-per-document is LOW (many small documents,
    ratio <~ 10): the holdout cost is naturally small, and unrestricted
    selection gives a more independent, more representative test set.
  Rule of thumb: estimate rows_lost ~= expected_test_documents * ratio;
  if that is more than ~15-20% of the corpus, set the cap.

Usage:
    python build_test_set.py --config config.yaml
Dependencies:
    pip install pandas numpy pyyaml scikit-learn
    optional (recommended): pip install sentence-transformers datasketch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from logging_utils import console, log_config_summary, logger, setup_logging

RNG = None  # seeded numpy Generator, set in main()


def _make_progress(description) -> Progress:
    """Rich progress bar with consistent columns, bound to project console."""
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]{description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


# ----------------------------------------------------------------------
# 1. Load + validate
# ----------------------------------------------------------------------
def load_corpus(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["input"]["csv_path"])
    if not path.exists():
        logger.error("Input file not found: %s", path)
        sys.exit(1)
    df = pd.read_csv(path, dtype=str)
    required = {"id", "en", "fa", "domain"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Missing required columns: %s", missing)
        sys.exit(1)

    n0 = len(df)
    df = df.dropna(subset=["id", "en", "fa", "domain"]).copy()
    df["en"] = df["en"].str.strip()
    df["fa"] = df["fa"].str.strip()
    df = df[(df["en"] != "") & (df["fa"] != "")]
    sep = cfg["input"].get("id_separator", ":")
    df["document_id"] = df["id"].str.split(sep).str[0]
    df = df.reset_index(drop=True)
    logger.info(
        "Loaded %d rows (%d dropped as empty/NaN), %d documents, %d domains",
        len(df),
        n0 - len(df),
        df["document_id"].nunique(),
        df["domain"].nunique(),
    )
    return df


# ----------------------------------------------------------------------
# 2. Feature annotation
# ----------------------------------------------------------------------
MATH_RE = re.compile(
    r"(\$[^$]+\$|\\\(|\\\[|\\frac|\\sum|\\int|\\alpha|\\beta|\\gamma|\\lambda|\\sigma"
    r"|[=<>≤≥±∓×÷√∞∈∉⊂⊆∪∩∑∏∫∂∇Δ∆]"
    r"|\b[a-zA-Z]\s*\^\s*[0-9n]|\b[xyz]\s*=\s*)"
)
NUM_UNIT_RE = re.compile(
    r"\b\d+(\.\d+)?\s*(%|mm|cm|km|kg|mg|g|ml|l|s|ms|hz|khz|mhz|ghz|k|°c|°f|kpa|mpa"
    r"|mol|ppm|db|nm|µm|um|ev|kev|mev|gev|w|kw|mw|v|mv|a|ma)\b",
    re.IGNORECASE,
)
STAT_RE = re.compile(
    r"\b[pP]\s*[<>=]\s*0?\.\d+|\bn\s*=\s*\d+|\bCI\b|\br\s*=\s*[-0.]|\bF\(\d"
)
ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}s?\b")
PERSIAN_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())


def annotate_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    src_col = cfg["input"]["source_lang_col"]
    tgt_col = cfg["input"]["target_lang_col"]
    sel = cfg["selection"]

    logger.info("Annotating features on source column '%s' ...", src_col)
    src = df[src_col].fillna("")
    tgt = df[tgt_col].fillna("")

    df["n_tokens"] = src.map(lambda t: len(tokenize(t)))

    edges = sel["length_buckets"]
    labels = [f"len_{edges[i]}_{edges[i+1]}" for i in range(len(edges) - 1)]
    df["length_bucket"] = pd.cut(
        df["n_tokens"], bins=edges, labels=labels, right=False, include_lowest=True
    ).astype(str)

    df["has_math"] = src.map(lambda t: bool(MATH_RE.search(t)))
    df["has_numbers_units"] = src.map(
        lambda t: bool(NUM_UNIT_RE.search(t)) or bool(STAT_RE.search(t))
    )
    df["has_acronyms"] = src.map(lambda t: bool(ACRONYM_RE.search(t)))
    # mixed script: Latin embedded in the Persian side, or Persian in the English side
    df["has_mixed_script"] = (
        [
            bool(LATIN_CHAR_RE.search(f)) or bool(PERSIAN_CHAR_RE.search(e))
            for e, f in zip(src, tgt)
        ]
        if src_col == "en"
        else [
            bool(PERSIAN_CHAR_RE.search(e)) or bool(LATIN_CHAR_RE.search(f))
            for e, f in zip(src, tgt)
        ]
    )

    # --- rare-term novelty score -------------------------------------
    max_count = cfg["features"]["rare_token_max_count"]
    logger.info("Computing corpus token frequencies for rare-term score ...")
    counter: Counter = Counter()
    token_lists = src.map(tokenize)
    for toks in token_lists:
        counter.update(set(toks))  # document frequency, robust to repetition

    def rare_score(toks: list[str]) -> float:
        if not toks:
            return 0.0
        rare = sum(1 for t in toks if counter[t] <= max_count)
        return rare / len(toks)

    df["rare_term_score"] = token_lists.map(rare_score)
    q = sel["rare_term_top_quantile"]
    threshold = df["rare_term_score"].quantile(q)
    df["is_rare_term"] = df["rare_term_score"] >= threshold

    for col in [
        "has_math",
        "has_numbers_units",
        "has_acronyms",
        "has_mixed_script",
        "is_rare_term",
    ]:
        logger.info(
            "  %-18s : %6d rows (%.1f%%)", col, df[col].sum(), 100 * df[col].mean()
        )
    return df


# ----------------------------------------------------------------------
# 3. Deduplication
# ----------------------------------------------------------------------
def _shingles(text: str, k: int = 3) -> set[str]:
    toks = tokenize(text)
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def dedup_exact(df: pd.DataFrame, src_col: str) -> pd.DataFrame:
    # Use fast vectorized lowercase and space normalization
    norm = df[src_col].str.lower().str.replace(r"\s+", " ", regex=True)
    keep = ~norm.duplicated(keep="first")
    logger.info("Exact dedup: removed %d rows", (~keep).sum())
    return df[keep].reset_index(drop=True)

from concurrent.futures import ProcessPoolExecutor

def _compute_minhash(text: str):
    from datasketch import MinHash
    m = MinHash(num_perm=128)
    for sh in _shingles(text):
        m.update(sh.encode("utf-8"))
    return m

def dedup_minhash(df: pd.DataFrame, src_col: str, threshold: float) -> pd.DataFrame:
    try:
        from datasketch import MinHashLSH
    except ImportError:
        logger.warning("datasketch not installed — skipping MinHash near-dup pass.")
        return df

    logger.info("MinHash near-dup pass (threshold=%.2f) ...", threshold)
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    drop = set()

    texts = df[src_col].tolist()
    
    # Process MinHashes across CPU cores in chunks
    with ProcessPoolExecutor() as executor:
        minhashes = list(executor.map(_compute_minhash, texts, chunksize=10000))

    for idx, m in enumerate(minhashes):
        if lsh.query(m):
            drop.add(idx)
        else:
            lsh.insert(str(idx), m)

    logger.info("MinHash dedup: removed %d rows", len(drop))
    return df.drop(index=drop).reset_index(drop=True)

def dedup_embeddings(
    df: pd.DataFrame, emb: np.ndarray, threshold: float, batch_size: int = 10000
) -> tuple[pd.DataFrame, np.ndarray]:
    logger.info("Embedding near-dup pass (cosine >= %.2f) ...", threshold)
    
    n = len(df)
    drop: set[int] = set()
    with _make_progress("Generating drop list") as prog:
        task = prog.add_task("drop item", total=len(range(0, n, batch_size)))
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            # Batch dot product against preceding embeddings
            sims = emb[start:end] @ emb[:end].T
            
            for i_local in range(end - start):
                i_global = start + i_local
                if i_global in drop:
                    continue
                # Find elements in preceding embeddings above threshold
                matches = np.where(sims[i_local, :i_global] >= threshold)[0]
                if len(matches) > 0:
                    drop.add(i_global)
            prog.advance(task)

    keep_mask = np.array([i not in drop for i in range(n)])
    logger.info("Embedding dedup: removed %d rows", len(drop))
    return df[keep_mask].reset_index(drop=True), emb[keep_mask]


# ----------------------------------------------------------------------
# 4. Embeddings (LaBSE with TF-IDF fallback)
# ----------------------------------------------------------------------
def compute_embeddings(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    src_col = cfg["input"]["source_lang_col"]
    texts = df[src_col].tolist()
    ecfg = cfg["embeddings"]

    cache = Path(ecfg.get("cache_path", "")) if ecfg.get("cache_path") else None

    hasher = hashlib.sha256()
    for text in texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\n")
    data_hash = hasher.hexdigest()[:16]

    if cache:
        meta = cache.with_suffix(".json")
        if cache.exists() and meta.exists():
            try:
                if json.loads(meta.read_text())["data_hash"] == data_hash:
                    logger.info("Loading cached embeddings from %s", cache)
                    return np.load(cache)
            except Exception:
                pass

    if ecfg.get("enabled", True):
        try:
            from sentence_transformers import SentenceTransformer

            device = ecfg.get("device", "auto")
            if device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            logger.info(
                "Encoding %d texts with %s on %s ...", len(texts), ecfg["model"], device
            )
            model = SentenceTransformer(ecfg["model"], device=device)
            emb = model.encode(
                texts,
                batch_size=ecfg.get("batch_size", 64),
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            emb = np.asarray(emb, dtype=np.float32)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — falling back to TF-IDF "
                "(diversity selection still works, just weaker semantics)."
            )
            emb = _tfidf_embeddings(texts)
    else:
        emb = _tfidf_embeddings(texts)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        cache.with_suffix(".json").write_text(json.dumps({"data_hash": data_hash}))
    return emb


def _tfidf_embeddings(texts: list[str]) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    logger.info("Computing TF-IDF + SVD embeddings (fallback) ...")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=50000)
    X = vec.fit_transform(texts)
    dim = min(256, X.shape[1] - 1, len(texts) - 1)
    emb = TruncatedSVD(n_components=dim, random_state=0).fit_transform(X)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (emb / norms).astype(np.float32)


# ----------------------------------------------------------------------
# 5. Quota allocation (domain x length bucket)
# ----------------------------------------------------------------------
def allocate_quotas(df: pd.DataFrame, cfg: dict) -> dict[tuple[str, str], int]:
    sel = cfg["selection"]
    total = sel["total_size"]
    domains = df["domain"].value_counts()

    # --- domain-level targets ---
    if sel["domain_allocation"] == "proportional":
        dom_share = domains / domains.sum()
    else:  # flattened
        alpha = float(sel.get("flatten_alpha", 0.5))
        prop = domains / domains.sum()
        unif = pd.Series(1.0 / len(domains), index=domains.index)
        dom_share = (1 - alpha) * prop + alpha * unif
        dom_share = dom_share / dom_share.sum()

    dom_target = (dom_share * total).round().astype(int)
    min_dom = sel.get("min_per_domain", 0)
    for d in dom_target.index:
        avail = int(domains[d])
        want = max(dom_target[d], min_dom)
        if avail < want:
            logger.warning(
                "Domain '%s' has only %d rows (< target %d) — taking all of them.",
                d,
                avail,
                want,
            )
            want = avail
        dom_target[d] = want
    # renormalize to hit total exactly (trim/pad the largest domains)
    diff = total - int(dom_target.sum())
    order = dom_target.sort_values(ascending=False).index.tolist()
    i = 0
    while diff != 0 and order:
        d = order[i % len(order)]
        step = 1 if diff > 0 else -1
        cap = int(domains[d])
        if 0 < dom_target[d] + step <= cap:
            dom_target[d] += step
            diff -= step
        i += 1
        if i > 10000:
            break

    # --- split each domain target across length buckets ---
    edges = sel["length_buckets"]
    bucket_labels = [f"len_{edges[i]}_{edges[i+1]}" for i in range(len(edges) - 1)]
    shares = sel["length_bucket_shares"]
    assert abs(sum(shares) - 1.0) < 1e-6, "length_bucket_shares must sum to 1.0"

    quotas: dict[tuple[str, str], int] = {}
    for d, dt in dom_target.items():
        avail_per_bucket = df[df["domain"] == d]["length_bucket"].value_counts()
        want = {b: int(round(dt * s)) for b, s in zip(bucket_labels, shares)}
        # cap by availability, redistribute surplus to buckets with room
        for b in bucket_labels:
            want[b] = min(want[b], int(avail_per_bucket.get(b, 0)))
        deficit = dt - sum(want.values())
        while deficit > 0:
            progressed = False
            for b in bucket_labels:
                room = int(avail_per_bucket.get(b, 0)) - want[b]
                if room > 0 and deficit > 0:
                    want[b] += 1
                    deficit -= 1
                    progressed = True
            if not progressed:
                break
        for b in bucket_labels:
            if want[b] > 0:
                quotas[(d, b)] = want[b]

    logger.info(
        "Allocated quotas over %d (domain x length) cells, total = %d",
        len(quotas),
        sum(quotas.values()),
    )
    return quotas


# ----------------------------------------------------------------------
# 5b. Candidate-document restriction (max_test_documents)
# ----------------------------------------------------------------------
def restrict_candidate_documents(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Cap how many documents the test/dev sets may draw from.

    Returns a VIEW of `df` (original integer indices preserved, so it stays
    aligned with the embedding matrix) containing only rows from the chosen
    candidate documents. If `selection.max_test_documents` is null/absent,
    returns `df` unchanged.

    Document choice is stratified by domain (slots proportional to each
    domain's share of the corpus, min 1 per domain) and scored so that the
    downstream domain x length quotas and hard-phenomena quotas remain
    satisfiable: documents earn points for covering more length buckets,
    carrying hard-phenomenon rows, and having enough chunks to fill cells.
    """
    max_docs = cfg["selection"].get("max_test_documents")
    if not max_docs:
        return df

    n_docs_total = df["document_id"].nunique()
    ratio = len(df) / max(n_docs_total, 1)
    if max_docs >= n_docs_total:
        logger.warning(
            "max_test_documents=%d >= total documents (%d) — restriction has "
            "no effect.",
            max_docs,
            n_docs_total,
        )
        return df
    logger.info(
        "Restricting test/dev candidates to %d of %d documents "
        "(~%.0f chunks/document -> holdout cost bounded at ~%d rows)",
        max_docs,
        n_docs_total,
        ratio,
        int(max_docs * ratio),
    )

    edges = cfg["selection"]["length_buckets"]
    n_buckets = len(edges) - 1
    flag_cols = [
        c
        for c in (
            "has_math",
            "has_numbers_units",
            "has_acronyms",
            "has_mixed_script",
            "is_rare_term",
        )
        if c in df.columns
    ]

    # --- score every document ------------------------------------------
    grp = df.groupby("document_id", observed=True)
    agg_dict = {
        "domain": "first",
        "length_bucket": "nunique",
        "n_chunks": ("domain", "size")
    }
    for c in flag_cols:
        agg_dict[f"share_{c}"] = (c, "mean")

    doc_stats = df.groupby("document_id", observed=True).agg(**{
        "domain": ("domain", "first"),
        "n_buckets": ("length_bucket", "nunique"),
        "n_chunks": ("domain", "size"),
        **{f"share_{c}": (c, "mean") for c in flag_cols}
    })
    for c in flag_cols:
        doc_stats[f"share_{c}"] = grp[c].mean()

    # coverage score: bucket coverage dominates, then hard-flag density,
    # then log-size (a doc must have enough chunks to fill its cells);
    # tiny seeded jitter breaks ties deterministically.
    score = (
        doc_stats["n_buckets"] / n_buckets * 2.0
        + sum(doc_stats[f"share_{c}"] for c in flag_cols) / max(len(flag_cols), 1)
        + np.log1p(doc_stats["n_chunks"]) / np.log1p(doc_stats["n_chunks"].max())
    )
    doc_stats["score"] = score + RNG.random(len(doc_stats)) * 1e-6

    # --- allocate document slots per domain (proportional, min 1) ------
    dom_counts = df["domain"].value_counts()
    raw = (dom_counts / dom_counts.sum() * max_docs).round().astype(int)
    slots = raw.clip(lower=1)
    # trim overshoot from the largest allocations
    while slots.sum() > max_docs:
        slots[slots.idxmax()] -= 1
    # give leftover slots to domains that still have unselected documents
    docs_per_domain = doc_stats["domain"].value_counts()
    while slots.sum() < max_docs:
        room = (docs_per_domain - slots).sort_values(ascending=False)
        if room.iloc[0] <= 0:
            break
        slots[room.index[0]] += 1

    chosen: list[str] = []
    for dom, k in slots.items():
        dom_docs = doc_stats[doc_stats["domain"] == dom]
        k = min(int(k), len(dom_docs))
        chosen.extend(dom_docs.nlargest(k, "score").index.tolist())

    cand = df[df["document_id"].isin(chosen)]
    logger.info(
        "Candidate pool: %d rows from %d documents (%s)",
        len(cand),
        len(chosen),
        ", ".join(f"{d}={int(k)}" for d, k in slots.items()),
    )
    if len(cand) < cfg["selection"]["total_size"] * 3:
        logger.warning(
            "Candidate pool is only %.1fx total_size — quotas may be hard to "
            "fill. Consider raising max_test_documents.",
            len(cand) / cfg["selection"]["total_size"],
        )
    return cand


# ----------------------------------------------------------------------
# 6. Per-cell k-center greedy selection
# ----------------------------------------------------------------------
def kcenter_greedy(emb: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """Farthest-point sampling on normalized embeddings (cosine distance)."""
    n = emb.shape[0]
    if k >= n:
        return list(range(n))
    start = int(rng.integers(n))
    selected = [start]
    # cosine distance = 1 - dot (embeddings are L2-normalized)
    min_dist = 1.0 - emb @ emb[start]
    for _ in range(k - 1):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, 1.0 - emb @ emb[nxt])
    return selected


def select_stratified(
    df: pd.DataFrame, emb: np.ndarray, quotas: dict[tuple[str, str], int], cfg: dict
) -> list[int]:
    strategy = cfg["diversity"]["strategy"]
    selected: list[int] = []
    with _make_progress("Selecting Stratified") as prog:
        task = prog.add_task("stratified select", total=len(quotas))
        for (dom, bucket), k in quotas.items():
            cell_idx = df.index[
                (df["domain"] == dom) & (df["length_bucket"] == bucket)
            ].to_numpy()
            if len(cell_idx) == 0:
                prog.advance(task)
                continue
            if strategy == "kcenter":
                local = kcenter_greedy(emb[cell_idx], k, RNG)
                picks = cell_idx[local]
            else:
                picks = RNG.choice(cell_idx, size=min(k, len(cell_idx)), replace=False)
            selected.extend(int(i) for i in picks)
            prog.advance(task)
    logger.info("Stratified selection: %d rows", len(selected))
    return selected


# ----------------------------------------------------------------------
# 7. Hard-phenomena minimum-quota top-up
# ----------------------------------------------------------------------
def enforce_hard_quotas(
    df: pd.DataFrame, emb: np.ndarray, selected: list[int], cfg: dict
) -> list[int]:
    sel_cfg = cfg["selection"]
    total = sel_cfg["total_size"]
    quotas = dict(sel_cfg.get("hard_phenomena_min_share", {}))
    quotas["is_rare_term"] = sel_cfg.get("rare_term_min_share", 0.0)

    selected_set = set(selected)
    with _make_progress("Enforce Hard Quotas") as prog:
        task = prog.add_task("hard quotas", total=len(quotas))
        for flag, share in quotas.items():
            need = int(np.ceil(share * total))
            have = int(df.loc[list(selected_set), flag].sum())
            if have >= need:
                logger.info("Quota %-18s ok: %d / %d", flag, have, need)
                prog.advance(task)
                continue
            deficit = need - have
            candidates = df.index[df[flag] & ~df.index.isin(selected_set)].to_numpy()
            if len(candidates) == 0:
                logger.warning(
                    "Quota %s unmet (%d/%d): no candidates left in corpus.",
                    flag,
                    have,
                    need,
                )
                prog.advance(task)
                continue
            deficit = min(deficit, len(candidates))
            # pick the candidates farthest (in embedding space) from current selection
            sel_arr = np.fromiter(selected_set, dtype=int)
            sims = emb[candidates] @ emb[sel_arr].T  # (cand, sel)
            farthest = candidates[np.argsort(sims.max(axis=1))[:deficit]]

            # swap out rows that DON'T carry any hard flag, preferring the most
            # redundant (highest max-similarity to the rest of the selection)
            flag_cols = [c for c in quotas if c in df.columns]
            no_flag = [i for i in selected_set if not df.loc[i, flag_cols].any()]
            if len(no_flag) < len(farthest):
                no_flag += [i for i in selected_set if i not in no_flag]
            no_flag_arr = np.array(no_flag[: max(len(farthest) * 3, len(farthest))])
            red_sims = emb[no_flag_arr] @ emb[sel_arr].T
            np.fill_diagonal(red_sims[:, :0], 0)  # no-op guard
            # exclude self-similarity by masking positions where index matches
            for r, i in enumerate(no_flag_arr):
                self_pos = np.where(sel_arr == i)[0]
                red_sims[r, self_pos] = -1
            order = no_flag_arr[np.argsort(-red_sims.max(axis=1))][: len(farthest)]

            for out_i, in_i in zip(order, farthest):
                selected_set.discard(int(out_i))
                selected_set.add(int(in_i))
            logger.info(
                "Quota %-18s topped up: %d -> %d (swapped %d rows)",
                flag,
                have,
                need,
                len(farthest),
            )
            prog.advance(task)
    return sorted(selected_set)


# ----------------------------------------------------------------------
# 8. Contamination: document-level holdout + train-pool purge
# ----------------------------------------------------------------------
def contamination_pass(
    df: pd.DataFrame, emb: np.ndarray, selected: list[int], cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ccfg = cfg["contamination"]
    sel_mask = df.index.isin(selected)
    test_df = df[sel_mask].copy()
    pool_df = df[~sel_mask].copy()

    if ccfg.get("document_level_holdout", True):
        held_docs = set(test_df["document_id"])
        before = len(pool_df)
        pool_df = pool_df[~pool_df["document_id"].isin(held_docs)]
        logger.info(
            "Document-level holdout: removed %d sibling-chunk rows from "
            "train pool (%d held-out documents)",
            before - len(pool_df),
            len(held_docs),
        )

    if ccfg.get("near_dup_check_enabled", True) and len(pool_df) and len(test_df):
        thr = float(ccfg["near_dup_cosine_threshold"])
        test_emb = emb[test_df.index.to_numpy()]
        pool_idx = pool_df.index.to_numpy()
        drop = []
        chunk = 2048
        with _make_progress("Contamination Pass") as prog:
            task = prog.add_task("near-dup purge", total=len(pool_idx))
            for s in range(0, len(pool_idx), chunk):
                block = pool_idx[s : s + chunk]
                sims = emb[block] @ test_emb.T
                bad = block[sims.max(axis=1) >= thr]
                drop.extend(int(b) for b in bad)
                prog.update(task, advance=len(block))
        pool_df = pool_df.drop(index=drop)
        logger.info(
            "Cross-document near-dup purge: removed %d leaky rows from "
            "train pool (cosine >= %.2f)",
            len(drop),
            thr,
        )

    return test_df.reset_index(drop=True), pool_df.reset_index(drop=True)


# ----------------------------------------------------------------------
# 9 + 10. Dev/test split and gold subset
# ----------------------------------------------------------------------
def split_dev_test(
    test_df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    scfg = cfg["splits"]
    if not scfg.get("dev_test_split", False):
        test_df["split"] = "test"
        return test_df, None
    frac = float(scfg["dev_fraction"])
    dev_rows = []
    for _, grp in test_df.groupby(["domain", "length_bucket"], observed=True):
        k = int(round(len(grp) * frac))
        if k > 0:
            dev_rows.extend(
                RNG.choice(grp.index.to_numpy(), size=k, replace=False).tolist()
            )
    dev_mask = test_df.index.isin(dev_rows)
    dev_df = test_df[dev_mask].copy()
    dev_df["split"] = "dev"
    tst_df = test_df[~dev_mask].copy()
    tst_df["split"] = "test"
    logger.info(
        "Split: %d dev / %d test (stratified by domain x length)",
        len(dev_df),
        len(tst_df),
    )
    return tst_df.reset_index(drop=True), dev_df.reset_index(drop=True)


def flag_gold_subset(frames: list[pd.DataFrame], cfg: dict) -> None:
    gcfg = cfg["gold_subset"]
    for f in frames:
        if f is not None:
            f["human_verify"] = False
    if not gcfg.get("enabled", False):
        return
    size = int(gcfg["size"])
    all_df = pd.concat([f for f in frames if f is not None])
    per_domain = max(1, size // all_df["domain"].nunique())
    chosen: list = []
    for _, grp in all_df.groupby("domain", observed=True):
        k = min(per_domain, len(grp))
        chosen.extend(RNG.choice(grp["id"].to_numpy(), size=k, replace=False))
    chosen = set(chosen[:size])
    for f in frames:
        if f is not None:
            f.loc[f["id"].isin(chosen), "human_verify"] = True
    logger.info("Gold subset: flagged %d rows human_verify=true", len(chosen))


# ----------------------------------------------------------------------
# 11. Report + manifest
# ----------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_outputs(test_df, dev_df, pool_df, cfg: dict) -> None:
    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        "id",
        "document_id",
        "en",
        "fa",
        "domain",
        "n_tokens",
        "length_bucket",
        "has_math",
        "has_numbers_units",
        "has_acronyms",
        "has_mixed_script",
        "rare_term_score",
        "is_rare_term",
        "split",
        "human_verify",
    ]

    paths = {}
    test_path = out / cfg["output"]["test_file"]
    test_df[keep_cols].to_csv(test_path, index=False)
    paths["test"] = test_path
    if dev_df is not None:
        dev_path = out / cfg["output"]["dev_file"]
        dev_df[keep_cols].to_csv(dev_path, index=False)
        paths["dev"] = dev_path
    pool_path = out / cfg["output"]["train_pool_file"]
    pool_df[["id", "document_id", "en", "fa", "domain"]].to_csv(pool_path, index=False)
    paths["train_pool"] = pool_path

    # report
    def summarize(f: pd.DataFrame) -> dict:
        return {
            "n_rows": len(f),
            "domains": f["domain"].value_counts().to_dict(),
            "length_buckets": f["length_bucket"].value_counts().to_dict(),
            "flags": {
                c: int(f[c].sum())
                for c in [
                    "has_math",
                    "has_numbers_units",
                    "has_acronyms",
                    "has_mixed_script",
                    "is_rare_term",
                ]
            },
            "human_verify": int(f["human_verify"].sum()),
            "n_documents": f["document_id"].nunique(),
        }

    report = {"config": cfg, "test": summarize(test_df)}
    if dev_df is not None:
        report["dev"] = summarize(dev_df)
    report["train_pool_rows"] = len(pool_df)
    report_path = out / cfg["output"]["report_file"]
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    manifest = {
        name: {"path": str(p), "sha256": sha256_file(p)} for name, p in paths.items()
    }
    (out / cfg["output"]["manifest_file"]).write_text(json.dumps(manifest, indent=2))
    logger.info(
        "Wrote outputs to %s (see %s). Freeze the manifest hashes — if a "
        "hash ever changes mid-project, your comparisons are void.",
        out,
        cfg["output"]["manifest_file"],
    )


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    global RNG
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="testset_config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    setup_logging(cfg, run_name="build_test_set")
    log_config_summary(cfg)
    logger.info("Building test set from config: %s", args.config)
    RNG = np.random.default_rng(cfg["selection"]["seed"])
    src_col = cfg["input"]["source_lang_col"]

    df = load_corpus(cfg)
    df = annotate_features(df, cfg)

    # dedup: exact + minhash happen pre-embedding (cheaper corpus to encode)
    if cfg["dedup"]["enabled"]:
        df = dedup_exact(df, src_col)
        if cfg["dedup"].get("minhash_enabled", True):
            df = dedup_minhash(df, src_col, cfg["dedup"]["minhash_threshold"])

    emb = compute_embeddings(df, cfg)

    if cfg["dedup"]["enabled"] and cfg["dedup"].get("embedding_dedup_enabled", True):
        df, emb = dedup_embeddings(df, emb, cfg["dedup"]["embedding_cosine_threshold"])

    df = df.reset_index(drop=True)

    # 5b: optionally cap the documents test/dev may draw from; cand_df is a
    # VIEW with original indices (stays aligned with emb). Quota allocation,
    # selection, and the hard-quota top-up all operate inside it, so every
    # selected row — including top-up swaps — respects the document cap.
    # The contamination pass still runs on the FULL frame: non-candidate
    # documents flow untouched into the train pool.
    cand_df = restrict_candidate_documents(df, cfg)
    quotas = allocate_quotas(cand_df, cfg)
    selected = select_stratified(cand_df, emb, quotas, cfg)
    selected = enforce_hard_quotas(cand_df, emb, selected, cfg)
    test_all, pool_df = contamination_pass(df, emb, selected, cfg)

    test_df, dev_df = split_dev_test(test_all, cfg)
    flag_gold_subset([test_df, dev_df], cfg)
    write_outputs(test_df, dev_df, pool_df, cfg)


if __name__ == "__main__":
    main()
