from __future__ import annotations

import math
import re
from collections import Counter
from itertools import combinations
from typing import Any, Callable

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF


DIGIT_TRANSLATION = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
NUMBER_RE = re.compile(r"(?<!\w)[+-]?\d+(?:[.,٫]\d+)*(?:[%٪])?")
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,}\b")
FORMULA_RE = re.compile(r"(?:[A-Za-z]+\d+[A-Za-z0-9]*|[A-Za-z0-9]+\s*[=±×÷<>≤≥]\s*[A-Za-z0-9.+-]+)")


METRIC_DIRECTIONS = {
    "sentence_bleu": "higher",
    "sentence_chrf": "higher",
    "number_preservation": "higher",
    "acronym_preservation": "higher",
    "formula_preservation": "higher",
    "empty_output": "lower",
    "source_copy": "lower",
    "hit_max_new_tokens": "lower",
    "comet": "higher",
    "metricx": "lower",
}


def _normalize_digits(text: str) -> str:
    return text.translate(DIGIT_TRANSLATION).replace("٫", ".").replace("٪", "%")


def _multiset_recall(source_items: list[str], output_items: list[str]) -> float:
    if not source_items:
        return math.nan
    source, output = Counter(source_items), Counter(output_items)
    retained = sum(min(count, output[item]) for item, count in source.items())
    return retained / sum(source.values())


def preservation_score(source: str, translation: str, pattern: re.Pattern[str], normalizer: Callable[[str], str] | None = None) -> float:
    normalize = normalizer or (lambda value: value)
    return _multiset_recall(
        [normalize(match) for match in pattern.findall(source)],
        [normalize(match) for match in pattern.findall(translation)],
    )


def score_transparent_metrics(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    bleu = BLEU(effective_order=True)
    chrf = CHRF(word_order=int(settings.get("chrf_word_order", 2)))
    frame = frame.copy()
    frame["sentence_bleu"] = [bleu.sentence_score(mt, [ref]).score for mt, ref in zip(frame["translation"], frame["reference"])]
    frame["sentence_chrf"] = [chrf.sentence_score(mt, [ref]).score for mt, ref in zip(frame["translation"], frame["reference"])]
    frame["number_preservation"] = [
        preservation_score(src, mt, NUMBER_RE, lambda value: _normalize_digits(value).replace(",", "."))
        for src, mt in zip(frame["source"], frame["translation"])
    ]
    frame["acronym_preservation"] = [
        preservation_score(src, mt, ACRONYM_RE, str.casefold)
        for src, mt in zip(frame["source"], frame["translation"])
    ]
    frame["formula_preservation"] = [
        preservation_score(src, mt, FORMULA_RE, lambda value: re.sub(r"\s+", "", _normalize_digits(value)).casefold())
        for src, mt in zip(frame["source"], frame["translation"])
    ]
    frame["empty_output"] = frame["translation"].str.strip().eq("").astype(float)
    frame["source_copy"] = [float(mt.strip().casefold() == src.strip().casefold()) for src, mt in zip(frame["source"], frame["translation"])]
    return frame


def add_comet_scores(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    import torch
    from comet import download_model, load_from_checkpoint

    checkpoint = download_model(settings.get("model", "Unbabel/wmt22-comet-da"))
    model = load_from_checkpoint(checkpoint)
    data = [{"src": row.source, "mt": row.translation, "ref": row.reference} for row in frame.itertuples()]
    output = model.predict(
        data,
        batch_size=int(settings.get("batch_size", 8)),
        gpus=int(settings.get("gpus", 1)) if torch.cuda.is_available() else 0,
    )
    result = frame.copy()
    result["comet"] = list(output.scores)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def add_metricx_scores(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    import torch
    from metricx24.models import MT5ForRegression
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(settings.get("tokenizer", "google/mt5-xl"))
    model = MT5ForRegression.from_pretrained(
        settings.get("model", "google/metricx-24-hybrid-large-v2p6"), dtype="auto"
    ).to(device).eval()
    scores: list[float] = []
    max_length = int(settings.get("max_length", 1536))
    with torch.inference_mode():
        for row in frame.itertuples():
            text = f"source: {row.source} candidate: {row.translation} reference: {row.reference}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
            inputs = {key: value[:, :-1].to(device) for key, value in inputs.items()}
            scores.append(float(model(**inputs).predictions.item()))
    result = frame.copy()
    result["metricx"] = scores
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def score_candidates(frames: list[pd.DataFrame], metric_config: dict[str, Any]) -> pd.DataFrame:
    transparent = metric_config.get("transparent", {})
    scored_frames = (
        [score_transparent_metrics(frame, transparent) for frame in frames]
        if transparent.get("enabled", True)
        else frames
    )
    scored = pd.concat(scored_frames, ignore_index=True)
    if metric_config.get("comet", {}).get("enabled", False):
        scored = add_comet_scores(scored, metric_config["comet"])
    if metric_config.get("metricx", {}).get("enabled", False):
        scored = add_metricx_scores(scored, metric_config["metricx"])
    return scored


def metric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in METRIC_DIRECTIONS if column in frame.columns]


def summarize_scores(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = metric_columns(frame)
    aggregations = {metric: "mean" for metric in metrics}
    if "latency_seconds" in frame.columns:
        aggregations["latency_seconds"] = "mean"
    if "output_tokens" in frame.columns:
        aggregations["output_tokens"] = "mean"
    group_columns = [
        column for column in ["candidate_id", "candidate_label", "candidate_family", "candidate_size"]
        if column in frame.columns
    ]
    summary = frame.groupby(group_columns, as_index=False, dropna=False).agg(aggregations)
    counts = frame.groupby("candidate_id").size().rename("examples")
    summary["examples"] = summary["candidate_id"].map(counts)
    return summary


def _bootstrap_delta(a: np.ndarray, b: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(a), size=(samples, len(a)))
    deltas = (b[indexes] - a[indexes]).mean(axis=1)
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def pairwise_comparisons(frame: pd.DataFrame, samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate_ids = sorted(frame["candidate_id"].unique())
    for metric in metric_columns(frame):
        if metric not in METRIC_DIRECTIONS:
            continue
        direction = METRIC_DIRECTIONS[metric]
        pivot = frame.pivot(index="example_id", columns="candidate_id", values=metric)
        for candidate_a, candidate_b in combinations(candidate_ids, 2):
            paired = pivot[[candidate_a, candidate_b]].dropna()
            if paired.empty:
                continue
            a, b = paired[candidate_a].to_numpy(float), paired[candidate_b].to_numpy(float)
            raw_delta = b - a
            favorable = raw_delta if direction == "higher" else -raw_delta
            low, high = _bootstrap_delta(a, b, samples, seed)
            rows.append({
                "metric": metric,
                "direction": direction,
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "paired_examples": len(paired),
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "delta_b_minus_a": float(raw_delta.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "b_win_rate": float((favorable > 0).mean()),
                "tie_rate": float((favorable == 0).mean()),
                "a_win_rate": float((favorable < 0).mean()),
                "significant_95": bool(low > 0 or high < 0),
            })
    columns = [
        "metric", "direction", "candidate_a", "candidate_b", "paired_examples",
        "mean_a", "mean_b", "delta_b_minus_a", "ci95_low", "ci95_high",
        "b_win_rate", "tie_rate", "a_win_rate", "significant_95",
    ]
    return pd.DataFrame(rows, columns=columns)


def slice_summary(frame: pd.DataFrame, slices: list[str]) -> pd.DataFrame:
    available = [column for column in slices if column in frame.columns]
    rows: list[pd.DataFrame] = []
    for column in available:
        grouped = frame.groupby(["candidate_id", column], dropna=False)
        part = grouped[metric_columns(frame)].mean().reset_index().rename(columns={column: "slice_value"})
        part.insert(1, "slice", column)
        part["examples"] = grouped.size().to_numpy()
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["candidate_id", "slice", "slice_value", "examples"])
