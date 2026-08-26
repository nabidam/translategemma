#!/usr/bin/env python3
"""Score a benchmark run, with MetricX's cache bug worked around.

`translation_benchmark.metrics.add_metricx_scores` calls the MetricX model with
caching left on. MetricX builds its decoder with `is_encoder_decoder=False`, so
MT5Stack allocates a plain `DynamicCache` rather than an `EncoderDecoderCache`;
T5Attention then appends the cross-attention keys to the same cache as the
decoder's self-attention keys, making `key_length` one longer than the encoder
mask. The forward pass dies in `position_bias + causal_mask` with

    RuntimeError: The size of tensor a (266) must match the size of tensor b (265)

`evaluate_translations.py` passes `use_cache=False` for this reason and carries
the explanation in a comment; the benchmark's own implementation does not. The
decoder runs a single dummy step here, so there is nothing for a cache to
accelerate and disabling it costs nothing.

This module changes no repository file: it replaces the function on the imported
module (the same shim approach as scripts/vllm_rope_shim.py) and then runs the
benchmark's ordinary scoring stage.

    python -m scripts.finetune_benchmark.score \
        --config logs/finetune_benchmark/<run>/derived_configs/benchmark_in_domain.yaml \
        --candidates translategemma-base nllb-5k
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from logging_utils import logger, setup_logging  # noqa: E402
from translation_benchmark import metrics as benchmark_metrics  # noqa: E402
from translation_benchmark.config import load_benchmark_config  # noqa: E402


def _cache_path(output_dir: Path, metric: str) -> Path:
    return output_dir / f".cache_{metric}_scores.csv"


def _row_key(row: Any) -> tuple[str, str, str]:
    """Identity of one scored row: candidate, example, and the text scored.

    The translation's digest is part of the key on purpose. Without it, a
    regenerated candidate would silently inherit the previous run's scores for
    text that no longer exists.
    """
    translation = str(getattr(row, "translation", ""))
    digest = hashlib.sha1(translation.encode("utf-8")).hexdigest()[:16]
    return (
        str(getattr(row, "candidate_id", "")),
        str(getattr(row, "example_id", "")),
        digest,
    )


def _load_cache(path: Path) -> dict[tuple[str, str, str], float]:
    if not path.is_file():
        return {}
    try:
        frame = pd.read_csv(path, dtype={"candidate_id": str, "example_id": str, "digest": str})
    except (pd.errors.EmptyDataError, ValueError):
        return {}
    if not {"candidate_id", "example_id", "digest", "score"}.issubset(frame.columns):
        return {}
    return {
        (row.candidate_id, row.example_id, row.digest): float(row.score)
        for row in frame.itertuples()
    }


def _append_cache(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records, columns=["candidate_id", "example_id", "digest", "score"])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def cached_stage(metric: str, output_dir: Path, compute: Any) -> Any:
    """Wrap a scoring stage so its finished rows survive a later stage's failure.

    `translation_benchmark.metrics.score_candidates` runs transparent metrics,
    then COMET, then MetricX in one function, and `pipeline.score` writes nothing
    until all three return. A MetricX crash therefore discarded a completed
    XCOMET pass over every candidate — hours of GPU time for no artefact. Each
    stage now reads and writes its own cache file beside the run's other output,
    so a re-run recomputes only what is genuinely missing.
    """

    def run_stage(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
        path = _cache_path(output_dir, metric)
        cache = _load_cache(path)
        keys = [_row_key(row) for row in frame.itertuples()]
        pending = [position for position, key in enumerate(keys) if key not in cache]
        if not pending:
            logger.info("%s: all %d rows cached in %s", metric, len(frame), path.name)
            result = frame.copy()
            result[metric] = [cache[key] for key in keys]
            return result
        if len(pending) < len(frame):
            logger.info(
                "%s: %d of %d rows cached; scoring the remaining %d",
                metric, len(frame) - len(pending), len(frame), len(pending),
            )
        subset = frame.iloc[pending].copy()

        def sink(records: list[dict[str, Any]]) -> None:
            _append_cache(path, records)

        scored = compute(subset, settings, sink=sink) if _accepts_sink(compute) else compute(subset, settings)
        if not _accepts_sink(compute):
            _append_cache(path, [
                {"candidate_id": key[0], "example_id": key[1], "digest": key[2], "score": value}
                for key, value in zip((keys[position] for position in pending), scored[metric])
            ])
        cache.update({keys[position]: value for position, value in zip(pending, scored[metric])})
        result = frame.copy()
        result[metric] = [cache[key] for key in keys]
        return result

    return run_stage


def _accepts_sink(function: Any) -> bool:
    import inspect

    try:
        return "sink" in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def _metricx_batches(
    frame: pd.DataFrame, tokenizer: Any, max_length: int, batch_size: int
) -> Any:
    """Yield (indices, encodings) for length-sorted batches of scoring inputs.

    Each row is tokenized on its own so the trailing EOS can be dropped —
    MetricX-24 is trained without it — and only then padded, which is the order
    upstream's own collator uses. Padding to the batch maximum after sorting by
    length keeps the padded waste small; the caller restores the original order.
    """
    texts = [
        f"source: {row.source} candidate: {row.translation} reference: {row.reference}"
        for row in frame.itertuples()
    ]
    encoded = [tokenizer(text, truncation=True, max_length=max_length)["input_ids"][:-1] for text in texts]
    order = sorted(range(len(encoded)), key=lambda index: len(encoded[index]))
    pad_id = tokenizer.pad_token_id or 0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        width = max(len(encoded[index]) for index in indices)
        yield (
            indices,
            [encoded[index] + [pad_id] * (width - len(encoded[index])) for index in indices],
            [[1] * len(encoded[index]) + [0] * (width - len(encoded[index])) for index in indices],
        )


def add_metricx_scores(
    frame: pd.DataFrame, settings: dict[str, Any], sink: Any = None
) -> pd.DataFrame:
    """MetricX-24 segment scores (lower is better), batched.

    The benchmark's implementation scores one example per forward pass
    (`padding=False`, a single-row loop), which `docs/EVALUATION_RUNBOOK.md`
    lists as a known gap: nothing amortises a 3.7B checkpoint over the test set.
    With a batch and an attention mask the same scores come out of far fewer
    forward passes. Set metrics.metricx.batch_size to tune it; 1 reproduces the
    original row-at-a-time behaviour exactly.
    """
    import torch
    from metricx24.models import MT5ForRegression
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(settings.get("tokenizer", "google/mt5-xl"))
    model = MT5ForRegression.from_pretrained(
        settings.get("model", "google/metricx-24-hybrid-large-v2p6"), dtype="auto"
    ).to(device).eval()
    max_length = int(settings.get("max_length", 1536))
    batch_size = max(1, int(settings.get("batch_size", 16)))
    total = len(frame)
    logger.info(
        "MetricX: scoring %d rows on %s with %s (batch %d)",
        total, device, settings.get("model"), batch_size,
    )
    scores = [float("nan")] * total
    rows = list(frame.itertuples())
    done = 0
    with torch.inference_mode():
        for indices, input_ids, attention_mask in _metricx_batches(frame, tokenizer, max_length, batch_size):
            # use_cache=False is required, not an optimisation: see the module
            # docstring. The decoder runs a single dummy step, so there is
            # nothing for a cache to accelerate.
            predictions = model(
                input_ids=torch.tensor(input_ids, device=device),
                attention_mask=torch.tensor(attention_mask, device=device),
                use_cache=False,
            ).predictions
            batch_records = []
            for position, index in enumerate(indices):
                scores[index] = float(predictions[position].item())
                if sink is not None:
                    key = _row_key(rows[index])
                    batch_records.append(
                        {"candidate_id": key[0], "example_id": key[1], "digest": key[2],
                         "score": scores[index]}
                    )
            if sink is not None:
                # Written per batch, not at the end: MetricX is the longest
                # stage, and a failure halfway should not cost the first half.
                sink(batch_records)
            done += len(indices)
            if done % (batch_size * 10) < batch_size or done == total:
                logger.info("MetricX: %d/%d rows", done, total)
    result = frame.copy()
    result["metricx"] = scores
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="A benchmark config, usually a generated one.")
    parser.add_argument("--candidates", nargs="+", default=None)
    args = parser.parse_args()

    config = load_benchmark_config(args.config)
    setup_logging(config.raw, run_name="finetune_benchmark_score")
    metrics_config = config.raw.get("metrics", {})
    if (metrics_config.get("metricx") or {}).get("enabled"):
        benchmark_metrics.add_metricx_scores = cached_stage(
            "metricx", config.output_dir, add_metricx_scores
        )
        logger.info("MetricX: use_cache=False, batched, and cached per row.")
    if (metrics_config.get("comet") or {}).get("enabled"):
        # Not patched for behaviour, only wrapped for caching: a completed COMET
        # pass must survive a MetricX failure.
        benchmark_metrics.add_comet_scores = cached_stage(
            "comet", config.output_dir, benchmark_metrics.add_comet_scores
        )
        logger.info("COMET: cached per row.")

    # Imported after the patch so the module-level lookup inside
    # score_candidates resolves to the replacement.
    from translation_benchmark.pipeline import score

    paths = score(config, args.candidates)
    logger.info("Scores written to %s", config.output_dir)
    for name, path in paths.items():
        logger.info("  %-12s %s", name, path)


if __name__ == "__main__":
    main()
