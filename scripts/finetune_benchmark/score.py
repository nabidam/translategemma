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


def add_metricx_scores(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
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
            for position, index in enumerate(indices):
                scores[index] = float(predictions[position].item())
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
    if (config.raw.get("metrics", {}).get("metricx") or {}).get("enabled"):
        benchmark_metrics.add_metricx_scores = add_metricx_scores
        logger.info("MetricX scoring patched to run with use_cache=False.")

    # Imported after the patch so the module-level lookup inside
    # score_candidates resolves to the replacement.
    from translation_benchmark.pipeline import score

    paths = score(config, args.candidates)
    logger.info("Scores written to %s", config.output_dir)
    for name, path in paths.items():
        logger.info("  %-12s %s", name, path)


if __name__ == "__main__":
    main()
