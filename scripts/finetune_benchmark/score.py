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


def add_metricx_scores(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """MetricX-24 segment scores (lower is better), one forward pass per row."""
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
    total = len(frame)
    logger.info("MetricX: scoring %d rows on %s with %s", total, device, settings.get("model"))
    with torch.inference_mode():
        for position, row in enumerate(frame.itertuples(), start=1):
            text = f"source: {row.source} candidate: {row.translation} reference: {row.reference}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
            # MetricX-24 is trained without the trailing EOS token.
            inputs = {key: value[:, :-1].to(device) for key, value in inputs.items()}
            # The one-line difference from the benchmark's implementation. See
            # this module's docstring for why it is required rather than an
            # optimisation.
            scores.append(float(model(**inputs, use_cache=False).predictions.item()))
            if position % 250 == 0 or position == total:
                logger.info("MetricX: %d/%d rows", position, total)
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
