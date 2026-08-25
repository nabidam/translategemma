#!/usr/bin/env python3
"""Run the repository's SFT pipeline with an optimizer-step cap.

`train.py` already supports a bounded run — `run_pipeline(config, max_steps=N)`
is the path its own `--canary` mode uses — but its CLI exposes that only through
the canary section, which also forces a row cap and its own output directory.
The compute-matched budget needs the step cap and nothing else, so this wrapper
calls the same entry point directly. No training logic lives here: the model,
data, collator, packing and Trainer setup are all `train.py`'s.

    python -m scripts.finetune_benchmark.train_causal_lora \
        --config logs/finetune_benchmark/derived_configs/train_translategemma-5k.yaml \
        --max-steps 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accelerate import PartialState  # noqa: E402

from logging_utils import load_config, log_config_summary, logger, setup_logging  # noqa: E402
from train import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Optimizer-step cap. Omitted, this is exactly `python train.py --config ...`.",
    )
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be a positive integer")

    config = load_config(args.config)
    setup_logging(config)
    if PartialState().is_main_process:
        log_config_summary(config)
    try:
        logger.info(
            "Bounded SFT run: max_steps=%s (epochs configured: %s)",
            args.max_steps or "unset", config["training"]["epochs"],
        )
        run_pipeline(config, max_steps=args.max_steps)
        logger.info("Pipeline complete.")
    except Exception:
        logger.exception("Pipeline failed.")
        raise


if __name__ == "__main__":
    main()
