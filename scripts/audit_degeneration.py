"""Classify decoding failures in evaluate_translations.py output.

MetricX and COMET are averages; they hide the failure that dominated the
2026-08-10 adapter run, where a correct translation was followed by a thousand
newlines. This script counts the failures directly, per system, from the
``<prefix>_detailed_scores.csv`` files the evaluator already writes, so a
regression in termination behaviour is a number rather than a browsing session.

    uv run python scripts/audit_degeneration.py --eval-dir evaluation
    uv run python scripts/audit_degeneration.py --fail-over 0.05

Failure classes:

whitespace_flood   Output ends in a long run of whitespace. The model produced
                   the translation and then filled the budget instead of
                   emitting the stop token.
loop               A word n-gram repeats N+ times in the trimmed output.
boilerplate        Translator/watermark text leaked from the training corpus.
length_blowup      Trimmed output is far longer than the reference.
near_budget        Output is close enough to max_new_tokens (approximated by
                   characters) that it was probably cut off rather than ended.
empty              Blank output.

clean is the share of rows hitting none of the above; it is the headline number.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Run from anywhere: the project modules live in the repository root, not here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from degeneration import DEFAULTS, audit_outputs  # noqa: E402
from logging_utils import load_config, logger  # noqa: E402

HYPOTHESIS_COLUMN = "generated_farsi"


def audit_frame(frame, target_column, settings, sample_limit):
    """Run the shared classifier over one system's detailed_scores.csv."""
    if HYPOTHESIS_COLUMN not in frame.columns:
        raise ValueError(f"frame has no '{HYPOTHESIS_COLUMN}' column")
    references = (
        frame[target_column] if target_column in frame.columns else [""] * len(frame)
    )
    return audit_outputs(frame[HYPOTHESIS_COLUMN], references, settings, sample_limit)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--eval-dir", default=None, help="Defaults to evaluation.output_dir.")
    parser.add_argument("--report", default=None, help="Defaults to <eval-dir>/degeneration_audit.json.")
    parser.add_argument("--samples", type=int, default=10)
    # Thresholds default to degeneration.DEFAULTS so the offline report and the
    # in-run gate classify identically unless deliberately overridden.
    parser.add_argument("--trailing-threshold", type=int, default=DEFAULTS["trailing_threshold"])
    parser.add_argument("--repeat-ngram", type=int, default=DEFAULTS["repeat_ngram"])
    parser.add_argument("--repeat-threshold", type=int, default=DEFAULTS["repeat_threshold"])
    parser.add_argument("--length-blowup", type=float, default=DEFAULTS["length_blowup"])
    parser.add_argument("--near-budget-chars", type=int, default=DEFAULTS["near_budget_chars"],
                        help="Character proxy for evaluation.max_new_tokens being exhausted.")
    parser.add_argument("--fail-over", type=float, default=None,
                        help="Exit 1 when any system's total failure rate exceeds this fraction.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    eval_cfg, data_cfg = config["evaluation"], config["data"]
    target_column = data_cfg["target_column"]
    settings = {
        "trailing_threshold": args.trailing_threshold,
        "repeat_ngram": args.repeat_ngram,
        "repeat_threshold": args.repeat_threshold,
        "length_blowup": args.length_blowup,
        "near_budget_chars": args.near_budget_chars,
    }

    eval_dir = Path(args.eval_dir or eval_cfg["output_dir"])
    suffix = f"_{eval_cfg['detailed_filename']}"
    paths = {path.name[: -len(suffix)]: path for path in sorted(eval_dir.glob(f"*{suffix}"))}
    if not paths:
        raise FileNotFoundError(f"No '*{suffix}' files in {eval_dir}.")

    report = {"eval_dir": str(eval_dir.resolve()), "systems": {}}
    for name, path in paths.items():
        stats = audit_frame(pd.read_csv(path), target_column, settings, args.samples)
        report["systems"][name] = stats
        logger.info(
            "%-10s rows=%d clean=%.1f%%  mean_chars=%.0f (trimmed %.0f, trailing %.0f)",
            name, stats["rows"], 100 * stats["clean_rate"],
            stats["mean_chars"], stats["mean_chars_trimmed"], stats["mean_trailing_chars"],
        )
        for failure, values in stats["failures"].items():
            logger.info("    %-16s %6d rows  %6.2f%%", failure, values["rows"], 100 * values["rate"])

    report_path = Path(args.report) if args.report else eval_dir / "degeneration_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report written to %s", report_path)

    if args.fail_over is not None:
        breached = {
            name: stats["failure_rate"]
            for name, stats in report["systems"].items()
            if stats["failure_rate"] > args.fail_over
        }
        if breached:
            logger.error("Failure rate over --fail-over=%.4f: %s", args.fail_over, breached)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
