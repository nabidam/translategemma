"""Report the tokenized length distribution of an SFT split.

training.max_length is the single largest lever on step time, because every
example in a batch is padded and attended up to the batch maximum. Choosing it
from a character-count guess overpays for tokens that carry no signal; choosing
it from a percentile is a measurement.

The lengths reported here are the lengths train.py will see: the same chat
template, the same tokenizer, the same add_special_tokens=False call. Only the
truncation step is skipped, so the "truncated at" columns are meaningful.

    uv run python scripts/analyze_token_lengths.py --config config.yaml

Everything is read from the length_analysis section of config.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

# Run from anywhere: the project modules live in the repository root, not here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from language_pairs import resolve_language_pair  # noqa: E402
from logging_utils import load_config, logger  # noqa: E402
from train import (  # noqa: E402
    format_translategemma_message,
    limit_dataset,
    map_workers,
)


def measure_lengths(dataset, processor, config):
    """Return the rendered token length of every example, in dataset order."""
    data_cfg = config["data"]
    tokenizer = processor.tokenizer

    def measure(example):
        source_lang, target_lang = resolve_language_pair(example, data_cfg)
        messages = format_translategemma_message(
            example[data_cfg["source_column"]], example[data_cfg["target_column"]],
            source_lang, target_lang,
        )
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"token_length": len(tokenizer(text, add_special_tokens=False)["input_ids"])}

    num_proc = map_workers(config["training"]["tokenize_num_proc"], len(dataset))
    measured = dataset.map(measure, remove_columns=dataset.column_names, num_proc=num_proc, desc="Measuring lengths")
    return measured["token_length"]


def percentile(sorted_lengths, fraction):
    """Nearest-rank percentile; no numpy dependency and no interpolation."""
    rank = max(1, min(len(sorted_lengths), math.ceil(fraction * len(sorted_lengths))))
    return sorted_lengths[rank - 1]


def summarize(lengths, config):
    analysis_cfg = config["length_analysis"]
    ordered = sorted(lengths)
    total = len(ordered)
    percentiles = {
        f"p{value:g}": percentile(ordered, float(value) / 100.0)
        for value in analysis_cfg["percentiles"]
    }
    all_tokens = sum(ordered)
    candidates = []
    for limit in analysis_cfg["candidate_max_lengths"]:
        over = sum(1 for length in ordered if length > limit)
        candidates.append({
            "max_length": limit,
            "truncated_examples": over,
            "truncated_fraction": over / total,
            "retained_token_fraction": sum(min(length, limit) for length in ordered) / all_tokens,
        })
    return {
        "examples": total,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / total,
        "percentiles": percentiles,
        "candidate_max_lengths": candidates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(args.config)
    analysis_cfg = config["length_analysis"]
    path = analysis_cfg["dataset_path"] or config["data"]["train_sft_dataset_path"]

    dataset = load_dataset("json", data_files=path, split="train")
    dataset = limit_dataset(dataset, analysis_cfg["max_examples"], "length analysis")
    processor = AutoProcessor.from_pretrained(
        config["model"]["base_model_id"], use_fast=True, fix_mistral_regex=False
    )

    report = summarize(measure_lengths(dataset, processor, config), config)
    report["dataset_path"] = path
    report["configured_max_length"] = config["training"]["max_length"]

    logger.info("Examples=%s min=%s mean=%.1f max=%s", report["examples"], report["min"], report["mean"], report["max"])
    for name, value in report["percentiles"].items():
        logger.info("  %s = %s tokens", name, value)
    for candidate in report["candidate_max_lengths"]:
        logger.info(
            "  max_length=%-5s truncates %.2f%% of examples, keeps %.2f%% of target tokens",
            candidate["max_length"], 100 * candidate["truncated_fraction"], 100 * candidate["retained_token_fraction"],
        )

    report_path = Path(analysis_cfg["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
