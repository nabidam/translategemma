"""Verify that SFT training actually teaches the stop token, and that the
evaluation prompt is the prompt the model was trained on.

The adapter evaluated on 2026-08-10 translates well and then never stops. Two
mechanisms in train.py can produce exactly that, and neither is visible in
eval_loss:

1. The stop token is never in the labels. train.py builds labels as
   ``[-100] * prompt_length + input_ids[prompt_length:]`` over a *truncated*
   ``full_ids[:max_length]``. Any example longer than training.max_length loses
   its tail, and the tail is where <end_of_turn> lives. Those examples train the
   model to continue forever. This script reports how many rows lose their stop
   token, and whether the stop token survives at all in the untruncated case.

2. Train and inference see different prompts. train.py derives the prompt by
   rendering the *training* template with a marker (add_generation_prompt=False),
   while evaluate_translations.py renders with add_generation_prompt=True. The
   comment at train.py:229 concedes these are not guaranteed to be the same
   token prefix. If they differ, the adapter is queried off-distribution at
   evaluation time, which is a textbook cause of degenerate decoding.

3. The decoder is not listening for the token the training template teaches.
   evaluate_translations.py builds its generation config with
   ``GenerationConfig.from_model_config(AutoConfig...)``, which reads only the
   *model* config's eos_token_id and ignores the repository's
   generation_config.json. Gemma-family chat checkpoints carry the turn ender
   (``<end_of_turn>``) in generation_config.json, not always in config.json. If
   the training template ends every target with ``<end_of_turn>`` while
   generate() only stops on ``<eos>``, the adapter emits its stop token, is not
   stopped, and keeps decoding past the turn boundary: newline filler, then a
   fresh turn that re-translates the same source. That is exactly the observed
   output shape, and it explains why the untouched base model is unaffected.

Needs the tokenizer/processor only: no model weights, no GPU.

    uv run python scripts/audit_training_termination.py --config config.yaml
    uv run python scripts/audit_training_termination.py --max-examples 20000
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path

# Run from anywhere: the project modules live in the repository root, not here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402
from transformers import AutoProcessor, GenerationConfig  # noqa: E402

from language_pairs import resolve_language_pair  # noqa: E402
from logging_utils import load_config, logger  # noqa: E402
from model_loading import (  # noqa: E402
    load_generation_safe_model_config,
    make_deterministic_generation_config,
)
from train import (  # noqa: E402
    format_translategemma_message,
    limit_dataset,
)

BOUNDARY_MARKER = "<|translategemma-target-boundary|>"


def as_id_set(value):
    """Normalize eos_token_id, which may be None, an int, or a list of ints."""
    if value is None:
        return set()
    return {int(value)} if isinstance(value, int) else {int(item) for item in value}


def audit_generation_config(processor, config):
    """Compare the stop set evaluate_translations.py uses against the real one.

    ``GenerationConfig.from_model_config`` reads config.json only. The published
    generation_config.json is the file that lists every turn-ending token, so a
    difference between the two is a decoder that cannot stop on the token the
    fine-tune was trained to emit.
    """
    base_model_id = config["model"]["base_model_id"]
    tokenizer = processor.tokenizer

    effective = make_deterministic_generation_config(
        load_generation_safe_model_config(base_model_id), processor
    )
    used = as_id_set(effective.eos_token_id)
    try:
        published = as_id_set(GenerationConfig.from_pretrained(base_model_id).eos_token_id)
    except OSError:
        published = set()

    end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    end_of_turn = end_of_turn if isinstance(end_of_turn, int) and end_of_turn >= 0 else None
    missing = sorted(published - used)

    return {
        "base_model_id": base_model_id,
        "eos_token_ids_used_by_evaluation": sorted(used),
        "eos_tokens_used_by_evaluation": [tokenizer.convert_ids_to_tokens(i) for i in sorted(used)],
        "eos_token_ids_in_generation_config_json": sorted(published),
        "missing_from_evaluation": missing,
        "missing_tokens": [tokenizer.convert_ids_to_tokens(i) for i in missing],
        "end_of_turn_token_id": end_of_turn,
        "end_of_turn_is_a_stop_token": end_of_turn in used if end_of_turn is not None else None,
    }


def stop_token_ids(tokenizer):
    """Every id that generate() may treat as a stop, plus the chat turn ender."""
    ids = set()
    for candidate in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")):
        if isinstance(candidate, int) and candidate >= 0:
            ids.add(candidate)
    return ids


def render_prompt_like_training(processor, messages):
    """Reproduce train.py's prompt derivation exactly, marker and all."""
    marker_messages = [messages[0], {"role": "assistant", "content": BOUNDARY_MARKER}]
    marker_text = processor.apply_chat_template(
        marker_messages, tokenize=False, add_generation_prompt=False
    )
    return marker_text[: marker_text.rindex(BOUNDARY_MARKER)]


def compare_prompts(processor, dataset, data_cfg, limit):
    """Check the training prompt prefix against the evaluation generation prompt."""
    tokenizer = processor.tokenizer
    mismatches = []
    checked = 0
    for record in dataset:
        if checked >= limit:
            break
        checked += 1
        source_lang, target_lang = resolve_language_pair(record, data_cfg)
        messages = format_translategemma_message(
            record[data_cfg["source_column"]], record[data_cfg["target_column"]],
            source_lang, target_lang,
        )
        train_prompt = render_prompt_like_training(processor, messages)
        eval_prompt = processor.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
        if train_prompt != eval_prompt:
            train_ids = tokenizer(train_prompt, add_special_tokens=False)["input_ids"]
            eval_ids = tokenizer(eval_prompt, add_special_tokens=False)["input_ids"]
            if len(mismatches) < 5:
                mismatches.append({
                    "train_prompt_tail": train_prompt[-160:],
                    "eval_prompt_tail": eval_prompt[-160:],
                    "train_prompt_token_tail": tokenizer.convert_ids_to_tokens(train_ids[-12:]),
                    "eval_prompt_token_tail": tokenizer.convert_ids_to_tokens(eval_ids[-12:]),
                    "same_token_prefix": eval_ids[: len(train_ids)] == train_ids,
                })
    return {"checked": checked, "mismatched": len(mismatches) > 0, "examples": mismatches}


def audit_labels(processor, dataset, config, limit):
    """Count examples whose stop token is dropped by max_length truncation."""
    data_cfg, train_cfg = config["data"], config["training"]
    tokenizer = processor.tokenizer
    tokenizer.truncation_side = train_cfg["truncation_side"]
    max_length = train_cfg["max_length"]
    stops = stop_token_ids(tokenizer)

    totals = collections.Counter()
    tail_tokens = collections.Counter()
    long_examples = []
    checked = 0

    for record in dataset:
        if checked >= limit:
            break
        checked += 1
        source_lang, target_lang = resolve_language_pair(record, data_cfg)
        messages = format_translategemma_message(
            record[data_cfg["source_column"]], record[data_cfg["target_column"]],
            source_lang, target_lang,
        )
        full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # What the untruncated rendering ends with: if a stop token is absent
        # here, the template itself never teaches termination.
        if not stops.intersection(full_ids[-3:]):
            totals["template_without_stop_token"] += 1
        tail_tokens[tuple(tokenizer.convert_ids_to_tokens(full_ids[-3:]))] += 1

        truncated_ids = full_ids[:max_length]
        if len(full_ids) > max_length:
            totals["truncated"] += 1
            if not stops.intersection(truncated_ids[-3:]):
                totals["stop_token_lost_to_truncation"] += 1
                if len(long_examples) < 10:
                    long_examples.append({
                        "id": record.get(data_cfg.get("id_column", "id")),
                        "rendered_tokens": len(full_ids),
                        "max_length": max_length,
                    })

        prompt_text = render_prompt_like_training(processor, messages)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_length = min(len(prompt_ids), len(truncated_ids))
        if full_ids[: len(prompt_ids)] != prompt_ids:
            totals["prompt_boundary_mismatch"] += 1
        label_ids = truncated_ids[prompt_length:]
        if not label_ids:
            totals["no_target_tokens"] += 1
        elif not stops.intersection(label_ids[-3:]):
            totals["labels_without_stop_token"] += 1

    return {
        "checked": checked,
        "max_length": max_length,
        "stop_token_ids": sorted(stops),
        "stop_tokens": [tokenizer.convert_ids_to_tokens(i) for i in sorted(stops)],
        "counts": {name: count for name, count in totals.items()},
        "rates": {name: count / max(1, checked) for name, count in totals.items()},
        "most_common_rendered_tails": [
            {"tail": list(tail), "count": count} for tail, count in tail_tokens.most_common(10)
        ],
        "truncated_examples": long_examples,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default=None, help="Defaults to data.train_sft_dataset_path.")
    parser.add_argument("--max-examples", type=int, default=50000,
                        help="Rows to render. The full corpus is not needed to measure a rate.")
    parser.add_argument("--prompt-compare-examples", type=int, default=200)
    parser.add_argument("--report", default="logs/training_termination_audit.json")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    path = args.dataset or config["data"]["train_sft_dataset_path"]

    processor = AutoProcessor.from_pretrained(
        config["model"]["base_model_id"], use_fast=True, fix_mistral_regex=False
    )
    dataset = load_dataset("json", data_files=path, split="train")
    dataset = limit_dataset(dataset, args.max_examples, "termination audit", selection_seed=config["training"]["seed"])

    generation = audit_generation_config(processor, config)
    prompts = compare_prompts(processor, dataset, config["data"], args.prompt_compare_examples)
    labels = audit_labels(processor, dataset, config, args.max_examples)

    if generation["end_of_turn_is_a_stop_token"] is False:
        logger.error(
            "generate() will NOT stop on <end_of_turn> (id=%s). Stop set in use: %s. "
            "generation_config.json lists: %s. The adapter's stop token is being decoded through.",
            generation["end_of_turn_token_id"],
            generation["eos_token_ids_used_by_evaluation"],
            generation["eos_token_ids_in_generation_config_json"],
        )
    elif generation["missing_from_evaluation"]:
        logger.warning("Stop ids dropped by from_model_config: %s (%s)",
                       generation["missing_from_evaluation"], generation["missing_tokens"])
    else:
        logger.info("Stop set matches generation_config.json: %s",
                    generation["eos_tokens_used_by_evaluation"])

    logger.info("Stop tokens: %s -> %s", labels["stop_tokens"], labels["stop_token_ids"])
    for name, count in labels["counts"].items():
        logger.info("  %-32s %8d rows  %6.2f%%", name, count, 100 * labels["rates"][name])
    logger.info("Most common rendered tails: %s", labels["most_common_rendered_tails"][:3])
    if prompts["mismatched"]:
        logger.error(
            "Training prompt != evaluation generation prompt. The adapter is queried "
            "off-distribution at inference. First example: %s",
            prompts["examples"][0],
        )
    else:
        logger.info("Training prompt matches the evaluation generation prompt on %s rows.", prompts["checked"])

    report = {
        "dataset_path": path,
        "generation_stop_set": generation,
        "prompt_parity": prompts,
        "labels": labels,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
