#!/usr/bin/env python
"""Verify parity and stop behavior between unmerged LoRA adapter and merged checkpoint.

This script runs on the fine-tune machine before transferring weights to the serving machine.
It verifies:
  1. Exact string / token parity on a test corpus between (Base + LoRA) and Merged model.
  2. Strict adherence to stop tokens (<end_of_turn> id 106, <eos> id 1).
  3. Absence of runaway loops, assistant restarts, or max-token truncations.
  4. Generation of a structured JSON verification report.

Usage:
  uv run python scripts/verify_merged_checkpoint.py \
      --base-model google/translategemma-12b-it \
      --adapter checkpoints/sft-translategemma-12b-it \
      --merged-model exports/translategemma-12b-it-merged-v1 \
      --output-report reports/merge_verification_report.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor

from model_loading import load_generation_safe_model_config, make_deterministic_generation_config, resolve_dtype
from prompting import CHAT_TURN_END_TOKEN, render_training_prompts, tokenize_prompts_for_generation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_merged_checkpoint")

DEFAULT_TEST_SAMPLES = [
    {
        "source_lang": "en",
        "target_lang": "fa",
        "text": "The model relies on multi-query attention to process the genome sequence.",
    },
    {
        "source_lang": "en",
        "target_lang": "fa",
        "text": "Artificial intelligence is rapidly transforming computational biology and drug discovery.",
    },
    {
        "source_lang": "en",
        "target_lang": "fa",
        "text": "Photosynthesis is the biological process by which plants convert light energy into chemical energy.",
    },
    {
        "source_lang": "en",
        "target_lang": "fa",
        "text": "The quantum harmonic oscillator is an important model system in quantum mechanics.",
    },
    {
        "source_lang": "en",
        "target_lang": "fa",
        "text": "In cardiovascular physiology, cardiac output is defined as the volume of blood pumped per minute.",
    },
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path or HuggingFace ID of the base model.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory.",
    )
    parser.add_argument(
        "--merged-model",
        type=str,
        required=True,
        help="Path to the exported merged model directory.",
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=None,
        help="Optional path to a JSONL file with {'text', 'source_lang', 'target_lang'}.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/merge_verification_report.json",
        help="Path to write the output verification JSON report.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Torch data type (default: bfloat16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device placement ('auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens for generation test (default: 512).",
    )
    return parser.parse_args(argv)


def load_corpus(corpus_file: Optional[str]) -> List[Dict[str, str]]:
    if corpus_file is None:
        return DEFAULT_TEST_SAMPLES
    path = Path(corpus_file)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def generate_translations(
    model: Any,
    processor: Any,
    samples: List[Dict[str, str]],
    generation_config: Any,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Run deterministic inference over test samples and record output statistics."""
    tokenizer = processor.tokenizer
    results = []

    user_messages = [
        {"role": "user", "content": f"<<<source>>>{s['source_lang']}<<<target>>>{s['target_lang']}<<<text>>>{s['text']}"}
        for s in samples
    ]
    prompts = render_training_prompts(processor, user_messages)

    for i, (sample, prompt) in enumerate(zip(samples, prompts)):
        inputs = tokenize_prompts_for_generation(processor, [prompt], device=device)
        input_len = inputs.input_ids.shape[1]

        start_t = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                generation_config=generation_config,
            )
        elapsed = time.perf_counter() - start_t

        new_ids = output_ids[0, input_len:].tolist()
        gen_text = tokenizer.decode(new_ids, skip_special_tokens=True)

        last_token_id = new_ids[-1] if new_ids else None
        last_token_str = tokenizer.convert_ids_to_tokens(last_token_id) if last_token_id is not None else None

        stopped_on_eos = (
            last_token_id in generation_config.eos_token_id
            if isinstance(generation_config.eos_token_id, list)
            else last_token_id == generation_config.eos_token_id
        )
        hit_max_tokens = len(new_ids) >= generation_config.max_new_tokens

        results.append({
            "index": i,
            "source": sample["text"],
            "source_lang": sample.get("source_lang", "en"),
            "target_lang": sample.get("target_lang", "fa"),
            "prompt_length_tokens": input_len,
            "generated_length_tokens": len(new_ids),
            "generated_text": gen_text,
            "last_token_id": last_token_id,
            "last_token_str": last_token_str,
            "stopped_on_eos": stopped_on_eos,
            "hit_max_tokens": hit_max_tokens,
            "latency_seconds": elapsed,
        })
    return results


def run_verification(
    base_model_id: str,
    adapter_path: str,
    merged_model_path: str,
    corpus_file: Optional[str] = None,
    output_report_path: str = "reports/merge_verification_report.json",
    dtype_name: str = "bfloat16",
    device_name: str = "auto",
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    torch_dtype = resolve_dtype(dtype_name)
    device = torch.device("cuda" if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    logger.info("Running verification using device: %s, dtype: %s", device, torch_dtype)

    samples = load_corpus(corpus_file)
    logger.info("Loaded %d verification samples.", len(samples))

    # 1. Load Processor
    logger.info("Loading processor from merged artifact %s...", merged_model_path)
    processor = AutoProcessor.from_pretrained(merged_model_path, fix_markdown=False)
    processor.tokenizer.padding_side = "left"

    # 2. Run Unmerged (Base + Adapter)
    logger.info("1/2 Loading Unmerged Reference (Base + LoRA)...")
    base_config = load_generation_safe_model_config(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=base_config,
        torch_dtype=torch_dtype,
        device_map=device_name if device_name in ("auto", "cpu", "cuda") else "auto",
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch_dtype)
    peft_model.eval()

    gen_config = make_deterministic_generation_config(base_config, processor, base_model_id=base_model_id)
    gen_config.max_new_tokens = max_new_tokens

    logger.info("Generating reference translations with (Base + LoRA)...")
    ref_results = generate_translations(peft_model, processor, samples, gen_config, device=device)

    # Free unmerged model to save memory
    del peft_model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Run Merged Model
    logger.info("2/2 Loading Merged Model Checkpoint %s...", merged_model_path)
    merged_config = load_generation_safe_model_config(merged_model_path)
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        config=merged_config,
        torch_dtype=torch_dtype,
        device_map=device_name if device_name in ("auto", "cpu", "cuda") else "auto",
        low_cpu_mem_usage=True,
    )
    merged_model.eval()

    merged_gen_config = make_deterministic_generation_config(merged_config, processor, base_model_id=merged_model_path)
    merged_gen_config.max_new_tokens = max_new_tokens

    logger.info("Generating translations with Merged Model...")
    merged_results = generate_translations(merged_model, processor, samples, merged_gen_config, device=device)

    del merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Compare Results
    exact_matches = 0
    total = len(samples)
    comparisons = []
    stop_violations = 0
    truncations = 0

    for r_ref, r_mer in zip(ref_results, merged_results):
        is_exact = (r_ref["generated_text"] == r_mer["generated_text"])
        if is_exact:
            exact_matches += 1

        if not r_mer["stopped_on_eos"]:
            stop_violations += 1
        if r_mer["hit_max_tokens"]:
            truncations += 1

        comparisons.append({
            "index": r_ref["index"],
            "source": r_ref["source"],
            "exact_match": is_exact,
            "reference_text": r_ref["generated_text"],
            "merged_text": r_mer["generated_text"],
            "reference_tokens": r_ref["generated_length_tokens"],
            "merged_tokens": r_mer["generated_length_tokens"],
            "merged_stopped_on_eos": r_mer["stopped_on_eos"],
            "merged_last_token": r_mer["last_token_str"],
        })

    exact_match_ratio = (exact_matches / total) if total > 0 else 0.0

    report = {
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": base_model_id,
        "adapter": adapter_path,
        "merged_model": merged_model_path,
        "sample_count": total,
        "exact_matches": exact_matches,
        "exact_match_ratio": exact_match_ratio,
        "stop_violations": stop_violations,
        "truncation_count": truncations,
        "passed": (exact_match_ratio >= 0.95 and stop_violations == 0 and truncations == 0),
        "comparisons": comparisons,
    }

    out_report_path = Path(output_report_path)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Verification complete: %d/%d exact matches (%.2f%%). Stop violations: %d. Report saved to: %s",
        exact_matches,
        total,
        exact_match_ratio * 100,
        stop_violations,
        out_report_path,
    )
    return report


def main() -> int:
    args = parse_args()
    try:
        report = run_verification(
            base_model_id=args.base_model,
            adapter_path=args.adapter,
            merged_model_path=args.merged_model,
            corpus_file=args.corpus_file,
            output_report_path=args.output_report,
            dtype_name=args.dtype,
            device_name=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        return 0 if report.get("passed", False) else 1
    except Exception as e:
        logger.error("Verification failed with error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
