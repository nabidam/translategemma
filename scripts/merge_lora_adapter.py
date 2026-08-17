#!/usr/bin/env python3
"""Merge a trained LoRA adapter into its base model for vLLM serving.

vLLM can serve LoRA adapters directly, but only for a subset of module types and
at a per-request cost. A merged checkpoint is a plain HF model directory, so it
loads on the fast path and needs no `--enable-lora` plumbing.

The merge is always done in the base model's full precision (bfloat16 by
default), never against 4-bit weights: merging into a quantised base
dequantises, adds, then re-quantises, which loses part of the adapter delta.
Training with model.use_4bit therefore still merges into a bf16 base here.

Two things the merged directory needs beyond the weights, both written here:

  * the processor (tokenizer + chat template), because vLLM loads the tokenizer
    from the model directory and the base repo is not implied by it;
  * a generation_config.json whose stop set includes <end_of_turn> (106).
    TranslateGemma's config.json publishes only <eos> (1), and a decoder missing
    106 does not stop a fine-tuned model at all. See
    docs/2026-08-10_adapter_degeneration_analysis.md.

Run it inside the offline image, e.g.

    docker compose run --rm trainer python scripts/merge_lora_adapter.py \
      ./translategemma-farsi-science/sft_final ./merged/translategemma-farsi-sft

The base model id is read from the adapter's adapter_config.json unless
--base-model overrides it.

Run it as a plain single process, never under `accelerate launch`: merging is an
elementwise weight update, so N ranks would each redo the whole merge and race to
write the same output directory. Multiple GPUs help only when one cannot hold the
model, which is what `--device auto` is for.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor

from logging_utils import console, logger
from model_loading import (
    load_generation_safe_model_config,
    make_deterministic_generation_config,
    resolve_dtype,
)

# Files worth carrying over from the adapter directory so the merged checkpoint
# still records how it was produced. Copied, not merged into config.json.
PROVENANCE_FILENAMES = ("adapter_config.json", "run_metadata.json", "training_args.bin")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into its base model and save a vLLM-ready checkpoint.",
    )
    parser.add_argument(
        "adapter_dir",
        help="Directory holding adapter_config.json and the adapter weights.",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write the merged model, tokenizer and generation config to.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Override the base model id/path recorded in adapter_config.json.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Torch dtype the base model is loaded in and the merged model is saved in.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Where the base model is placed for the merge: cpu (default, needs no free "
            "VRAM), a single device such as cuda:0, or auto to shard the weights across "
            "every visible GPU. auto only helps when one GPU cannot hold the model; the "
            "merge itself is elementwise and gains nothing from more devices."
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help=(
            "Attention kernel used while loading. Irrelevant to the merge itself, so the "
            "default avoids requiring a FlashAttention build on the merging host."
        ),
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Shard size passed to save_pretrained.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def resolve_adapter_dir(adapter_dir):
    """Return an absolute adapter directory, failing early if it is unusable.

    Mirrors evaluate_translations.resolve_adapter_path: PeftModel.from_pretrained
    treats any path without adapter_config.json as a Hub repo id, which offline
    surfaces as an opaque HFValidationError instead of a path problem.
    """
    path = Path(adapter_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Adapter path does not exist: {path} (resolved from {adapter_dir!r}, cwd={Path.cwd()})"
        )
    if not (path / "adapter_config.json").is_file():
        available = sorted(
            child.parent.relative_to(path).as_posix()
            for child in path.rglob("adapter_config.json")
        )
        hint = f" Adapters found below it: {available[:10]}" if available else ""
        raise FileNotFoundError(f"No adapter_config.json in {path}.{hint}")
    return path


def read_adapter_config(adapter_dir):
    with open(adapter_dir / "adapter_config.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_base_model(adapter_config, override):
    """Return the base model id/path, preferring an explicit override."""
    if override:
        return override
    base_model_id = adapter_config.get("base_model_name_or_path")
    if not base_model_id:
        raise ValueError(
            "adapter_config.json has no base_model_name_or_path; pass --base-model explicitly."
        )
    return base_model_id


def resolve_device_map(device):
    """Turn the --device value into a transformers device_map.

    "auto" spreads the layers over every visible GPU (and spills to CPU), which
    is only useful when no single GPU can hold the model; anything else pins the
    whole model to that one device.
    """
    if device == "auto":
        return "auto"
    return {"": device}


def prepare_output_dir(output_dir, overwrite):
    """Create the output directory, refusing to write over existing weights."""
    path = Path(output_dir).expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}. Pass --overwrite to reuse it.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_provenance_files(adapter_dir, output_dir):
    """Copy the adapter's provenance files under an adapter_ prefix."""
    copied = []
    for filename in PROVENANCE_FILENAMES:
        source = adapter_dir / filename
        if source.is_file():
            shutil.copy2(source, output_dir / f"adapter_{filename}")
            copied.append(filename)
    return copied


def write_merge_metadata(output_dir, adapter_dir, base_model_id, args, adapter_config):
    metadata = {
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "adapter_dir": str(adapter_dir),
        "base_model_id": base_model_id,
        "dtype": args.dtype,
        "merge_device": args.device,
        "lora_r": adapter_config.get("r"),
        "lora_alpha": adapter_config.get("lora_alpha"),
        "target_modules": adapter_config.get("target_modules"),
    }
    path = output_dir / "merge_metadata.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, sort_keys=True, default=list)
    return path


def summarize(adapter_dir, output_dir, base_model_id, args, adapter_config):
    table = Table(show_header=False, box=None)
    table.add_column(style="bold magenta")
    table.add_column(style="green")
    table.add_row("Adapter", str(adapter_dir))
    table.add_row("Base model", str(base_model_id))
    table.add_row("Output", str(output_dir))
    table.add_row("Dtype", args.dtype)
    table.add_row("Merge device", args.device)
    table.add_row("LoRA r / alpha", f"{adapter_config.get('r')} / {adapter_config.get('lora_alpha')}")
    console.print(Panel(table, title="LoRA merge", border_style="cyan"))


def main():
    args = parse_args()

    adapter_dir = resolve_adapter_dir(args.adapter_dir)
    adapter_config = read_adapter_config(adapter_dir)
    base_model_id = resolve_base_model(adapter_config, args.base_model)
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    dtype = resolve_dtype(args.dtype)

    summarize(adapter_dir, output_dir, base_model_id, args, adapter_config)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Loading processor", total=5)

        processor = AutoProcessor.from_pretrained(
            base_model_id, use_fast=True, fix_mistral_regex=False
        )
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        progress.update(task, advance=1, description="Loading base model")

        model_config = load_generation_safe_model_config(base_model_id)
        generation_config = make_deterministic_generation_config(
            model_config, processor, base_model_id
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            config=model_config,
            generation_config=generation_config,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map=resolve_device_map(args.device),
            low_cpu_mem_usage=True,
        )
        progress.update(task, advance=1, description="Applying adapter")

        model = PeftModel.from_pretrained(model, str(adapter_dir))
        progress.update(task, advance=1, description="Merging adapter into base weights")

        # merge_and_unload returns the underlying transformers model with the
        # LoRA deltas folded in, so what gets saved is an ordinary checkpoint.
        model = model.merge_and_unload()
        model.eval()
        # PEFT copies the base model's generation config through unchanged; keep
        # the corrected stop set that make_deterministic_generation_config built.
        model.generation_config = generation_config
        progress.update(task, advance=1, description="Saving merged model")

        model.save_pretrained(
            str(output_dir),
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        processor.save_pretrained(str(output_dir))
        progress.update(task, advance=1, description="Merged model saved")

    copied = copy_provenance_files(adapter_dir, output_dir)
    metadata_path = write_merge_metadata(
        output_dir, adapter_dir, base_model_id, args, adapter_config
    )

    stop_tokens = processor.tokenizer.convert_ids_to_tokens(generation_config.eos_token_id)
    logger.info("Stop tokens written to generation_config.json: %s -> %s",
                stop_tokens, generation_config.eos_token_id)
    if copied:
        logger.info("Copied provenance files: %s", ", ".join(copied))
    logger.info("Merge metadata: %s", metadata_path)

    total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    console.print(
        Panel(
            f"Merged model ready at [bold]{output_dir}[/bold] ({total_bytes / 1e9:.1f} GB)\n"
            f"Serve it with: vllm serve {output_dir}",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
