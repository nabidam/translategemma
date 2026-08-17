#!/usr/bin/env python3
"""Quantise a merged TranslateGemma checkpoint to FP8 for vLLM serving.

Runs *after* scripts/merge_lora_adapter.py, never instead of part of it. The
merge folds the LoRA delta into full-precision weights; quantising first, or
merging into an already-quantised base, rounds the delta away -- a LoRA update
is small enough per channel to disappear into the quantisation step, which
costs exactly the fine-tune the merge exists to preserve.

The scheme is FP8_DYNAMIC: per-channel FP8 weight scales, activation scales
computed at run time. That choice is deliberate over 4-bit AWQ/GPTQ:

  * It needs **no calibration corpus**. AWQ/GPTQ pick their scales from sample
    activations, so a general-purpose calibration set silently biases the model
    away from the domain it was fine-tuned for. Nothing here has to be
    representative of anything.
  * FP8 is near-lossless in practice, where 4-bit is measurably lossy, and the
    thing 4-bit degrades most is the low-magnitude weight structure a LoRA
    adapter writes.
  * Ada (sm_89) and Blackwell (sm_120) run FP8 on tensor cores, so the halved
    footprint comes with faster GEMMs rather than a throughput trade.

Two things the output directory needs beyond the weights, both handled here:

  * the processor (tokenizer + chat template), copied byte for byte rather than
    re-saved, because vLLM loads the tokenizer from the model directory and this
    image runs a different transformers than the API does;
  * generation_config.json, whose stop set includes <end_of_turn> (106). Lose
    it and the decoder does not stop on a fine-tuned model at all, while still
    returning fluent text. See docs/2026-08-10_adapter_degeneration_analysis.md.
    The final check below refuses to leave a directory without it.

Run it in the quantiser image, which carries llm-compressor. It is a separate
image because no llm-compressor release installs against the training lock; the
version table is in scripts/quantize.Dockerfile.

    docker compose build quantizer
    docker compose run --rm quantizer \\
      /models/translategemma-12b-merged /models/translategemma-12b-merged-fp8

The quantisation is a weight transform, not a forward pass, so --device cpu (the
default) is enough and leaves the GPU to whatever is serving. It also means the
torch build here need not match the one that serves the result: what is written
is a compressed-tensors checkpoint -- safetensors plus scales plus a config
entry -- and the FP8 *kernels* are a serving-time concern.
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

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from logging_utils import console, logger

# The chat turn ender, as a bare id: the check it guards runs after the model is
# already saved, so it must not depend on anything that could fail to import.
CHAT_TURN_END_TOKEN_ID = 106

# Modules left in full precision. lm_head is the standard exclusion: it is
# large, numerically sensitive, and quantising it buys little. The vision tower
# and projector are excluded because TranslateGemma is a Gemma 3 multimodal
# checkpoint whose image path is unused for translation but still present in the
# weights -- quantising a path no calibration ever exercises is pure risk. The
# regex entries simply match nothing on a text-only checkpoint.
DEFAULT_IGNORE = [
    "lm_head",
    "re:.*vision_tower.*",
    "re:.*multi_modal_projector.*",
    "re:.*vision_model.*",
]

# Carried over from the merged directory so the quantised checkpoint still
# records how it was produced.
PROVENANCE_FILENAMES = (
    "merge_metadata.json",
    "adapter_adapter_config.json",
    "adapter_run_metadata.json",
)

# The processor: tokenizer, chat template, and the preprocessor configs that
# come with a multimodal checkpoint. COPIED byte for byte, never re-saved
# through save_pretrained.
#
# This image runs a different transformers than the API and the evaluation
# harness do -- it has to, since no llm-compressor release installs against the
# training lock (see scripts/quantize.Dockerfile). Re-serialising the tokenizer
# under that other version is exactly the kind of silent rewrite that produced
# docs/2026-08-10_adapter_degeneration_analysis.md: a chat template that renders
# one space differently still yields fluent Farsi. Copying cannot drift.
PROCESSOR_FILENAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantise a merged checkpoint to FP8 and save a vLLM-ready directory.",
    )
    parser.add_argument(
        "input_dir",
        help="The merged, full-precision checkpoint written by merge_lora_adapter.py.",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write the FP8 model, tokenizer and generation config to.",
    )
    parser.add_argument(
        "--scheme",
        default="FP8_DYNAMIC",
        help=(
            "compressed-tensors scheme. FP8_DYNAMIC (default) needs no calibration "
            "data. FP8 (static per-tensor activation scales) does, and this script "
            "does not supply any."
        ),
    )
    parser.add_argument(
        "--ignore",
        default=None,
        help=(
            "Comma-separated module patterns to leave in full precision. Default: "
            f"{','.join(DEFAULT_IGNORE)}"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Where the weights are placed for the transform: cpu (default, needs no "
            "free VRAM and leaves the GPU to whatever is serving), or a single device "
            "such as cuda:0. 'auto' shards across every visible GPU and is only worth "
            "it when no single device holds the model; the save consolidates back to "
            "one device either way."
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


def load_oneshot():
    """Import llm-compressor's oneshot entry point across its two layouts."""
    try:
        from llmcompressor import oneshot
    except ImportError:
        try:
            from llmcompressor.transformers import oneshot
        except ImportError as error:
            raise SystemExit(
                "llm-compressor is not installed in this image. Run this script through "
                "the quantiser image: docker compose build quantizer && "
                "docker compose run --rm quantizer ..."
            ) from error
    return oneshot


def resolve_input_dir(input_dir):
    """Return an absolute merged-checkpoint directory, failing early if unusable."""
    path = Path(input_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path} (from {input_dir!r})")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"No config.json in {path}; this is not a model directory.")
    if (path / "adapter_config.json").is_file():
        raise ValueError(
            f"{path} looks like a LoRA adapter, not a merged checkpoint. Run "
            "scripts/merge_lora_adapter.py first: quantising a base model and then "
            "applying an adapter to it loses most of the adapter."
        )
    return path


def already_quantized(input_dir):
    """True if the input carries a quantization_config, i.e. this is a re-run."""
    with open(input_dir / "config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return "quantization_config" in config or "compression_config" in config


def prepare_output_dir(output_dir, overwrite):
    path = Path(output_dir).expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Pass --overwrite to reuse it."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device_map(device):
    if device == "auto":
        return "auto"
    return {"": device}


def load_model(input_dir, device_map):
    """Load the checkpoint under the class its own config.json declares.

    AutoModelForCausalLM is ambiguous for a multimodal Gemma 3 config: it
    resolves to the text-only class on some transformers versions and to the
    conditional-generation wrapper on others, and this image deliberately runs a
    different transformers than the rest of the repository. Naming the
    architecture removes the ambiguity and loads what vLLM will load.

    dtype="auto" keeps the merged checkpoint's own precision; this script must
    not be the place a bf16 merge silently becomes fp16.
    """
    import transformers

    config = AutoConfig.from_pretrained(str(input_dir))
    for name in getattr(config, "architectures", None) or []:
        model_class = getattr(transformers, name, None)
        if model_class is not None:
            break
    else:
        model_class = AutoModelForCausalLM
        logger.warning(
            "config.json declares %s, which this transformers does not expose; "
            "falling back to AutoModelForCausalLM.",
            getattr(config, "architectures", None),
        )
    logger.info("Loading %s from %s", model_class.__name__, input_dir)
    return model_class.from_pretrained(
        str(input_dir),
        dtype="auto",
        device_map=device_map,
        low_cpu_mem_usage=True,
    )


def consolidate_for_save(model):
    """Detach every offload hook before save_pretrained.

    transformers has a second save path for models with offloaded parameters:
    it builds a module map from the modules that carry an accelerate hook, so it
    can re-gather their weights, then looks up every key of the state dict in
    that map. Keys belonging to modules *without* a hook are simply absent, and
    the lookup raises a bare KeyError, e.g.

        KeyError: 'vision_tower.vision_model.embeddings.patch_embedding.weight'

    The modules the ignore list leaves dense are exactly the ones that hit it:
    compression never touches them, so they never get a hook, while every
    quantised Linear does.

    The hooks come from compressed-tensors, which onloads and offloads each
    module as it compresses it. They are attached per module and do NOT show up
    in `hf_device_map` -- an earlier version of this function gated on that
    attribute and therefore never ran. Detaching writes the offloaded weights
    back to real tensors, which leaves nothing for transformers to re-gather and
    puts the ordinary save path back in play.
    """
    from accelerate.hooks import remove_hook_from_module

    hooked = sum(1 for _, module in model.named_modules() if hasattr(module, "_hf_hook"))
    if not hooked:
        return model

    logger.info("Detaching offload hooks from %d module(s) before saving.", hooked)
    remove_hook_from_module(model, recurse=True)
    # One device for the save, whatever devices the hooks restored to. The
    # weights are FP8 by this point, so this is about half the size of the
    # merge it started from.
    model = model.to("cpu")
    # Harmless when absent; transformers also consults this attribute.
    try:
        del model.hf_device_map
    except AttributeError:
        pass
    return model


def copy_processor_files(input_dir, output_dir):
    """Copy the tokenizer and chat template verbatim from the merged checkpoint.

    Raises if the tokenizer itself is missing: vLLM loads it from the model
    directory, and a directory without one is unservable.
    """
    copied = []
    for filename in PROCESSOR_FILENAMES:
        source = input_dir / filename
        if source.is_file():
            shutil.copy2(source, output_dir / filename)
            copied.append(filename)
    if not any(name.startswith("tokenizer") for name in copied):
        raise FileNotFoundError(
            f"No tokenizer files in {input_dir}. vLLM loads the tokenizer from the model "
            "directory, so the merged checkpoint must carry one; re-run "
            "scripts/merge_lora_adapter.py."
        )
    return copied


def copy_provenance_files(input_dir, output_dir):
    copied = []
    for filename in PROVENANCE_FILENAMES:
        source = input_dir / filename
        if source.is_file():
            shutil.copy2(source, output_dir / filename)
            copied.append(filename)
    return copied


def ensure_generation_config(input_dir, output_dir):
    """Guarantee the quantised directory carries the merged stop set.

    save_pretrained writes generation_config.json from the loaded model, but a
    quantisation pipeline that reconstructs the model can drop it, and the
    failure is silent: vLLM then falls back to config.json's eos_token_id, which
    for TranslateGemma is <eos> alone. Copy it if it is missing, and verify
    either way.
    """
    target = output_dir / "generation_config.json"
    source = input_dir / "generation_config.json"
    if not target.is_file():
        if not source.is_file():
            raise FileNotFoundError(
                f"Neither {source} nor {target} exists. The merged checkpoint must carry a "
                "generation_config.json; re-run scripts/merge_lora_adapter.py."
            )
        shutil.copy2(source, target)
        logger.warning("generation_config.json was not written by the save; copied it over.")

    with open(target, "r", encoding="utf-8") as handle:
        generation_config = json.load(handle)
    eos = generation_config.get("eos_token_id")
    stop_ids = [eos] if isinstance(eos, int) else list(eos or [])
    if CHAT_TURN_END_TOKEN_ID not in stop_ids:
        raise ValueError(
            f"{target} has eos_token_id={stop_ids}, which omits <end_of_turn> "
            f"({CHAT_TURN_END_TOKEN_ID}). A decoder missing it does not stop a fine-tuned "
            "model at all. Fix the merged checkpoint before serving this."
        )
    return stop_ids


def verify_quantized(output_dir):
    """Confirm the saved config really advertises quantisation to vLLM."""
    with open(output_dir / "config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)
    quantization_config = config.get("quantization_config") or config.get("compression_config")
    if not quantization_config:
        raise ValueError(
            f"{output_dir / 'config.json'} carries no quantization_config. vLLM would load "
            "these weights as if they were dense, so the save did not do what it claimed."
        )
    return quantization_config


def write_quantization_metadata(output_dir, input_dir, args, ignore):
    metadata = {
        "quantized_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "scheme": args.scheme,
        "ignore": ignore,
        "device": args.device,
    }
    path = output_dir / "quantization_metadata.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return path


def summarize(input_dir, output_dir, args, ignore):
    table = Table(show_header=False, box=None)
    table.add_column(style="bold magenta")
    table.add_column(style="green")
    table.add_row("Merged model", str(input_dir))
    table.add_row("Output", str(output_dir))
    table.add_row("Scheme", args.scheme)
    table.add_row("Device", args.device)
    table.add_row("Left dense", ", ".join(ignore))
    console.print(Panel(table, title="FP8 quantisation", border_style="cyan"))


def directory_size_gb(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main():
    args = parse_args()
    oneshot = load_oneshot()

    from llmcompressor.modifiers.quantization import QuantizationModifier

    input_dir = resolve_input_dir(args.input_dir)
    if already_quantized(input_dir):
        raise SystemExit(
            f"{input_dir} is already quantised. Quantising a quantised checkpoint compounds "
            "the error; start from the bf16 merge."
        )
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    ignore = (
        [part.strip() for part in args.ignore.split(",") if part.strip()]
        if args.ignore
        else list(DEFAULT_IGNORE)
    )

    summarize(input_dir, output_dir, args, ignore)
    input_size_gb = directory_size_gb(input_dir)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Loading merged model", total=3)

        model = load_model(input_dir, resolve_device_map(args.device))
        progress.update(task, advance=1, description=f"Quantising to {args.scheme}")

        # No dataset argument: FP8_DYNAMIC derives weight scales from the
        # weights and leaves activation scales to run time, so there is nothing
        # to calibrate on and nothing to bias.
        recipe = QuantizationModifier(targets="Linear", scheme=args.scheme, ignore=ignore)
        oneshot(model=model, recipe=recipe)
        progress.update(task, advance=1, description="Saving quantised model")

        model = consolidate_for_save(model)
        try:
            model.save_pretrained(
                str(output_dir),
                save_compressed=True,
                max_shard_size=args.max_shard_size,
            )
        except KeyError as error:
            # llm-compressor's save wrapper offloads the model to CPU while it
            # compresses, *inside* this call, so the layout it saves under is not
            # the one this script hands it. transformers then rebuilds a module
            # map from the offloaded modules and looks every state-dict key up in
            # it; the modules --ignore left dense are absent, and the lookup
            # raises a bare KeyError naming one of their weights.
            raise SystemExit(
                f"save_pretrained could not place {error} in its offloaded-module map.\n\n"
                "llm-compressor offloads the model to CPU inside save_pretrained, and the "
                "modules left in full precision by --ignore are missing from the map it "
                "builds. Two ways around it, cheapest first:\n"
                "  1. --device cpu, so nothing is dispatched to a GPU and there is no "
                "offload for the save to undo. Slower, and the usual fix.\n"
                "  2. --ignore lm_head, which quantises the vision tower too. It is never "
                "exercised by translation, but check that the serving vLLM loads the result "
                "before relying on it.\n"
                "Neither changes the decoder weights this model translates with."
            ) from error
        progress.update(task, advance=1, description="Quantised model saved")

    quantization_config = verify_quantized(output_dir)
    processor_files = copy_processor_files(input_dir, output_dir)
    stop_ids = ensure_generation_config(input_dir, output_dir)
    copied = copy_provenance_files(input_dir, output_dir)
    metadata_path = write_quantization_metadata(output_dir, input_dir, args, ignore)

    # Loaded from the *output* directory, so this reports what was actually
    # written rather than what was intended.
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), use_fast=True)
    stop_tokens = tokenizer.convert_ids_to_tokens(stop_ids)
    logger.info("Copied processor files: %s", ", ".join(processor_files))
    logger.info("Stop set preserved: %s -> %s", stop_tokens, stop_ids)
    logger.info("quantization_config: %s", json.dumps(quantization_config, sort_keys=True)[:400])
    if copied:
        logger.info("Copied provenance files: %s", ", ".join(copied))
    logger.info("Quantisation metadata: %s", metadata_path)

    output_size_gb = directory_size_gb(output_dir)
    console.print(
        Panel(
            f"FP8 model ready at [bold]{output_dir}[/bold]\n"
            f"{input_size_gb:.1f} GB -> {output_size_gb:.1f} GB "
            f"({output_size_gb / input_size_gb:.0%} of the merge)\n\n"
            f"Serve it with: vllm serve {output_dir}\n"
            "Then score it against the bf16 merge on the same test set before trusting it:\n"
            "  uv run python evaluate_translations.py --config config.yaml",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
