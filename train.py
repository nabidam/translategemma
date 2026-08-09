"""Config-driven QLoRA training for TranslateGemma."""

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import logging
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from trl import DPOTrainer, pack_dataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from canary_config import canary_run_config
from language_pairs import (
    DEFAULT_SOURCE_LANG_COLUMN,
    DEFAULT_TARGET_LANG_COLUMN,
    resolve_language_pair,
)
from logging_utils import logger, setup_logging, log_config_summary, load_config

from accelerate import Accelerator, PartialState


PREPARED_CACHE_VERSION = 2


class RichLoggingCallback(TrainerCallback):
    """Send Trainer metrics to the project log file and console."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info(
                "[step=%s epoch=%.2f] %s",
                state.global_step,
                state.epoch or 0.0,
                " ".join(f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
                          for key, value in logs.items()),
            )

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Train begin: total_steps=%s epochs=%s batch=%s", state.max_steps, state.num_train_epochs, args.per_device_train_batch_size)

    def on_train_end(self, args, state, control, **kwargs):
        logger.info("Train end: total_steps=%s epoch=%.4f", state.global_step, state.epoch or 0.0)


class TranslationDataCollator:
    """Pad pre-tokenized translations while keeping prompt labels masked with -100."""

    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        # The dataset is deliberately tokenized once (and can be cached and
        # processed in parallel) because labels need prompt-token masking.
        # Calling the tokenizer on raw text here would discard those labels,
        # so Transformers' generic fast-tokenizer padding advice does not
        # apply to this collator.
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    def __call__(self, features):
        model_inputs = [
            {"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]}
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_inputs, padding=True, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt"
        )
        # Take the width from the padded batch rather than from the raw feature
        # lengths: pad_to_multiple_of can round the batch up past the longest
        # example, and labels must match input_ids exactly.
        max_length = batch["input_ids"].shape[1]
        labels = []
        for feature in features:
            padding = [-100] * (max_length - len(feature["labels"]))
            labels.append(
                padding + feature["labels"]
                if self.tokenizer.padding_side == "left"
                else feature["labels"] + padding
            )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def training_world_size():
    """Return the active Accelerate process count without assuming CUDA."""
    return PartialState().num_processes


def resolve_gradient_accumulation_steps(config):
    """Derive accumulation from the configured global batch when requested."""
    cfg = config["training"]
    effective_batch = cfg.get("effective_batch_size")
    if effective_batch is None:
        return cfg["gradient_accumulation_steps"]
    denominator = cfg["batch_size"] * training_world_size()
    if effective_batch % denominator:
        raise ValueError(
            "training.effective_batch_size must be exactly divisible by "
            "training.batch_size * world_size; "
            f"got {effective_batch} / ({cfg['batch_size']} * {training_world_size()})"
        )
    accumulation = effective_batch // denominator
    if accumulation < 1:
        raise ValueError(
            f"training.effective_batch_size={effective_batch} is smaller than the "
            f"{denominator}-sample distributed micro-batch"
        )
    return accumulation


def format_translategemma_message(source, target, source_lang, target_lang):
    return [
        {"role": "user", "content": [{"type": "text", "source_lang_code": source_lang, "target_lang_code": target_lang, "text": source}]},
        {"role": "assistant", "content": str(target)},
    ]


def resolve_dtype(name):
    """Turn a config dtype name such as "bfloat16" into a torch.dtype."""
    dtype = getattr(torch, str(name), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"model.dtype must name a torch dtype (for example bfloat16); got {name!r}")
    return dtype


def load_generation_safe_model_config(base_model_id):
    """Load model config without TranslateGemma's invalid sampling defaults."""
    config = AutoConfig.from_pretrained(base_model_id)
    # Apply this to both multimodal wrappers and their decoder config. Model
    # construction creates GenerationConfig objects for nested models too.
    for candidate in (config, config.get_text_config()):
        candidate.temperature = 1.0
        candidate.top_p = 1.0
        candidate.top_k = 50
    return config


def make_deterministic_generation_config(model_config, processor):
    """Return explicit, warning-free defaults for translation generation."""
    tokenizer = processor.tokenizer
    generation_config = GenerationConfig.from_model_config(model_config)
    generation_config.do_sample = False
    generation_config.temperature = 1.0
    generation_config.top_p = 1.0
    generation_config.top_k = 50
    if generation_config.bos_token_id is None:
        generation_config.bos_token_id = tokenizer.bos_token_id
    if generation_config.eos_token_id is None:
        generation_config.eos_token_id = tokenizer.eos_token_id
    if generation_config.pad_token_id is None:
        generation_config.pad_token_id = tokenizer.pad_token_id
    if generation_config.pad_token_id is None:
        generation_config.pad_token_id = tokenizer.eos_token_id
    # from_pretrained reconstructs a supplied GenerationConfig via from_dict,
    # which does not preserve _original_object_hash. Leaving this marked as
    # model-derived makes generate() enter a legacy hash check and crash.
    generation_config._from_model_config = False
    return generation_config


def map_workers(requested, dataset_size):
    """Clamp a configured process count to what datasets.map will accept."""
    if not requested or requested <= 1:
        return None
    workers = min(int(requested), dataset_size)
    return workers if workers > 1 else None


def limit_dataset(dataset, max_examples, split_name, selection_seed=None):
    """Return at most max_examples rows, optionally sampling deterministically."""
    if max_examples is None or len(dataset) <= max_examples:
        return dataset
    logger.info("Limiting %s split from %s to %s examples.", split_name, len(dataset), max_examples)
    if selection_seed is not None:
        dataset = dataset.shuffle(seed=selection_seed)
    return dataset.select(range(max_examples))


def load_sft_split(path, config, split_name, max_examples=None, selection_seed=None):
    data_cfg = config["data"]
    dataset = load_dataset("json", data_files=path, split="train")
    required = {data_cfg["source_column"], data_cfg["target_column"]}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{split_name} dataset {path} is missing columns: {sorted(missing)}")
    if not len(dataset):
        raise ValueError(f"{split_name} dataset {path} contains no examples")
    dataset = limit_dataset(dataset, max_examples, split_name, selection_seed)
    logger.info("Loaded %s split: %s examples from %s", split_name, len(dataset), path)
    return dataset


def tokenize_sft_dataset(dataset, processor, config, split_name):
    """Render the exact TranslateGemma template and mask all source/prompt tokens."""
    data_cfg, train_cfg = config["data"], config["training"]
    tokenizer = processor.tokenizer
    tokenizer.truncation_side = train_cfg["truncation_side"]
    max_length = train_cfg["max_length"]
    boundary_marker = "<|translategemma-target-boundary|>"

    def tokenize_example(example):
        source_lang, target_lang = resolve_language_pair(example, data_cfg)
        messages = format_translategemma_message(
            example[data_cfg["source_column"]], example[data_cfg["target_column"]],
            source_lang, target_lang,
        )
        full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # TranslateGemma's generation prompt is not guaranteed to be a literal token
        # prefix of its completed assistant turn. Render the same *training* template
        # with a unique assistant marker instead, then use the marker position as the
        # loss boundary. This preserves the previously working rendering behavior.
        marker_messages = [messages[0], {"role": "assistant", "content": boundary_marker}]
        marker_text = processor.apply_chat_template(marker_messages, tokenize=False, add_generation_prompt=False)
        try:
            response_start = marker_text.rindex(boundary_marker)
        except ValueError as error:
            raise ValueError("TranslateGemma chat template did not preserve the assistant boundary marker.") from error
        prompt_text = marker_text[:response_start]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids:
            # A tokenizer can occasionally merge the first response character with a
            # preceding template token. Mask the shared token rather than failing the
            # complete run or accidentally training on source/prompt content.
            prompt_length = 0
            for full_token, prompt_token in zip(full_ids, prompt_ids):
                if full_token != prompt_token:
                    break
                prompt_length += 1
            logger.warning(
                "Template/token boundary mismatch in %s; conservatively masking %s shared prompt tokens.",
                split_name,
                prompt_length,
            )
        else:
            prompt_length = len(prompt_ids)
        was_truncated = len(full_ids) > max_length
        input_ids = full_ids[:max_length]
        prompt_length = min(prompt_length, len(input_ids))
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            # Read by Trainer's LengthGroupedSampler via
            # training.length_column_name. Recorded post-truncation, which is
            # the width this example actually contributes to its batch.
            train_cfg["length_column_name"]: len(input_ids),
            "labels": [-100] * prompt_length + input_ids[prompt_length:],
            "has_target": len(input_ids) > prompt_length,
            "was_truncated": was_truncated,
        }

    num_proc = map_workers(train_cfg["tokenize_num_proc"], len(dataset))
    tokenized = dataset.map(tokenize_example, remove_columns=dataset.column_names, num_proc=num_proc, desc=f"Tokenizing {split_name}")
    truncated = sum(tokenized["was_truncated"])
    without_target = len(tokenized) - sum(tokenized["has_target"])
    if without_target:
        logger.warning("%s examples in %s contain no target tokens at max_length=%s and will be excluded.", without_target, split_name, max_length)
        tokenized = tokenized.filter(lambda example: example["has_target"], num_proc=num_proc, desc=f"Filtering {split_name}")
    if not len(tokenized):
        raise ValueError(f"No usable {split_name} examples remain after tokenization.")
    logger.info("%s tokenized: examples=%s truncated=%s (%.2f%%), max_length=%s", split_name, len(tokenized), truncated, 100 * truncated / (len(tokenized) + without_target), max_length)
    return tokenized.remove_columns(["has_target", "was_truncated"])


def _prepared_cache_path(path, config, split_name, max_examples, packed, selection_seed=None):
    """Return a stable cache path that changes with the input and preprocessing recipe."""
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    train_cfg = config["training"]
    identity = {
        "cache_version": PREPARED_CACHE_VERSION,
        "source": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "split": split_name,
        "max_examples": max_examples,
        "base_model_id": config["model"]["base_model_id"],
        "transformers_version": importlib.metadata.version("transformers"),
        "trl_version": importlib.metadata.version("trl"),
        "source_lang": config["data"].get("source_lang"),
        "target_lang": config["data"].get("target_lang"),
        "source_lang_column": config["data"].get("source_lang_column", DEFAULT_SOURCE_LANG_COLUMN),
        "target_lang_column": config["data"].get("target_lang_column", DEFAULT_TARGET_LANG_COLUMN),
        "source_column": config["data"]["source_column"],
        "target_column": config["data"]["target_column"],
        "max_length": train_cfg["max_length"],
        "truncation_side": train_cfg["truncation_side"],
        "packed": packed,
        "packing_strategy": train_cfg["packing_strategy"] if packed else None,
    }
    if selection_seed is not None:
        identity["selection_seed"] = selection_seed
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    safe_split = re.sub(r"[^A-Za-z0-9_.-]+", "-", split_name).strip("-") or "split"
    return Path(config["data"]["prepared_cache_dir"]) / f"{safe_split}-{digest}"


def validate_packing_config(config):
    """Reject packing modes that can merge attention or split SFT examples."""
    if config["training"]["packing_strategy"] != "bfd":
        raise ValueError("SFT packing requires training.packing_strategy='bfd'.")
    if config["model"]["attn_implementation"] != "flash_attention_3":
        raise ValueError(
            "training.packing requires model.attn_implementation=flash_attention_3 "
            "to preserve attention boundaries between packed examples."
        )


def prepare_sft_dataset(
    path,
    processor,
    config,
    split_name,
    max_examples=None,
    packed=False,
    selection_seed=None,
):
    """Tokenize and optionally pack once on rank zero, then load on every rank."""
    if packed:
        validate_packing_config(config)
    state = PartialState()
    cache_path = _prepared_cache_path(
        path, config, split_name, max_examples, packed, selection_seed
    )
    ready_path = cache_path / "_READY"
    if state.is_main_process and not ready_path.is_file():
        logger.info("Building rank-zero %s cache at %s", split_name, cache_path)
        dataset = tokenize_sft_dataset(
            load_sft_split(path, config, split_name, max_examples, selection_seed),
            processor,
            config,
            split_name,
        )
        if packed:
            # TRL's BFD packer records seq_lengths. Its padding-free collator
            # converts them to resetting position_ids, which FA3 uses as
            # document boundaries. Existing -100 completion masks are retained.
            dataset = dataset.remove_columns(
                [name for name in ("attention_mask", config["training"]["length_column_name"])
                 if name in dataset.column_names]
            )
            dataset = pack_dataset(
                dataset,
                seq_length=config["training"]["max_length"],
                strategy=config["training"]["packing_strategy"],
            )
            logger.info("Packed %s into %s blocks.", split_name, len(dataset))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{cache_path.name}-", dir=cache_path.parent) as temp_dir:
            temp_path = Path(temp_dir) / "dataset"
            dataset.save_to_disk(temp_path)
            (temp_path / "_READY").write_text("ok\n", encoding="utf-8")
            try:
                temp_path.replace(cache_path)
            except FileExistsError:
                # Another independent launch completed the identical immutable
                # cache first. Its ready marker makes it safe to reuse.
                if not ready_path.is_file():
                    raise
    if not state.is_main_process:
        # Do not use an NCCL barrier while rank zero performs CPU-bound
        # preprocessing. A cold multi-million-row cache can take longer than
        # the process group's timeout, causing every waiting rank to fail just
        # before rank zero publishes the cache. The directory rename above is
        # atomic, so the ready marker is the synchronization primitive here.
        logger.info("Waiting for rank-zero %s cache at %s", split_name, cache_path)
        while not ready_path.is_file():
            time.sleep(1)
    if not ready_path.is_file():
        raise RuntimeError(f"Prepared dataset cache was not completed: {cache_path}")
    dataset = load_from_disk(cache_path)
    logger.info("Loaded %s prepared cache: examples=%s path=%s", split_name, len(dataset), cache_path)
    return dataset


def make_sft_data_collator(processor, config):
    """Select the boundary-aware packed collator or the ordinary padded collator."""
    train_cfg = config["training"]
    if not train_cfg["packing"]:
        return TranslationDataCollator(processor.tokenizer, train_cfg["pad_to_multiple_of"])
    validate_packing_config(config)
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Packing requires a tokenizer pad_token_id or eos_token_id.")
    return DataCollatorForLanguageModeling(
        pad_token_id=pad_token_id,
        padding_free=True,
        # BFD blocks are flattened and contain no padding. Adding alignment
        # padding here would create artificial position-id resets and make
        # non-padding token accounting ambiguous.
        pad_to_multiple_of=None,
    )


def setup_processor(config):
    """Load the processor independently so data can be prepared before GPU weights."""
    return AutoProcessor.from_pretrained(
        config["model"]["base_model_id"], use_fast=True, fix_mistral_regex=False
    )


def setup_model_and_processor(config, processor=None):
    model_cfg, train_cfg = config["model"], config["training"]
    # Transformers 4.57.6 can misclassify locally bundled non-Mistral models
    # as needing its Mistral regex patch. TranslateGemma uses a single Split
    # pre-tokenizer, while that patch assumes an indexable Sequence and crashes.
    processor = processor or setup_processor(config)
    # Without an explicit dtype, transformers materialises every unquantised
    # parameter in float32 -- embeddings, norms and a 262k-row lm_head.
    dtype = resolve_dtype(model_cfg["dtype"])
    # model.dtype and training.bf16 describe the same decision from two angles
    # (parameter storage and autocast). Disagreeing values cast on every matmul.
    if (dtype == torch.bfloat16) != bool(train_cfg["bf16"]):
        raise ValueError(f"model.dtype={model_cfg['dtype']!r} contradicts training.bf16={train_cfg['bf16']}.")
    model_config = load_generation_safe_model_config(model_cfg["base_model_id"])
    load_kwargs = {
        "config": model_config,
        "generation_config": make_deterministic_generation_config(model_config, processor),
        "dtype": dtype,
        "attn_implementation": model_cfg["attn_implementation"],
    }
    if model_cfg["use_4bit"]:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=model_cfg["bnb_4bit_use_double_quant"],
            bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=dtype,
        )
    
    accelerator = Accelerator()
    logger.info("Loading %s: dtype=%s 4bit=%s attn=%s", model_cfg["base_model_id"], dtype, model_cfg["use_4bit"], model_cfg["attn_implementation"])
    model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model_id"], **load_kwargs)
    model = model.to(accelerator.device)
    if model_cfg["use_4bit"]:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_cfg["gradient_checkpointing"],
            gradient_checkpointing_kwargs=train_cfg["gradient_checkpointing_kwargs"],
        )
    elif train_cfg["gradient_checkpointing"]:
        # prepare_model_for_kbit_training does this for the quantised path. The
        # frozen embedding output must require grad or checkpointed blocks have
        # no graph to recompute through, and the LoRA adapters get no gradient.
        model.enable_input_require_grads()
    if train_cfg["gradient_checkpointing"]:
        # Gemma 3/TranslateGemma keeps the cache setting used by the decoder
        # on its nested text config. Set both levels so checkpointed training
        # never enters forward with use_cache=True.
        model.config.use_cache = False
        model.config.get_text_config().use_cache = False
    lora_cfg = LoraConfig(
        r=config["lora"]["r"], lora_alpha=config["lora"]["alpha"], lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        exclude_modules=config["lora"].get("exclude_modules"),
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    vision_targets = [
        name for name in model.targeted_module_names if "vision_tower" in name
    ]
    if config["lora"].get("exclude_modules") and vision_targets:
        raise RuntimeError(
            "LoRA unexpectedly targeted vision-tower modules despite "
            f"lora.exclude_modules: {vision_targets[:10]}"
        )
    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable parameters: %s/%s (%.2f%%)", trainable, total, 100 * trainable / total)
    return model, processor


def make_training_arguments(config, output_dir, learning_rate, epochs, has_eval_dataset, max_steps=None, group_by_length=False):
    cfg = config["training"]
    accumulation_steps = resolve_gradient_accumulation_steps(config)
    args = {
        "output_dir": str(output_dir), "per_device_train_batch_size": cfg["batch_size"],
        "per_device_eval_batch_size": cfg["eval_batch_size"], "gradient_accumulation_steps": accumulation_steps,
        "learning_rate": float(learning_rate), "num_train_epochs": epochs, "bf16": cfg["bf16"],
        "logging_steps": cfg["logging_steps"], "save_strategy": cfg["save_strategy"],
        "save_steps": cfg["save_steps"], "eval_steps": cfg["eval_steps"],
        "save_total_limit": cfg["save_total_limit"], "load_best_model_at_end": cfg["load_best_model_at_end"] and has_eval_dataset,
        "metric_for_best_model": cfg["metric_for_best_model"], "greater_is_better": cfg["greater_is_better"],
        "warmup_ratio": cfg["warmup_ratio"], "lr_scheduler_type": cfg["lr_scheduler_type"],
        "max_grad_norm": cfg["max_grad_norm"], "gradient_checkpointing": cfg["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": cfg["gradient_checkpointing_kwargs"],
        "seed": cfg["seed"], "data_seed": cfg["seed"], "dataloader_num_workers": cfg["dataloader_num_workers"],
        "dataloader_pin_memory": cfg["dataloader_pin_memory"],
        # persistent_workers is only meaningful with worker processes, and
        # TrainingArguments rejects the combination outright.
        "dataloader_persistent_workers": cfg["dataloader_persistent_workers"] and cfg["dataloader_num_workers"] > 0,
        "use_liger_kernel": cfg["use_liger_kernel"],
        # Off unless the caller tokenized the split and can therefore supply
        # length_column_name. Left on for an untokenized dataset, Trainer falls
        # back to deriving lengths from the raw features and fails.
        "group_by_length": group_by_length,
        "length_column_name": cfg["length_column_name"],
        "optim": cfg["optimizer"], "report_to": cfg["report_to"], "logging_first_step": True,
        "remove_unused_columns": False,
    }
    evaluation_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    args[evaluation_key] = cfg["evaluation_strategy"] if has_eval_dataset else "no"
    if max_steps is not None:
        args["max_steps"] = max_steps
    logger.info(
        "Distributed batch: per_device=%s world_size=%s accumulation=%s effective=%s",
        cfg["batch_size"], training_world_size(), accumulation_steps,
        cfg["batch_size"] * training_world_size() * accumulation_steps,
    )
    return TrainingArguments(**args)


def write_run_metadata(config, split_sizes):
    state = PartialState()
    if not state.is_main_process:
        return
    output_dir = Path(config["model"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    packages = ("torch", "transformers", "peft", "trl", "datasets", "accelerate")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "git_revision": revision,
        "split_sizes": split_sizes, "config": config,
        "package_versions": {name: importlib.metadata.version(name) for name in packages},
        "cuda": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    path = output_dir / config["model"]["run_metadata_filename"]
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Run metadata saved to %s", path)


def resolve_validation_max_examples(config, run_max_examples=None):
    """Combine the configured validation cap with a bounded-run cap."""
    configured_max = config["training"].get("validation_max_examples")
    if configured_max is not None and (
        not isinstance(configured_max, int)
        or isinstance(configured_max, bool)
        or configured_max <= 0
    ):
        raise ValueError(
            "training.validation_max_examples must be null or a positive integer"
        )
    if run_max_examples is None:
        return configured_max
    if configured_max is None:
        return run_max_examples
    return min(configured_max, run_max_examples)


def prepare_sft_splits(processor, config, max_examples=None):
    """Prepare the configured SFT splits before allocating model weights."""
    data_cfg = config["data"]
    train_data = prepare_sft_dataset(
        data_cfg["train_sft_dataset_path"], processor, config, "train", max_examples,
        packed=config["training"]["packing"],
    )
    validation_path = data_cfg["validation_sft_dataset_path"]
    validation_max_examples = resolve_validation_max_examples(config, max_examples)
    validation_data = (
        prepare_sft_dataset(
            validation_path,
            processor,
            config,
            "validation",
            validation_max_examples,
            packed=False,
            selection_seed=config["training"]["seed"],
        )
        if validation_path else None
    )
    return train_data, validation_data


def run_sft(model, processor, config, max_examples=None, max_steps=None, prepared_splits=None):
    model_cfg = config["model"]
    train_data, validation_data = prepared_splits or prepare_sft_splits(
        processor, config, max_examples
    )
    split_sizes = {"train": len(train_data)}
    if validation_data is not None:
        split_sizes["validation"] = len(validation_data)
    write_run_metadata(config, split_sizes)
    args = make_training_arguments(config, Path(model_cfg["output_dir"]) / model_cfg["sft_checkpoint_subdir"], config["training"]["learning_rate"], config["training"]["epochs"], validation_data is not None, max_steps, group_by_length=config["training"]["group_by_length"])
    trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=validation_data,
                      data_collator=make_sft_data_collator(processor, config),
                      callbacks=[RichLoggingCallback()])
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
    save_path = Path(model_cfg["output_dir"]) / model_cfg["sft_final_subdir"]
    trainer.save_model(save_path)
    state = PartialState()
    if state.is_main_process:
        processor.save_pretrained(save_path)
        logger.info("Best SFT adapter saved to %s", save_path)
    state.wait_for_everyone()
    return model, str(save_path)


def run_dpo(model, processor, config, max_examples=None, max_steps=None):
    data_cfg, model_cfg = config["data"], config["model"]
    dataset = load_dataset("json", data_files=data_cfg["dpo_train_dataset_path"], split="train")
    chosen_column = data_cfg.get("dpo_chosen_column", "farsi_chosen")
    rejected_column = data_cfg.get("dpo_rejected_column", "farsi_rejected")
    required = {data_cfg["source_column"], chosen_column, rejected_column}
    if missing := required - set(dataset.column_names):
        raise ValueError(f"DPO dataset is missing columns: {sorted(missing)}")
    if not len(dataset):
        raise ValueError("DPO dataset contains no examples")
    dataset = limit_dataset(dataset, max_examples, "DPO train")

    def format_dpo(example):
        source_lang, target_lang = resolve_language_pair(example, data_cfg)
        prompt = format_translategemma_message(
            example[data_cfg["source_column"]], "", source_lang, target_lang
        )[0]
        return {
            "prompt": [prompt],
            "chosen": [{"role": "assistant", "content": example[chosen_column]}],
            "rejected": [
                {"role": "assistant", "content": example[rejected_column]}
            ],
        }

    dataset = dataset.map(format_dpo, remove_columns=dataset.column_names)
    args = make_training_arguments(config, Path(model_cfg["output_dir"]) / model_cfg["dpo_checkpoint_subdir"], config["training"]["dpo_learning_rate"], config["training"]["dpo_epochs"], False, max_steps)
    trainer = DPOTrainer(model=model, args=args, beta=config["training"]["dpo_beta"], train_dataset=dataset, processing_class=processor, callbacks=[RichLoggingCallback()])
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
    save_path = Path(model_cfg["output_dir"]) / model_cfg["dpo_final_subdir"]
    trainer.save_model(save_path)
    state = PartialState()
    if state.is_main_process:
        processor.save_pretrained(save_path)
    state.wait_for_everyone()
    return str(save_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Validate enabled data and tokenization without loading the model or writing outputs.")
    modes.add_argument("--smoke-test", action="store_true", help="Run enabled stages with at most 10 examples and one training step in a temporary directory.")
    modes.add_argument("--canary", action="store_true", help="Run the configured training loop on the smaller canary dataset subset.")
    return parser.parse_args()


def validate_dpo_split(path, config, max_examples=None):
    """Load and validate the DPO schema without constructing a trainer."""
    data_cfg = config["data"]
    dataset = load_dataset("json", data_files=path, split="train")
    required = {
        data_cfg["source_column"],
        data_cfg.get("dpo_chosen_column", "farsi_chosen"),
        data_cfg.get("dpo_rejected_column", "farsi_rejected"),
    }
    if missing := required - set(dataset.column_names):
        raise ValueError(f"DPO dataset is missing columns: {sorted(missing)}")
    if not len(dataset):
        raise ValueError("DPO dataset contains no examples")
    return limit_dataset(dataset, max_examples, "DPO train")


def run_dry_run(config):
    """Check each enabled stage without model weights, output files, or training."""
    logger.info("Dry run: validating enabled pipeline stages; no model weights or outputs will be created.")
    processor = None
    if config["training"]["run_sft"]:
        if config["training"]["packing"]:
            validate_packing_config(config)
        processor = AutoProcessor.from_pretrained(
            config["model"]["base_model_id"], use_fast=True, fix_mistral_regex=False
        )
        train_data = tokenize_sft_dataset(load_sft_split(config["data"]["train_sft_dataset_path"], config, "train", 10), processor, config, "train")
        validation_path = config["data"]["validation_sft_dataset_path"]
        validation_data = (
            tokenize_sft_dataset(load_sft_split(validation_path, config, "validation", 10), processor, config, "validation")
            if validation_path else None
        )
        make_training_arguments(config, Path(config["model"]["output_dir"]) / config["model"]["sft_checkpoint_subdir"], config["training"]["learning_rate"], config["training"]["epochs"], validation_data is not None, group_by_length=config["training"]["group_by_length"])
        logger.info("SFT preflight passed: train=%s validation=%s", len(train_data), len(validation_data) if validation_data is not None else 0)
    if config["training"]["run_dpo"]:
        dataset = validate_dpo_split(config["data"]["dpo_train_dataset_path"], config, 10)
        make_training_arguments(config, Path(config["model"]["output_dir"]) / config["model"]["dpo_checkpoint_subdir"], config["training"]["dpo_learning_rate"], config["training"]["dpo_epochs"], False)
        logger.info("DPO preflight passed: train=%s", len(dataset))

    state = PartialState()
    if state.is_main_process and config["evaluation"]["run_after_training"]:
        test_data = load_sft_split(config["data"]["test_dataset_path"], config, "test", 10)
        required = {config["data"]["domain_column"]}
        if missing := required - set(test_data.column_names):
            raise ValueError(f"Test dataset is missing columns: {sorted(missing)}")
        logger.info("Evaluation preflight passed: test=%s", len(test_data))
    logger.info("Dry run passed.")


def smoke_test_config(config, output_dir):
    """Make an isolated, bounded configuration for an end-to-end smoke test."""
    smoke = copy.deepcopy(config)
    smoke["model"]["output_dir"] = str(output_dir / "artifacts")
    training = smoke["training"]
    training.update({
        "epochs": 1, "dpo_epochs": 1, "batch_size": min(training["batch_size"], 2),
        "effective_batch_size": None, "gradient_accumulation_steps": 1,
        "eval_batch_size": min(training["eval_batch_size"], 2), "logging_steps": 1,
        "evaluation_strategy": "no", "save_strategy": "no", "load_best_model_at_end": False,
        "resume_from_checkpoint": None, "dataloader_num_workers": 0,
        # Ten rows do not justify a process pool, and the pool's startup cost
        # dominates the smoke test. use_liger_kernel is deliberately left as
        # configured: whether the fused kernels bind to this model is exactly
        # the kind of failure a smoke test should surface.
        "tokenize_num_proc": 0,
    })
    evaluation = smoke["evaluation"]
    evaluation.update({
        "output_dir": str(output_dir / "evaluation"), "run_baseline": False,
        "metricx_enabled": False, "comet_enabled": False, "max_new_tokens": min(evaluation["max_new_tokens"], 16),
        "smoke_test_max_examples": 10,
    })
    return smoke


def run_pipeline(config, max_examples=None, max_steps=None):
    set_seed(config["training"]["seed"])
    processor = setup_processor(config)
    prepared_splits = (
        prepare_sft_splits(processor, config, max_examples)
        if config["training"]["run_sft"] else None
    )
    model, processor = setup_model_and_processor(config, processor=processor)
    adapter_path = None
    if config["training"]["run_sft"]:
        model, adapter_path = run_sft(
            model, processor, config, max_examples, max_steps, prepared_splits=prepared_splits
        )
    if config["training"]["run_dpo"]:
        adapter_path = run_dpo(model, processor, config, max_examples, max_steps)
        
    state = PartialState()
    if state.is_main_process and config["evaluation"]["run_after_training"]:
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from evaluate_translations import run_evaluation
        run_evaluation(config, adapter_path=adapter_path)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    else:
        setup_logging(config)
    if PartialState().is_main_process:
        log_config_summary(config)
    try:
        if args.dry_run:
            run_dry_run(config)
        elif args.smoke_test:
            with tempfile.TemporaryDirectory(prefix="translategemma-smoke-") as temp_dir:
                smoke_config = smoke_test_config(config, Path(temp_dir))
                logger.info("Smoke test: at most 10 rows per split, one training step, temporary outputs in %s", temp_dir)
                run_pipeline(smoke_config, max_examples=10, max_steps=1)
        elif args.canary:
            canary_config, max_examples, max_steps = canary_run_config(config)
            logger.info(
                "Canary training: at most %s rows per split, max_steps=%s, outputs in %s",
                max_examples,
                max_steps or "configured epochs",
                canary_config["model"]["output_dir"],
            )
            run_pipeline(canary_config, max_examples=max_examples, max_steps=max_steps)
        else:
            run_pipeline(config)
        logger.info("Pipeline complete.")
    except Exception:
        logger.exception("Pipeline failed.")
        raise
