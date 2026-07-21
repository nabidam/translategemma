"""Config-driven QLoRA training for scientific English-to-Farsi TranslateGemma."""

import argparse
import copy
import importlib.metadata
import inspect
import json
import logging
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from trl import DPOTrainer

from logging_utils import logger, setup_logging, log_config_summary, load_config


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

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        model_inputs = [
            {"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]}
            for feature in features
        ]
        batch = self.tokenizer.pad(model_inputs, padding=True, return_tensors="pt")
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


def format_translategemma_message(source, target, source_lang, target_lang):
    return [
        {"role": "user", "content": [{"type": "text", "source_lang_code": source_lang, "target_lang_code": target_lang, "text": source}]},
        {"role": "assistant", "content": str(target)},
    ]


def limit_dataset(dataset, max_examples, split_name):
    """Return at most max_examples rows, preserving the dataset's order."""
    if max_examples is None or len(dataset) <= max_examples:
        return dataset
    logger.info("Limiting %s split from %s to %s examples.", split_name, len(dataset), max_examples)
    return dataset.select(range(max_examples))


def load_sft_split(path, config, split_name, max_examples=None):
    data_cfg = config["data"]
    dataset = load_dataset("json", data_files=path, split="train")
    required = {data_cfg["source_column"], data_cfg["target_column"]}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{split_name} dataset {path} is missing columns: {sorted(missing)}")
    if not len(dataset):
        raise ValueError(f"{split_name} dataset {path} contains no examples")
    dataset = limit_dataset(dataset, max_examples, split_name)
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
        messages = format_translategemma_message(
            example[data_cfg["source_column"]], example[data_cfg["target_column"]],
            data_cfg["source_lang"], data_cfg["target_lang"],
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
            "labels": [-100] * prompt_length + input_ids[prompt_length:],
            "has_target": len(input_ids) > prompt_length,
            "was_truncated": was_truncated,
        }

    tokenized = dataset.map(tokenize_example, remove_columns=dataset.column_names, desc=f"Tokenizing {split_name}")
    truncated = sum(tokenized["was_truncated"])
    without_target = len(tokenized) - sum(tokenized["has_target"])
    if without_target:
        logger.warning("%s examples in %s contain no target tokens at max_length=%s and will be excluded.", without_target, split_name, max_length)
        tokenized = tokenized.filter(lambda example: example["has_target"], desc=f"Filtering {split_name}")
    if not len(tokenized):
        raise ValueError(f"No usable {split_name} examples remain after tokenization.")
    logger.info("%s tokenized: examples=%s truncated=%s (%.2f%%), max_length=%s", split_name, len(tokenized), truncated, 100 * truncated / (len(tokenized) + without_target), max_length)
    return tokenized.remove_columns(["has_target", "was_truncated"])


def setup_model_and_processor(config):
    model_cfg, train_cfg = config["model"], config["training"]
    processor = AutoProcessor.from_pretrained(model_cfg["base_model_id"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg["use_4bit"], bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if train_cfg["bf16"] else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_id"], quantization_config=bnb_config, device_map=model_cfg["device_map"]
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_cfg["gradient_checkpointing"])
    if train_cfg["gradient_checkpointing"]:
        model.config.use_cache = False
    lora_cfg = LoraConfig(
        r=config["lora"]["r"], lora_alpha=config["lora"]["alpha"], lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable parameters: %s/%s (%.2f%%)", trainable, total, 100 * trainable / total)
    return model, processor


def make_training_arguments(config, output_dir, learning_rate, epochs, has_eval_dataset, max_steps=None):
    cfg = config["training"]
    args = {
        "output_dir": str(output_dir), "per_device_train_batch_size": cfg["batch_size"],
        "per_device_eval_batch_size": cfg["eval_batch_size"], "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "learning_rate": float(learning_rate), "num_train_epochs": epochs, "bf16": cfg["bf16"],
        "logging_steps": cfg["logging_steps"], "save_strategy": cfg["save_strategy"],
        "save_total_limit": cfg["save_total_limit"], "load_best_model_at_end": cfg["load_best_model_at_end"] and has_eval_dataset,
        "metric_for_best_model": cfg["metric_for_best_model"], "greater_is_better": cfg["greater_is_better"],
        "warmup_ratio": cfg["warmup_ratio"], "lr_scheduler_type": cfg["lr_scheduler_type"],
        "max_grad_norm": cfg["max_grad_norm"], "gradient_checkpointing": cfg["gradient_checkpointing"],
        "seed": cfg["seed"], "data_seed": cfg["seed"], "dataloader_num_workers": cfg["dataloader_num_workers"],
        "optim": cfg["optimizer"], "report_to": cfg["report_to"], "logging_first_step": True,
        "remove_unused_columns": False,
    }
    evaluation_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    args[evaluation_key] = cfg["evaluation_strategy"] if has_eval_dataset else "no"
    if max_steps is not None:
        args["max_steps"] = max_steps
    return TrainingArguments(**args)


def write_run_metadata(config, split_sizes):
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
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    path = output_dir / config["model"]["run_metadata_filename"]
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Run metadata saved to %s", path)


def run_sft(model, processor, config, max_examples=None, max_steps=None):
    model_cfg, data_cfg = config["model"], config["data"]
    train_data = tokenize_sft_dataset(load_sft_split(data_cfg["train_sft_dataset_path"], config, "train", max_examples), processor, config, "train")
    validation_path = data_cfg["validation_sft_dataset_path"]
    validation_data = (
        tokenize_sft_dataset(load_sft_split(validation_path, config, "validation", max_examples), processor, config, "validation")
        if validation_path else None
    )
    split_sizes = {"train": len(train_data)}
    if validation_data is not None:
        split_sizes["validation"] = len(validation_data)
    write_run_metadata(config, split_sizes)
    args = make_training_arguments(config, Path(model_cfg["output_dir"]) / model_cfg["sft_checkpoint_subdir"], config["training"]["learning_rate"], config["training"]["epochs"], validation_data is not None, max_steps)
    trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=validation_data,
                      data_collator=TranslationDataCollator(processor.tokenizer), callbacks=[RichLoggingCallback()])
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
    save_path = Path(model_cfg["output_dir"]) / model_cfg["sft_final_subdir"]
    trainer.save_model(save_path)
    processor.save_pretrained(save_path)
    logger.info("Best SFT adapter saved to %s", save_path)
    return model, str(save_path)


def run_dpo(model, processor, config, max_examples=None, max_steps=None):
    data_cfg, model_cfg = config["data"], config["model"]
    dataset = load_dataset("json", data_files=data_cfg["dpo_train_dataset_path"], split="train")
    required = {data_cfg["source_column"], "farsi_chosen", "farsi_rejected"}
    if missing := required - set(dataset.column_names):
        raise ValueError(f"DPO dataset is missing columns: {sorted(missing)}")
    if not len(dataset):
        raise ValueError("DPO dataset contains no examples")
    dataset = limit_dataset(dataset, max_examples, "DPO train")
    def format_dpo(example):
        prompt = format_translategemma_message(example[data_cfg["source_column"]], "", data_cfg["source_lang"], data_cfg["target_lang"])[0]
        return {"prompt": [prompt], "chosen": [{"role": "assistant", "content": example["farsi_chosen"]}], "rejected": [{"role": "assistant", "content": example["farsi_rejected"]}]}
    dataset = dataset.map(format_dpo, remove_columns=dataset.column_names)
    args = make_training_arguments(config, Path(model_cfg["output_dir"]) / model_cfg["dpo_checkpoint_subdir"], config["training"]["dpo_learning_rate"], config["training"]["dpo_epochs"], False, max_steps)
    trainer = DPOTrainer(model=model, args=args, beta=config["training"]["dpo_beta"], train_dataset=dataset, processing_class=processor, callbacks=[RichLoggingCallback()])
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
    save_path = Path(model_cfg["output_dir"]) / model_cfg["dpo_final_subdir"]
    trainer.save_model(save_path)
    processor.save_pretrained(save_path)
    return str(save_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Validate enabled data and tokenization without loading the model or writing outputs.")
    modes.add_argument("--smoke-test", action="store_true", help="Run enabled stages with at most 10 examples and one training step in a temporary directory.")
    return parser.parse_args()


def validate_dpo_split(path, config, max_examples=None):
    """Load and validate the DPO schema without constructing a trainer."""
    data_cfg = config["data"]
    dataset = load_dataset("json", data_files=path, split="train")
    required = {data_cfg["source_column"], "farsi_chosen", "farsi_rejected"}
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
        processor = AutoProcessor.from_pretrained(config["model"]["base_model_id"])
        train_data = tokenize_sft_dataset(load_sft_split(config["data"]["train_sft_dataset_path"], config, "train", 10), processor, config, "train")
        validation_path = config["data"]["validation_sft_dataset_path"]
        validation_data = (
            tokenize_sft_dataset(load_sft_split(validation_path, config, "validation", 10), processor, config, "validation")
            if validation_path else None
        )
        make_training_arguments(config, Path(config["model"]["output_dir"]) / config["model"]["sft_checkpoint_subdir"], config["training"]["learning_rate"], config["training"]["epochs"], validation_data is not None)
        logger.info("SFT preflight passed: train=%s validation=%s", len(train_data), len(validation_data) if validation_data is not None else 0)
    if config["training"]["run_dpo"]:
        dataset = validate_dpo_split(config["data"]["dpo_train_dataset_path"], config, 10)
        make_training_arguments(config, Path(config["model"]["output_dir"]) / config["model"]["dpo_checkpoint_subdir"], config["training"]["dpo_learning_rate"], config["training"]["dpo_epochs"], False)
        logger.info("DPO preflight passed: train=%s", len(dataset))
    if config["evaluation"]["run_after_training"]:
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
        "eval_batch_size": min(training["eval_batch_size"], 2), "logging_steps": 1,
        "evaluation_strategy": "no", "save_strategy": "no", "load_best_model_at_end": False,
        "resume_from_checkpoint": None, "dataloader_num_workers": 0,
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
    model, processor = setup_model_and_processor(config)
    adapter_path = None
    if config["training"]["run_sft"]:
        model, adapter_path = run_sft(model, processor, config, max_examples, max_steps)
    if config["training"]["run_dpo"]:
        adapter_path = run_dpo(model, processor, config, max_examples, max_steps)
    if config["evaluation"]["run_after_training"]:
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
    log_config_summary(config)
    try:
        if args.dry_run:
            run_dry_run(config)
        elif args.smoke_test:
            with tempfile.TemporaryDirectory(prefix="translategemma-smoke-") as temp_dir:
                smoke_config = smoke_test_config(config, Path(temp_dir))
                logger.info("Smoke test: at most 10 rows per split, one training step, temporary outputs in %s", temp_dir)
                run_pipeline(smoke_config, max_examples=10, max_steps=1)
        else:
            run_pipeline(config)
        logger.info("Pipeline complete.")
    except Exception:
        logger.exception("Pipeline failed.")
        raise
