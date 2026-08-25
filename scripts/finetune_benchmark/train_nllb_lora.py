#!/usr/bin/env python3
"""LoRA fine-tune an NLLB-200 seq2seq checkpoint on one training subset.

The repository's train.py is a decoder-only SFT pipeline (chat template, packing,
completion-only labels), none of which transfers to an encoder-decoder model, so
NLLB gets its own trainer here rather than a branch inside train.py. It stays
deliberately small: tokenize with the NLLB language tags, attach LoRA, hand the
rest to Seq2SeqTrainer.

Hyperparameters come from models.<key>.training in the sweep config, so the
NLLB and TranslateGemma arms are configured in the same file and the effective
batch size can be matched between them.

    python -m scripts.finetune_benchmark.train_nllb_lora \
        --config scripts/finetune_benchmark/sweep_config.yaml \
        --model-key nllb \
        --train data/finetune_benchmark/subsets/5k/train.jsonl \
        --validation data/finetune_benchmark/subsets/5k/validation.jsonl \
        --output-dir logs/finetune_benchmark/finetune/nllb-5k \
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finetune_benchmark.config import (  # noqa: E402
    BASE_TRAINING_CONFIG,
    load_sweep_config,
    load_yaml,
)
from logging_utils import logger, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Sweep config with the models.<key> definition.")
    parser.add_argument("--model-key", required=True, help="Key under models: in the sweep config.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, required=True)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Optimizer-step cap: the compute-matched budget, or a smoke-test limit.",
    )
    parser.add_argument(
        "--eval-save-steps", type=int, default=None,
        help="Step interval for evaluation and checkpointing. Overrides the config's epoch cadence, "
             "which a capped run may never reach.",
    )
    parser.add_argument("--max-examples", type=int, default=None, help="Smoke-test cap on training rows.")
    return parser.parse_args()


def _load_split(path: str, columns: dict[str, str], max_examples: int | None):
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=path, split="train")
    missing = [name for name in (columns["source"], columns["target"]) if name not in dataset.column_names]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {dataset.column_names}")
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    return dataset


def main() -> None:
    args = parse_args()
    sweep = load_sweep_config(args.config)
    setup_logging(sweep.raw, run_name=f"nllb_lora_{Path(args.output_dir).name}")

    model_cfg = sweep.raw["models"][args.model_key]
    training = model_cfg["training"]
    lora_cfg = training["lora"]
    data_columns = load_yaml(BASE_TRAINING_CONFIG)["data"]
    columns = {"source": data_columns["source_column"], "target": data_columns["target_column"]}

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = model_cfg["base_model_id"]
    logger.info("Loading [bold cyan]%s[/bold cyan] (%s -> %s)", model_id, model_cfg["source_lang_token"], model_cfg["target_lang_token"])

    # src_lang/tgt_lang drive the language tag the tokenizer prepends to inputs
    # and to labels. Without tgt_lang the decoder is trained to start with the
    # wrong tag and generation drifts to another language.
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, src_lang=model_cfg["source_lang_token"], tgt_lang=model_cfg["target_lang_token"]
    )
    target_token_id = tokenizer.convert_tokens_to_ids(model_cfg["target_lang_token"])
    if target_token_id == tokenizer.unk_token_id:
        raise ValueError(f"{model_id} has no token for target language {model_cfg['target_lang_token']!r}")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, dtype=torch.bfloat16 if training.get("bf16", True) else torch.float32
    )
    # Deterministic decoding defaults recorded on the checkpoint, so an adapter
    # loaded later cannot start generating Persian with an English BOS tag.
    model.generation_config.forced_bos_token_id = target_token_id
    model.config.use_cache = False  # Incompatible with gradient checkpointing.

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    if training.get("gradient_checkpointing", True):
        # LoRA freezes the embeddings, so without this the checkpointed segments
        # receive inputs that require no grad and the backward graph is dropped.
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_dataset = _load_split(args.train, columns, args.max_examples)
    eval_dataset = _load_split(args.validation, columns, args.max_examples) if args.validation else None
    logger.info("Train rows: %d, validation rows: %s", len(train_dataset), len(eval_dataset) if eval_dataset else "none")

    max_source = int(training["max_source_length"])
    max_target = int(training["max_target_length"])

    def tokenize(batch):
        encoded = tokenizer(
            batch[columns["source"]],
            text_target=batch[columns["target"]],
            max_length=max_source,
            truncation=True,
        )
        encoded["labels"] = [labels[:max_target] for labels in encoded["labels"]]
        return encoded

    remove_columns = train_dataset.column_names
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=remove_columns, desc="Tokenizing train")
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=remove_columns, desc="Tokenizing validation")

    cadence = "steps" if args.eval_save_steps else training.get("eval_strategy", "epoch")
    interval = int(args.eval_save_steps or training.get("eval_steps", 200))

    arguments = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=int(training["per_device_batch_size"]),
        per_device_eval_batch_size=int(training.get("eval_batch_size", training["per_device_batch_size"])),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(args.epochs),
        max_steps=int(args.max_steps) if args.max_steps else -1,
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        weight_decay=float(training.get("weight_decay", 0.0)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        label_smoothing_factor=float(training.get("label_smoothing_factor", 0.0)),
        bf16=bool(training.get("bf16", True)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=int(training.get("logging_steps", 25)),
        # Epoch cadence by default; a step cadence when the caller passes one,
        # because a step-capped run can stop before the first epoch ends and
        # would otherwise produce no evaluation and no checkpoint at all.
        eval_strategy=cadence if eval_dataset is not None else "no",
        eval_steps=interval,
        save_strategy=cadence if eval_dataset is not None else "no",
        save_steps=interval,
        save_total_limit=int(training.get("save_total_limit", 1)),
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=int(training.get("dataloader_num_workers", 2)),
        dataloader_pin_memory=True,
        optim=training.get("optim", "adamw_torch_fused"),
        seed=int(sweep.sweep["seed"]),
        report_to=[],
        predict_with_generate=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
        ),
        processing_class=tokenizer,
    )
    result = trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics = dict(result.metrics)
    if eval_dataset is not None:
        metrics.update(trainer.evaluate())
    metrics.update(
        {
            "model_id": model_id,
            "adapter_path": str(adapter_dir),
            "train_rows": len(train_dataset),
            "validation_rows": len(eval_dataset) if eval_dataset is not None else 0,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "effective_batch_size": int(training["per_device_batch_size"]) * int(training["gradient_accumulation_steps"]),
            "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        }
    )
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logger.info("Adapter written to [bold]%s[/bold]", adapter_dir)


if __name__ == "__main__":
    main()
