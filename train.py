import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
    is_torch_available,
)
from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DPOTrainer

from logging_utils import (
    console,
    logger,
    setup_logging,
    log_config_summary,
    load_config,
)


class RichLoggingCallback(TrainerCallback):
    """Forward Trainer metric logs into our logger so they hit both console and file."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        epoch = state.epoch if state.epoch is not None else 0.0
        parts = []
        for k, v in logs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.6g}")
            else:
                parts.append(f"{k}={v}")
        logger.info(f"[step={step} epoch={epoch:.2f}] " + " ".join(parts))

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info(
            f"Train begin: {state.max_steps} total steps, "
            f"epochs={state.num_train_epochs}, batch={args.per_device_train_batch_size}"
        )

    def on_train_end(self, args, state, control, **kwargs):
        epoch_val = float(state.epoch) if state.epoch is not None else 0.0
        logger.info(
            f"Train end: total_steps={state.global_step}, epoch={epoch_val:.4f}"
        )

    def on_epoch_begin(self, args, state, control, **kwargs):
        e = state.epoch if state.epoch is not None else 0.0
        logger.info(f"-- Epoch {e + 1:.2f} begin --")

    def on_epoch_end(self, args, state, control, **kwargs):
        e = state.epoch if state.epoch is not None else 0.0
        logger.info(f"-- Epoch {e:.2f} end --")


def format_translategemma_message(en_text, fa_text, source_lang="en", target_lang="fa"):
    """
    Constructs the highly specific JSON-like chat template required by TranslateGemma.
    Translategemma crashes if this structure is violated.
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang,
                    "target_lang_code": target_lang,
                    "text": en_text,
                }
            ],
        },
        {"role": "assistant", "content": str(fa_text)},
    ]


def prepare_sft_dataset(dataset, config):
    """Maps raw English/Farsi columns to the 'messages' array required by the processor."""

    def map_to_messages(example):
        example["messages"] = format_translategemma_message(
            example["english"],
            example["farsi"],
            config["data"]["source_lang"],
            config["data"]["target_lang"],
        )
        return example

    logger.info("Mapping SFT dataset to messages...")
    return dataset.map(map_to_messages)


def render_sft_dataset(dataset, processor):
    """Render TranslateGemma messages before TRL can normalize multimodal content.

    TranslateGemma requires user content to be a structured list but assistant content
    to be a raw string. TRL's VLM preprocessing converts all message contents to
    structured lists, which makes TranslateGemma's chat template reject assistant
    responses. Supplying a rendered ``text`` column avoids that conversion.
    """

    def render_messages(example):
        return {
            "text": processor.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    logger.info("Rendering SFT messages with TranslateGemma's chat template...")
    return dataset.map(render_messages, remove_columns=dataset.column_names)


def prepare_dpo_dataset(dataset, config):
    """
    DPO requires a prompt (user message), a chosen response, and a rejected response.
    We format the prompt using TranslateGemma's schema but omit the assistant turn.
    """

    def format_dpo(example):
        user_message = format_translategemma_message(
            example["english"],
            "",  # Empty because we only want the prompt
            config["data"]["source_lang"],
            config["data"]["target_lang"],
        )[
            0
        ]  # Grab just the user dict

        return {
            "prompt": [user_message],
            # DPO trainer expects the raw text for chosen/rejected, not full message dicts
            "chosen": [{"role": "assistant", "content": example["farsi_chosen"]}],
            "rejected": [{"role": "assistant", "content": example["farsi_rejected"]}],
        }

    logger.info("Mapping DPO dataset...")
    return dataset.map(format_dpo)


def setup_model_and_processor(config):
    """Loads the model with QLoRA configuration."""
    logger.info(
        f"Loading processor from [bold]{config['model']['base_model_id']}[/bold]"
    )
    processor = AutoProcessor.from_pretrained(config["model"]["base_model_id"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["use_4bit"],
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if config["training"]["bf16"] else torch.float16
        ),
    )

    logger.info("Loading base model with QLoRA (4-bit) quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model_id"],
        quantization_config=bnb_config,
        device_map=config["model"]["device_map"],
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    try:
        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            f"Trainable params: {trainable}/{total} ({100 * trainable / total:.2f}%)"
        )
    except Exception:
        pass
    return model, processor


def run_sft(model, processor, config):
    logger.info(
        "[bold green]=== Starting Supervised Fine-Tuning (SFT) ===[/bold green]"
    )
    dataset = load_dataset(
        "json", data_files=config["data"]["sft_dataset_path"], split="train"
    )
    logger.info(f"SFT dataset loaded: {len(dataset)} examples")
    dataset = prepare_sft_dataset(dataset, config)
    dataset = render_sft_dataset(dataset, processor)

    training_args = TrainingArguments(
        output_dir=f"{config['model']['output_dir']}/sft",
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        num_train_epochs=config["training"]["epochs"],
        bf16=config["training"]["bf16"],
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        disable_tqdm=False,
        report_to="none",
        logging_first_step=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        # The dataset already contains fully rendered text. This prevents TRL's VLM
        # preprocessing from converting assistant content into a list.
        processing_class=processor,
        callbacks=[RichLoggingCallback()],
    )

    trainer.train()
    save_path = f"{config['model']['output_dir']}/sft_final"
    model.save_pretrained(save_path)
    logger.info(f"SFT adapter saved to [bold]{save_path}[/bold]")
    return model


def run_dpo(model, processor, config):
    logger.info(
        "[bold green]=== Starting Direct Preference Optimization (DPO) ===[/bold green]"
    )
    dataset = load_dataset(
        "json", data_files=config["data"]["dpo_dataset_path"], split="train"
    )
    logger.info(f"DPO dataset loaded: {len(dataset)} examples")
    dataset = prepare_dpo_dataset(dataset, config)

    # We use the SFT model as both the reference and the model being trained.
    # DPO handles the cloning of the reference model internally.
    training_args = TrainingArguments(
        output_dir=f"{config['model']['output_dir']}/dpo",
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(
            config["training"]["dpo_learning_rate"]
        ),  # DPO requires much lower LR
        num_train_epochs=config["training"]["epochs"],
        bf16=config["training"]["bf16"],
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        disable_tqdm=False,
        report_to="none",
        logging_first_step=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=0.1,  # KL penalty margin - 0.1 is standard
        train_dataset=dataset,
        processing_class=processor,
        callbacks=[RichLoggingCallback()],
    )

    trainer.train()
    save_path = f"{config['model']['output_dir']}/dpo_final"
    model.save_pretrained(save_path)
    logger.info(f"DPO adapter saved to [bold]{save_path}[/bold]")


if __name__ == "__main__":
    config = load_config()
    setup_logging(config)
    log_config_summary(config)

    try:
        model, processor = setup_model_and_processor(config)

        if config["training"]["run_sft"]:
            model = run_sft(model, processor, config)

        if config["training"]["run_dpo"]:
            # Run DPO using the weights just updated by SFT
            run_dpo(model, processor, config)

        logger.info("[bold green]Pipeline Complete.[/bold green] Final adapter saved.")
    except Exception:
        logger.exception("Pipeline failed.")
        raise
