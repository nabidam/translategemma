import logging

import torch
import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from peft import PeftModel
from comet import download_model, load_from_checkpoint

from logging_utils import console, logger, setup_logging, log_config_summary, load_config

# NOTE: MetricX uses a custom T5 regression architecture.
# You need the official MetricX codebase to load it properly:
# pip install git+https://github.com/google-research/metricx.git
try:
    from metricx23.models import MT5ForRegression
except ImportError:
    raise ImportError(
        "Please install metricx: pip install git+https://github.com/google-research/metricx.git"
    )


def _make_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def generate_translations(test_df, config):
    logger.info(f"Loading Base Model: [bold]{config['model']['base_model_id']}[/bold]")
    processor = AutoProcessor.from_pretrained(config["model"]["base_model_id"])

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model_id"],
        device_map=config["model"]["device_map"],
        torch_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float16,
    )

    logger.info(f"Applying LoRA Adapter from: [bold]{config['evaluation']['adapter_path']}[/bold]")
    model = PeftModel.from_pretrained(base_model, config["evaluation"]["adapter_path"])

    hypotheses = []
    source_lang = config["data"]["source_lang"]
    target_lang = config["data"]["target_lang"]
    total = len(test_df)

    logger.info(f"Generating Farsi translations for {total} segments...")
    with _make_progress() as progress:
        task = progress.add_task("Translating", total=total)
        for idx, row in test_df.iterrows():
            # Strict TranslateGemma formatting
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": source_lang,
                            "target_lang_code": target_lang,
                            "text": row["english"],
                        }
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,  # Always greedy decoding for eval
                )

            input_len = inputs["input_ids"].shape[-1]
            translation = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
            hypotheses.append(translation)

            progress.update(task, advance=1)
            if idx > 0 and idx % 50 == 0:
                logger.debug(f"Translated {idx}/{total}")

    # Clean up VRAM to make room for evaluation models
    del model, base_model, processor
    torch.cuda.empty_cache()
    logger.info("Translation generation complete; VRAM cleared.")
    return hypotheses


def evaluate_metricx(sources, hypotheses, references, config):
    logger.info(f"Loading MetricX: [bold]{config['evaluation']['metricx_model_id']}[/bold]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"MetricX device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["evaluation"]["metricx_model_id"])
    model = MT5ForRegression.from_pretrained(
        config["evaluation"]["metricx_model_id"]
    ).to(device)
    model.eval()

    scores = []
    total = len(sources)
    logger.info("Scoring with MetricX-24 (Lower is better, [0-25])...")
    with _make_progress() as progress:
        task = progress.add_task("MetricX", total=total)
        with torch.no_grad():
            for src, hyp, ref in zip(sources, hypotheses, references):
                # MetricX-24 Hybrid expects inputs formatted as: "ref: {ref} hyp: {hyp} src: {src}"
                input_text = f"ref: {ref} hyp: {hyp} src: {src}"
                inputs = tokenizer(
                    input_text, return_tensors="pt", truncation=True, max_length=1024
                ).to(device)

                output = model(**inputs)
                # MetricX-24 scores are strictly clamped between 0 and 25
                score = torch.clamp(output.logits, min=0.0, max=25.0).item()
                scores.append(score)
                progress.update(task, advance=1)

    del model, tokenizer
    torch.cuda.empty_cache()
    logger.info(f"MetricX done. mean={sum(scores)/len(scores):.4f}" if scores else "MetricX done (empty)")
    return scores


def evaluate_comet(sources, hypotheses, references, config):
    logger.info(f"Loading COMET: [bold]{config['evaluation']['comet_model_id']}[/bold]")
    model_path = download_model(config["evaluation"]["comet_model_id"])
    model = load_from_checkpoint(model_path)

    data = [
        {"src": src, "mt": hyp, "ref": ref}
        for src, hyp, ref in zip(sources, hypotheses, references)
    ]

    logger.info("Scoring with COMET (Higher is better, [0-1])...")
    device = 1 if torch.cuda.is_available() else 0
    logger.info(f"COMET gpus={device}, batch_size={config['evaluation']['comet_batch_size']}")
    model_output = model.predict(
        data, batch_size=config["evaluation"]["comet_batch_size"], gpus=device
    )

    del model
    torch.cuda.empty_cache()
    logger.info(f"COMET done. system_score={model_output.system_score:.4f}")
    return model_output.scores, model_output.system_score


if __name__ == "__main__":
    config = load_config()
    setup_logging(config, run_name=None)
    log_config_summary(config)

    try:
        test_path = config["data"]["test_dataset_path"]
        logger.info(f"Loading test data from [bold]{test_path}[/bold]")
        test_df = pd.read_json(test_path, lines=True)
        logger.info(f"Loaded {len(test_df)} test segments")

        sources = test_df["english"].tolist()
        references = test_df["farsi"].tolist()

        # 1. Generate translations
        hypotheses = generate_translations(test_df, config)
        test_df["generated_farsi"] = hypotheses

        # 2. Score with MetricX
        metricx_scores = evaluate_metricx(sources, hypotheses, references, config)
        test_df["metricx_score"] = metricx_scores

        # 3. Score with COMET
        comet_scores, comet_system_score = evaluate_comet(
            sources, hypotheses, references, config
        )
        test_df["comet_score"] = comet_scores

        # 4. Save and summarize
        output_file = config["evaluation"]["output_file"]
        test_df.to_csv(output_file, index=False)
        logger.info(f"Saved detailed segment scores to [bold]{output_file}[/bold]")

        console.rule("[bold green]EVALUATION COMPLETE[/bold green]")
        avg_metricx = sum(metricx_scores) / len(metricx_scores) if metricx_scores else 0.0
        logger.info(
            f"MetricX System Score (0-25, LOWER is better): [bold]{avg_metricx:.4f}[/bold]"
        )
        logger.info(
            f"COMET System Score (0-1, HIGHER is better): [bold]{comet_system_score:.4f}[/bold]"
        )
    except Exception:
        logger.exception("Evaluation failed.")
        raise
