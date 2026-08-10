"""Evaluate a base TranslateGemma model or a LoRA adapter on the configured test split."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from accelerate import Accelerator
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from language_pairs import resolve_language_pair
from logging_utils import console, logger, setup_logging, log_config_summary, load_config
from train import (
    load_generation_safe_model_config,
    make_deterministic_generation_config,
    resolve_dtype,
)


def generate_translations(test_df, config, adapter_path=None, prefix="", force=False):
    model_cfg, eval_cfg, data_cfg = config["model"], config["evaluation"], config["data"]
    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f".cache_{prefix}_hypotheses.jsonl"

    if force and cache_path.exists():
        cache_path.unlink()

    cached_hypotheses = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        cached_hypotheses[entry["index"]] = entry["hypothesis"]
                    except Exception:
                        continue

    total_rows = len(test_df)
    if len(cached_hypotheses) == total_rows:
        logger.info(
            "Found all %d cached translations for [bold green]%s[/bold green]. Skipping model loading and generation.",
            total_rows,
            prefix,
        )
        return [cached_hypotheses[i] for i in range(total_rows)]

    if cached_hypotheses:
        logger.info(
            "Resuming generation for [bold green]%s[/bold green]: %d/%d completed from cache.",
            prefix,
            len(cached_hypotheses),
            total_rows,
        )

    model_name = adapter_path or model_cfg["base_model_id"]
    logger.info("Loading model for translation generation: [bold cyan]%s[/bold cyan]", model_name)
    processor = AutoProcessor.from_pretrained(
        model_cfg["base_model_id"], use_fast=True, fix_mistral_regex=False
    )
    model_config = load_generation_safe_model_config(model_cfg["base_model_id"])
    # Same model.dtype / model.attn_implementation the adapter was trained
    # under, so evaluation never silently measures a different numeric setup.
    accelerator = Accelerator()
    load_kwargs = {
        "config": model_config,
        "generation_config": make_deterministic_generation_config(model_config, processor),
        "dtype": resolve_dtype(model_cfg["dtype"]),
        "attn_implementation": model_cfg["attn_implementation"],
    }
    
    base_model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model_id"], **load_kwargs)
    base_model = base_model.to(accelerator.device)
    model = PeftModel.from_pretrained(base_model, adapter_path) if adapter_path else base_model
    model.eval()

    logger.info("Generating translations for remaining examples (%s)...", prefix)
    hypotheses = [None] * total_rows
    for i, hyp in cached_hypotheses.items():
        if i < total_rows:
            hypotheses[i] = hyp

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    with open(cache_path, "a", encoding="utf-8") as f_cache, progress:
        task = progress.add_task(f"Generating ({prefix})", total=total_rows, completed=len(cached_hypotheses))
        for i, (_, row) in enumerate(test_df.iterrows()):
            if hypotheses[i] is not None:
                continue

            source = row[data_cfg["source_column"]]
            source_lang, target_lang = resolve_language_pair(row, data_cfg)
            messages = [{"role": "user", "content": [{"type": "text", "source_lang_code": source_lang, "target_lang_code": target_lang, "text": source}]}]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                pad_token_id = processor.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = processor.tokenizer.eos_token_id
                generation_kwargs = {
                    "max_new_tokens": eval_cfg["max_new_tokens"], "do_sample": eval_cfg["do_sample"],
                    "num_beams": eval_cfg["num_beams"], "pad_token_id": pad_token_id,
                }
                if eval_cfg["do_sample"]:
                    generation_kwargs.update(temperature=eval_cfg["temperature"], top_p=eval_cfg["top_p"])
                else:
                    generation_kwargs.update(temperature=1.0, top_p=1.0, top_k=50)
                outputs = model.generate(**inputs, **generation_kwargs)
            hyp = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            hypotheses[i] = hyp
            f_cache.write(json.dumps({"index": i, "hypothesis": hyp}, ensure_ascii=False) + "\n")
            f_cache.flush()
            progress.update(task, advance=1)

    logger.info("Completed translation generation for [bold green]%s[/bold green].", prefix)
    del model, base_model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hypotheses


def evaluate_metricx(sources, hypotheses, references, config, prefix="", force=False):
    """Score with a reference-based MetricX-24 hybrid model (lower is better)."""
    try:
        from metricx24.models import MT5ForRegression
    except ImportError as error:
        raise ImportError("Install MetricX or set evaluation.metricx_enabled: false.") from error

    eval_cfg = config["evaluation"]
    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f".cache_{prefix}_metricx.jsonl"

    if force and cache_path.exists():
        cache_path.unlink()

    cached_scores = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        cached_scores[entry["index"]] = entry["score"]
                    except Exception:
                        continue

    total = len(sources)
    if len(cached_scores) == total:
        scores = [cached_scores[i] for i in range(total)]
        mean_score = float(pd.Series(scores).mean())
        logger.info(
            "Found all %d cached MetricX scores for [bold green]%s[/bold green]. Mean score: [bold yellow]%.4f[/bold yellow]",
            total,
            prefix,
            mean_score,
        )
        return scores

    if cached_scores:
        logger.info(
            "Resuming MetricX scoring for [bold green]%s[/bold green]: %d/%d completed from cache.",
            prefix,
            len(cached_scores),
            total,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading MetricX-24 model ([cyan]%s[/cyan])...", eval_cfg["metricx_model_id"])
    tokenizer = AutoTokenizer.from_pretrained(eval_cfg["metricx_tokenizer_id"])
    model = MT5ForRegression.from_pretrained(eval_cfg["metricx_model_id"], torch_dtype="auto").to(device).eval()

    scores = [None] * total
    for i, sc in cached_scores.items():
        if i < total:
            scores[i] = sc

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]{task.description}[/bold magenta]"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    with open(cache_path, "a", encoding="utf-8") as f_cache, progress:
        task = progress.add_task(f"MetricX Scoring ({prefix})", total=total, completed=len(cached_scores))
        with torch.inference_mode():
            for i, (source, hypothesis, reference) in enumerate(zip(sources, hypotheses, references)):
                if scores[i] is not None:
                    continue
                text = f"source: {source} candidate: {hypothesis} reference: {reference}"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=eval_cfg["metricx_max_length"], padding=False)
                inputs = {key: value[:, :-1].to(device) for key, value in inputs.items()}
                score = model(**inputs).predictions.item()
                scores[i] = score
                f_cache.write(json.dumps({"index": i, "score": score}, ensure_ascii=False) + "\n")
                f_cache.flush()
                progress.update(task, advance=1)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mean_score = float(pd.Series(scores).mean())
    logger.info("MetricX evaluation finished for [bold green]%s[/bold green]. Mean score (lower is better): [bold yellow]%.4f[/bold yellow]", prefix, mean_score)
    return scores


def evaluate_comet(sources, hypotheses, references, config, prefix=""):
    logger.info("Starting COMET evaluation for [bold green]%s[/bold green]...", prefix)
    from comet import download_model, load_from_checkpoint
    eval_cfg = config["evaluation"]
    model = load_from_checkpoint(download_model(eval_cfg["comet_model_id"]))
    data = [{"src": source, "mt": hypothesis, "ref": reference} for source, hypothesis, reference in zip(sources, hypotheses, references)]
    output = model.predict(data, batch_size=eval_cfg["comet_batch_size"], gpus=eval_cfg["comet_gpus"] if torch.cuda.is_available() else 0)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("COMET evaluation finished for [bold green]%s[/bold green]. System score (higher is better): [bold yellow]%.4f[/bold yellow]", prefix, output.system_score)
    return output.scores, output.system_score


def _write_human_review_sample(results, config, output_dir, prefix):
    data_cfg, eval_cfg = config["data"], config["evaluation"]
    domain_column = data_cfg["domain_column"]
    sample = (results.groupby(domain_column, group_keys=False)
              .apply(lambda group: group.sample(n=min(len(group), eval_cfg["human_review_samples_per_domain"]), random_state=eval_cfg["human_review_seed"]))
              .reset_index(drop=True))
    columns = [column for column in [data_cfg["id_column"], domain_column, data_cfg["source_column"], data_cfg["target_column"], "generated_farsi", "metricx_score", "comet_score"] if column in sample]
    sample[columns].to_csv(output_dir / f"{prefix}_{eval_cfg['human_review_filename']}", index=False)


def _run_one(config, adapter_path, prefix, force=False):
    data_cfg, eval_cfg = config["data"], config["evaluation"]
    console.print(Panel(f"[bold green]Starting Evaluation Stage: {prefix.upper()}[/bold green]\nAdapter: {adapter_path or 'Base Model'}", border_style="cyan"))
    test_df = pd.read_json(data_cfg["test_dataset_path"], lines=True)
    required = {data_cfg["source_column"], data_cfg["target_column"], data_cfg["domain_column"]}
    if missing := required - set(test_df.columns):
        raise ValueError(f"Test dataset is missing columns: {sorted(missing)}")
    if max_examples := eval_cfg.get("smoke_test_max_examples"):
        test_df = test_df.head(max_examples).copy()
        logger.info("Limiting evaluation to %s examples for smoke test.", len(test_df))
    sources, references = test_df[data_cfg["source_column"]].tolist(), test_df[data_cfg["target_column"]].tolist()
    results = test_df.copy()
    results["generated_farsi"] = generate_translations(test_df, config, adapter_path, prefix=prefix, force=force)
    summary = {"label": prefix, "examples": len(results), "adapter_path": adapter_path}
    if eval_cfg["metricx_enabled"]:
        results["metricx_score"] = evaluate_metricx(sources, results["generated_farsi"].tolist(), references, config, prefix=prefix, force=force)
        summary["metricx_mean_lower_is_better"] = float(results["metricx_score"].mean())
    if eval_cfg["comet_enabled"]:
        scores, system_score = evaluate_comet(sources, results["generated_farsi"].tolist(), references, config, prefix=prefix)
        results["comet_score"] = scores
        summary["comet_system_score_higher_is_better"] = float(system_score)
    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = output_dir / f"{prefix}_{eval_cfg['detailed_filename']}"
    sample_path = output_dir / f"{prefix}_{eval_cfg['human_review_filename']}"
    results.to_csv(detailed_path, index=False)
    logger.info("Detailed results saved to [bold]%s[/bold]", detailed_path)
    _write_human_review_sample(results, config, output_dir, prefix)
    logger.info("Human review sample saved to [bold]%s[/bold]", sample_path)
    return summary


def run_evaluation(config, adapter_path=None, force=False):
    eval_cfg = config["evaluation"]
    adapter_path = adapter_path or eval_cfg["adapter_path"]
    if not adapter_path:
        raise ValueError("Provide an adapter path or set evaluation.adapter_path before adapter evaluation.")
    summaries = []
    if eval_cfg["run_baseline"]:
        summaries.append(_run_one(config, None, eval_cfg["baseline_prefix"], force=force))
    summaries.append(_run_one(config, adapter_path, eval_cfg["adapter_prefix"], force=force))
    output_path = Path(eval_cfg["output_dir"]) / eval_cfg["summary_filename"]
    output_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print clean Rich comparison table
    table = Table(title="Evaluation Results Summary", header_style="bold yellow", border_style="green")
    table.add_column("Label", style="cyan")
    table.add_column("Examples", style="white")
    table.add_column("Adapter Path", style="dim")
    if eval_cfg["metricx_enabled"]:
        table.add_column("MetricX ↓", style="magenta")
    if eval_cfg["comet_enabled"]:
        table.add_column("COMET ↑", style="green")

    for s in summaries:
        row = [s["label"], str(s["examples"]), str(s["adapter_path"] or "Base Model")]
        if eval_cfg["metricx_enabled"]:
            row.append(f"{s.get('metricx_mean_lower_is_better', 0.0):.4f}")
        if eval_cfg["comet_enabled"]:
            row.append(f"{s.get('comet_system_score_higher_is_better', 0.0):.4f}")
        table.add_row(*row)

    console.print(table)
    logger.info("Evaluation results saved to [bold]%s[/bold]", output_path)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--force", action="store_true", help="Ignore existing cache files and re-generate from scratch.")
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config, run_name="evaluation")
    log_config_summary(config)
    run_evaluation(config, args.adapter_path, force=args.force)


if __name__ == "__main__":
    main()


