#!/usr/bin/env python3
"""Find a fast, safe validation batch size with one tiny SFT run.

This uses the normal model, data, LoRA, tokenization, packing, collator, and
Trainer settings from config.yaml. It trains on a very small prefix of the
configured SFT train split, then evaluates that same model on the configured
validation split for each candidate per-device evaluation batch size.

The held-out test split and evaluate_translations.py are deliberately not used.
Run with the same Accelerate configuration as the intended training job, for
example:

    accelerate launch --config_file accelerate_configs/h200_1gpu.yaml \
      scripts/benchmark_eval_batch.py --config config.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accelerate import PartialState
from transformers import Trainer, set_seed

from logging_utils import logger, setup_logging, load_config
from train import (
    make_sft_data_collator,
    make_training_arguments,
    prepare_sft_dataset,
    setup_model_and_processor,
    setup_processor,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--eval-batch-sizes",
        nargs="+",
        type=positive_int,
        help="Candidate per-device validation batch sizes; overrides config.yaml.",
    )
    parser.add_argument(
        "--train-max-examples",
        type=positive_int,
        help="Tiny SFT train prefix size; overrides eval_batch_search.train_max_examples.",
    )
    parser.add_argument(
        "--validation-max-examples",
        type=positive_int,
        help="Optional validation prefix size; default evaluates the full validation split.",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        help="Tiny SFT optimizer steps; overrides eval_batch_search.max_steps.",
    )
    parser.add_argument(
        "--output-dir",
        help="Result directory; overrides eval_batch_search.output_dir.",
    )
    return parser.parse_args()


def resolve_settings(config: dict, args: argparse.Namespace) -> dict:
    search = copy.deepcopy(config.get("eval_batch_search") or {})
    candidates = args.eval_batch_sizes or search.get("eval_batch_sizes")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("eval_batch_search.eval_batch_sizes must be a non-empty list")
    candidates = sorted(set(positive_int(str(value)) for value in candidates))

    train_max_examples = args.train_max_examples
    if train_max_examples is None:
        train_max_examples = search.get("train_max_examples", 1)
    validation_max_examples = args.validation_max_examples
    if validation_max_examples is None:
        validation_max_examples = search.get("validation_max_examples")
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = search.get("max_steps", 1)
    output_dir = args.output_dir or search.get(
        "output_dir", "logs/eval_batch_search"
    )

    for name, value in (
        ("train_max_examples", train_max_examples),
        ("max_steps", max_steps),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"eval_batch_search.{name} must be a positive integer")
    if validation_max_examples is not None and (
        not isinstance(validation_max_examples, int)
        or isinstance(validation_max_examples, bool)
        or validation_max_examples <= 0
    ):
        raise ValueError(
            "eval_batch_search.validation_max_examples must be null or a positive integer"
        )
    if not output_dir:
        raise ValueError("eval_batch_search.output_dir must not be empty")

    return {
        "eval_batch_sizes": candidates,
        "train_max_examples": train_max_examples,
        "validation_max_examples": validation_max_examples,
        "max_steps": max_steps,
        "output_dir": str(output_dir),
    }


def prepare_search_config(config: dict, output_dir: Path) -> dict:
    prepared = copy.deepcopy(config)
    training = prepared["training"]
    if not training.get("run_sft"):
        raise ValueError("Training batch search requires training.run_sft: true")
    training.update(
        {
            "run_dpo": False,
            "evaluation_strategy": "no",
            "save_strategy": "no",
            "load_best_model_at_end": False,
            "resume_from_checkpoint": None,
            "report_to": "none",
        }
    )
    prepared.setdefault("evaluation", {})["run_after_training"] = False
    prepared["model"]["output_dir"] = str(output_dir / "trainer")
    return prepared


def clear_eval_dataloader_cache(trainer: Trainer) -> None:
    """Ensure changing eval batch size creates a new DataLoader if supported."""
    # Transformers 4.57 caches persistent evaluation loaders by dataset key in
    # the plural `_eval_dataloaders` mapping. Clear it so the next evaluate()
    # call constructs a loader with the newly selected batch size.
    eval_dataloaders = trainer.__dict__.get("_eval_dataloaders")
    if isinstance(eval_dataloaders, dict):
        eval_dataloaders.clear()

    # Retain compatibility with versions that used a singular cached loader.
    if "_eval_dataloader" in trainer.__dict__:
        trainer._eval_dataloader = None


def measure_evaluation(trainer: Trainer, batch_size: int) -> dict:
    trainer.args.per_device_eval_batch_size = batch_size
    clear_eval_dataloader_cache(trainer)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        metrics = trainer.evaluate(metric_key_prefix=f"eval_bs_{batch_size}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak_memory = (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else None
        )
        return {
            "status": "success",
            "eval_batch_size": batch_size,
            "elapsed_seconds": elapsed,
            "peak_memory_allocated_gib": peak_memory,
            "metrics": metrics,
        }
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "status": "oom",
            "eval_batch_size": batch_size,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else None
            ),
            "error": str(error).splitlines()[0],
        }


def validation_loss(result: dict) -> float | None:
    metrics = result.get("metrics") or {}
    for key, value in metrics.items():
        if key.endswith("_loss") and isinstance(value, (int, float)):
            return float(value)
    return None


def render_summary(
    settings: dict, results: list[dict], report_path: Path, validation_rows: int
) -> None:
    console = Console()
    table = Table(title="Evaluation batch-size sweep", show_lines=True)
    table.add_column("Status", justify="center", no_wrap=True)
    table.add_column("Per-device batch", justify="right")
    table.add_column("Validation loss", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Peak VRAM", justify="right")
    table.add_column("Validation rows", justify="right")

    for result in results:
        status = result["status"]
        status_text = "[green]OK[/green]" if status == "success" else "[red]OOM[/red]"
        loss = validation_loss(result)
        peak = result.get("peak_memory_allocated_gib")
        table.add_row(
            status_text,
            str(result["eval_batch_size"]),
            f"{loss:.6f}" if loss is not None else "-",
            f"{result['elapsed_seconds']:.2f}s",
            f"{peak:.2f} GiB" if peak is not None else "-",
            str(validation_rows),
        )

    successful = [result for result in results if result["status"] == "success"]
    fastest = min(successful, key=lambda result: result["elapsed_seconds"]) if successful else None
    recommendation = (
        f"[bold green]Recommended per-device eval batch: {fastest['eval_batch_size']}[/bold green]\n"
        f"Fastest successful validation: {fastest['elapsed_seconds']:.2f}s"
        if fastest
        else "[bold red]No candidate completed successfully.[/bold red]"
    )
    console.print(table)
    console.print(
        Panel(
            f"{recommendation}\n"
            f"Candidates: {', '.join(str(value) for value in settings['eval_batch_sizes'])}\n"
            f"Report: {report_path}",
            title="Summary",
            border_style="cyan",
        )
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    settings = resolve_settings(config, args)
    output_dir = Path(settings["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    setup_logging(config, run_name="eval-batch-search")
    state = PartialState()
    if state.is_main_process:
        logger.info(
            "Evaluation batch search: train_examples=%s max_steps=%s candidates=%s",
            settings["train_max_examples"],
            settings["max_steps"],
            settings["eval_batch_sizes"],
        )

    search_config = prepare_search_config(config, output_dir)
    set_seed(search_config["training"]["seed"])
    processor = setup_processor(search_config)
    train_data = prepare_sft_dataset(
        search_config["data"]["train_sft_dataset_path"],
        processor,
        search_config,
        "eval batch search train",
        settings["train_max_examples"],
        packed=search_config["training"]["packing"],
    )
    validation_path = search_config["data"].get("validation_sft_dataset_path")
    if not validation_path:
        raise ValueError(
            "data.validation_sft_dataset_path is required; this search needs validation data"
        )
    validation_data = prepare_sft_dataset(
        validation_path,
        processor,
        search_config,
        "eval batch search validation",
        settings["validation_max_examples"],
        packed=False,
    )
    model, processor = setup_model_and_processor(search_config, processor=processor)

    # Use the configured training batch/effective batch math for the tiny SFT
    # step. Only the evaluation batch size changes during the sweep.
    training_args = make_training_arguments(
        search_config,
        output_dir / "trainer",
        search_config["training"]["learning_rate"],
        search_config["training"]["epochs"],
        True,
        max_steps=settings["max_steps"],
        group_by_length=search_config["training"]["group_by_length"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=make_sft_data_collator(processor, search_config),
    )

    if state.is_main_process:
        logger.info("Running tiny SFT prelude: train_rows=%s", len(train_data))
    trainer.train()
    state.wait_for_everyone()

    results = []
    for batch_size in settings["eval_batch_sizes"]:
        if state.is_main_process:
            logger.info("Evaluating with per-device batch size=%s", batch_size)
        result = measure_evaluation(trainer, batch_size)
        results.append(result)
        if result["status"] == "oom":
            logger.warning(
                "Evaluation OOM at batch size %s; stopping the ascending sweep.",
                batch_size,
            )
            break
        state.wait_for_everyone()

    if not state.is_main_process:
        return
    successful = [result for result in results if result["status"] == "success"]
    fastest = min(successful, key=lambda result: result["elapsed_seconds"]) if successful else None
    report = {
        "schema_version": 1,
        "config_path": str(config_path),
        "config": search_config,
        "search": settings,
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "world_size": state.num_processes,
        "results": results,
        "recommended_eval_batch_size": fastest["eval_batch_size"] if fastest else None,
        "recommendation_reason": "fastest successful validation run"
        if fastest
        else "no candidate completed successfully",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "eval_batch_results.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    render_summary(settings, results, report_path, len(validation_data))
    logger.info("Evaluation batch search report: %s", report_path)
    if fastest:
        logger.info(
            "Recommended eval batch size=%s (%.2fs, peak %.2f GiB)",
            fastest["eval_batch_size"],
            fastest["elapsed_seconds"],
            fastest["peak_memory_allocated_gib"] or 0.0,
        )


if __name__ == "__main__":
    main()
