#!/usr/bin/env python3
"""Benchmark the configured SFT training path across several local GPU counts.

The parent process launches one isolated Accelerate job per requested GPU count.
Each job uses its matching accelerate_configs profile plus the model, data,
LoRA, and training settings from config.yaml. The benchmark section (or CLI
flags) only selects the profiles, bounds the work, and reports the run.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Training configuration file.")
    parser.add_argument(
        "--gpu-counts",
        nargs="+",
        type=positive_int,
        help="GPU counts to compare; overrides benchmark.gpu_counts.",
    )
    parser.add_argument("--max-examples", type=positive_int, help="Override benchmark.max_examples.")
    parser.add_argument("--max-steps", type=positive_int, help="Override benchmark.max_steps.")
    parser.add_argument(
        "--per-device-batch-size",
        type=positive_int,
        help="Override benchmark.per_device_batch_size.",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=positive_int,
        help="Override benchmark.effective_batch_size.",
    )
    parser.add_argument("--output-dir", help="Override benchmark.output_dir.")
    parser.add_argument(
        "--accelerate-config-pattern",
        help=(
            "Accelerate profile path pattern; overrides "
            "benchmark.accelerate_config_pattern. Use {gpu_count} as the placeholder."
        ),
    )
    parser.add_argument(
        "--devices",
        help="Comma-separated physical GPU IDs exposed to every run, for example 0,1,2,3.",
    )
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument(
        "--run-evaluation", dest="run_evaluation", action="store_true",
        help="Measure validation loss after each timed training run.",
    )
    evaluation.add_argument(
        "--no-run-evaluation", dest="run_evaluation", action="store_false",
        help="Skip the post-training validation pass.",
    )
    parser.set_defaults(run_evaluation=None)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved Accelerate commands without loading a model.",
    )

    # Internal Accelerate worker arguments. They are intentionally hidden from
    # normal help because users invoke the parent process only.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-count", type=positive_int, help=argparse.SUPPRESS)
    parser.add_argument("--result-path", help=argparse.SUPPRESS)
    parser.add_argument("--accelerate-config", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def resolve_benchmark(config: dict, args: argparse.Namespace) -> dict:
    benchmark = copy.deepcopy(config.get("benchmark") or {})
    overrides = {
        "gpu_counts": args.gpu_counts,
        "max_examples": args.max_examples,
        "max_steps": args.max_steps,
        "per_device_batch_size": args.per_device_batch_size,
        "effective_batch_size": args.effective_batch_size,
        "output_dir": args.output_dir,
        "accelerate_config_pattern": args.accelerate_config_pattern,
        "run_evaluation": args.run_evaluation,
        "devices": args.devices,
    }
    for key, value in overrides.items():
        if value is not None:
            benchmark[key] = value

    benchmark.setdefault("gpu_counts", [1])
    benchmark.setdefault("max_examples", 20_000)
    benchmark.setdefault("max_steps", 200)
    benchmark.setdefault("warmup_steps", 5)
    benchmark.setdefault("per_device_batch_size", 6)
    benchmark.setdefault("effective_batch_size", 48)
    benchmark.setdefault("run_evaluation", True)
    benchmark.setdefault("output_dir", "logs/training_benchmark")
    benchmark.setdefault("report_filename", "benchmark_results.json")
    benchmark.setdefault(
        "accelerate_config_pattern", "accelerate_configs/h200_{gpu_count}gpu.yaml"
    )
    benchmark.setdefault("devices", None)

    counts = benchmark["gpu_counts"]
    if not isinstance(counts, list) or not counts:
        raise ValueError("benchmark.gpu_counts must be a non-empty list")
    if any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 for count in counts):
        raise ValueError("benchmark.gpu_counts must contain positive integers")
    benchmark["gpu_counts"] = list(dict.fromkeys(counts))
    for key in (
        "max_examples",
        "max_steps",
        "warmup_steps",
        "per_device_batch_size",
        "effective_batch_size",
    ):
        value = benchmark[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"benchmark.{key} must be a positive integer")
    if not benchmark["output_dir"] or not benchmark["report_filename"]:
        raise ValueError("benchmark.output_dir and benchmark.report_filename must be configured")
    pattern = benchmark["accelerate_config_pattern"]
    if not isinstance(pattern, str) or "{gpu_count}" not in pattern:
        raise ValueError(
            "benchmark.accelerate_config_pattern must contain the {gpu_count} placeholder"
        )
    return benchmark


def gradient_accumulation_steps(settings: dict, gpu_count: int) -> int:
    """Derive accumulation while holding the effective global batch constant."""
    denominator = settings["per_device_batch_size"] * gpu_count
    effective_batch_size = settings["effective_batch_size"]
    if effective_batch_size % denominator:
        raise ValueError(
            "benchmark.effective_batch_size must be exactly divisible by "
            "benchmark.per_device_batch_size * gpu_count; "
            f"got {effective_batch_size} / ({settings['per_device_batch_size']} * {gpu_count})"
        )
    accumulation = effective_batch_size // denominator
    if accumulation < 1:
        raise ValueError(
            f"benchmark.effective_batch_size={effective_batch_size} is smaller than the "
            f"{denominator}-sample batch produced by {gpu_count} GPUs"
        )
    return accumulation


def resolve_accelerate_config(config_path: Path, settings: dict, gpu_count: int) -> Path:
    """Resolve and validate the Accelerate profile for one matrix entry."""
    rendered = settings["accelerate_config_pattern"].format(gpu_count=gpu_count)
    accelerate_config = Path(rendered).expanduser()
    if not accelerate_config.is_absolute():
        accelerate_config = config_path.parent / accelerate_config
    accelerate_config = accelerate_config.resolve()
    if not accelerate_config.is_file():
        raise ValueError(
            f"No Accelerate config for {gpu_count} GPUs at {accelerate_config}"
        )
    profile = load_yaml(accelerate_config)
    if profile.get("num_processes") != gpu_count:
        raise ValueError(
            f"{accelerate_config} sets num_processes={profile.get('num_processes')!r}; "
            f"expected {gpu_count}"
        )
    distributed_type = str(profile.get("distributed_type", "")).upper().strip("'\"")
    if gpu_count > 1 and distributed_type != "MULTI_GPU":
        raise ValueError(
            f"{accelerate_config} must set distributed_type: MULTI_GPU for {gpu_count} GPUs"
        )
    return accelerate_config


def benchmark_config(
    config: dict, output_dir: Path, per_device_batch_size: int, accumulation_steps: int,
    effective_batch_size: int,
) -> dict:
    """Prepare an output-free run with benchmark-controlled batch math."""
    prepared = copy.deepcopy(config)
    if not prepared.get("training", {}).get("run_sft"):
        raise ValueError("Training benchmarks require training.run_sft: true")
    prepared["training"].update(
        {
            "run_dpo": False,
            "batch_size": per_device_batch_size,
            "effective_batch_size": effective_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "evaluation_strategy": "no",
            "save_strategy": "no",
            "load_best_model_at_end": False,
            "resume_from_checkpoint": None,
            "report_to": "none",
        }
    )
    prepared["evaluation"]["run_after_training"] = False
    prepared["model"]["output_dir"] = str(output_dir / "trainer")
    return prepared


def run_worker(
    config_path: Path,
    settings: dict,
    gpu_count: int,
    result_path: Path,
    accelerate_config: Path,
) -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import torch
    import torch.distributed as dist
    from transformers import Trainer, set_seed

    from train import (
        make_sft_data_collator,
        make_training_arguments,
        prepare_sft_dataset,
        setup_model_and_processor,
        setup_processor,
    )

    rank = int(os.environ.get("RANK", "0"))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [rank={rank}] %(levelname)s: %(message)s",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a training benchmark")

    raw_config = load_yaml(config_path)
    run_dir = result_path.parent
    accumulation_steps = gradient_accumulation_steps(settings, gpu_count)
    config = benchmark_config(
        raw_config,
        run_dir,
        settings["per_device_batch_size"],
        accumulation_steps,
        settings["effective_batch_size"],
    )
    set_seed(config["training"]["seed"])
    processor = setup_processor(config)
    data = prepare_sft_dataset(
        config["data"]["train_sft_dataset_path"],
        processor,
        config,
        "benchmark train",
        settings["max_examples"],
        packed=config["training"]["packing"],
    )
    validation = None
    validation_path = config["data"].get("validation_sft_dataset_path")
    if settings["run_evaluation"] and validation_path:
        validation = prepare_sft_dataset(
            validation_path, processor, config, "benchmark validation",
            settings["max_examples"], packed=False,
        )
    model, processor = setup_model_and_processor(config, processor=processor)

    class CountingTrainer(Trainer):
        """Count exactly the tensors consumed by forward passes without CPU syncs."""

        token_counts = None

        def compute_loss(self, model, inputs, *compute_args, **compute_kwargs):
            attention_mask = inputs.get("attention_mask")
            position_ids = inputs.get("position_ids")
            if model.training and (attention_mask is not None or position_ids is not None):
                if self.token_counts is None:
                    device = attention_mask.device if attention_mask is not None else position_ids.device
                    self.token_counts = torch.zeros(3, dtype=torch.float64, device=device)
                if attention_mask is not None:
                    self.token_counts[0] += attention_mask.shape[0]
                    self.token_counts[1] += attention_mask.numel()
                    self.token_counts[2] += attention_mask.sum(dtype=torch.float64)
                else:
                    # Packed padding-free batches reset position_ids at every
                    # original example boundary and contain no padding.
                    self.token_counts[0] += (position_ids == 0).sum(dtype=torch.float64)
                    self.token_counts[1] += position_ids.numel()
                    self.token_counts[2] += position_ids.numel()
            return super().compute_loss(model, inputs, *compute_args, **compute_kwargs)

    training_args = make_training_arguments(
        config,
        run_dir / "trainer",
        config["training"]["learning_rate"],
        config["training"]["epochs"],
        False,
        max_steps=settings["max_steps"],
        group_by_length=config["training"]["group_by_length"],
    )
    trainer = CountingTrainer(
        model=model,
        args=training_args,
        train_dataset=data,
        eval_dataset=validation,
        data_collator=make_sft_data_collator(processor, config),
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    train_output = trainer.train()
    torch.cuda.synchronize()
    local_elapsed = time.perf_counter() - started
    local_peak_allocated = torch.cuda.max_memory_allocated()
    local_peak_reserved = torch.cuda.max_memory_reserved()

    counts = trainer.token_counts
    if counts is None:
        counts = torch.zeros(3, dtype=torch.float64, device=model.device)
    elapsed = torch.tensor(local_elapsed, dtype=torch.float64, device=counts.device)
    peak_allocated = torch.tensor(local_peak_allocated, dtype=torch.float64, device=counts.device)
    peak_reserved = torch.tensor(local_peak_reserved, dtype=torch.float64, device=counts.device)
    if dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(peak_allocated, op=dist.ReduceOp.MAX)
        dist.all_reduce(peak_reserved, op=dist.ReduceOp.MAX)

    eval_metrics = trainer.evaluate() if validation is not None else {}
    if rank != 0:
        return

    sample_count, padded_tokens, non_padding_tokens = (float(value) for value in counts.cpu())
    elapsed_seconds = elapsed.item()
    result = {
        "gpu_count": gpu_count,
        "accelerate_config": str(accelerate_config),
        "accelerate_config_sha256": hashlib.sha256(accelerate_config.read_bytes()).hexdigest(),
        "gpu_name": torch.cuda.get_device_name(),
        "elapsed_seconds": elapsed_seconds,
        "optimizer_steps": trainer.state.global_step,
        "samples": int(sample_count),
        "padded_tokens": int(padded_tokens),
        "non_padding_tokens": int(non_padding_tokens),
        "samples_per_second": sample_count / elapsed_seconds,
        "padded_tokens_per_second": padded_tokens / elapsed_seconds,
        "non_padding_tokens_per_second": non_padding_tokens / elapsed_seconds,
        "padding_efficiency": non_padding_tokens / padded_tokens if padded_tokens else 0.0,
        "peak_memory_allocated_gib": peak_allocated.item() / 1024**3,
        "peak_memory_reserved_gib": peak_reserved.item() / 1024**3,
        "train_loss": train_output.metrics.get("train_loss"),
        "validation_loss": eval_metrics.get("eval_loss"),
        "per_device_batch_size": config["training"]["batch_size"],
        "gradient_accumulation_steps": accumulation_steps,
        "effective_global_batch_size": settings["effective_batch_size"],
        "max_examples": settings["max_examples"],
        "max_steps": settings["max_steps"],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def worker_command(
    config_path: Path,
    settings: dict,
    gpu_count: int,
    result_path: Path,
    accelerate_config: Path,
) -> list[str]:
    command = [
        "accelerate",
        "launch",
        "--config_file",
        str(accelerate_config),
        str(Path(__file__).resolve()),
        "--worker",
        "--config",
        str(config_path),
        "--gpu-count",
        str(gpu_count),
        "--max-examples",
        str(settings["max_examples"]),
        "--max-steps",
        str(settings["max_steps"]),
        "--per-device-batch-size",
        str(settings["per_device_batch_size"]),
        "--effective-batch-size",
        str(settings["effective_batch_size"]),
        "--result-path",
        str(result_path),
        "--accelerate-config",
        str(accelerate_config),
    ]
    command.append("--run-evaluation" if settings["run_evaluation"] else "--no-run-evaluation")
    return command


def add_scaling_metrics(results: list[dict]) -> None:
    baseline = min(results, key=lambda result: result["gpu_count"])
    baseline_rate = baseline["non_padding_tokens_per_second"]
    baseline_gpus = baseline["gpu_count"]
    for result in results:
        speedup = result["non_padding_tokens_per_second"] / baseline_rate
        ideal_speedup = result["gpu_count"] / baseline_gpus
        result["speedup_vs_baseline"] = speedup
        result["scaling_efficiency"] = speedup / ideal_speedup


def write_csv_report(path: Path, results: list[dict]) -> None:
    fields = list(results[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def print_report(results: list[dict]) -> None:
    header = f"{'GPUs':>4} {'accum':>6} {'global batch':>12} {'samples/s':>12} {'tokens/s':>14} {'peak GiB':>10} {'speedup':>9} {'efficiency':>11}"
    print("\n" + header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result['gpu_count']:>4} "
            f"{result['gradient_accumulation_steps']:>6} "
            f"{result['effective_global_batch_size']:>12} "
            f"{result['samples_per_second']:>12.2f} "
            f"{result['non_padding_tokens_per_second']:>14.0f} "
            f"{result['peak_memory_allocated_gib']:>10.2f} "
            f"{result['speedup_vs_baseline']:>9.2f}x "
            f"{result['scaling_efficiency']:>10.1%}"
        )


def run_parent(config_path: Path, config: dict, settings: dict, dry_run: bool) -> None:
    output_dir = Path(settings["output_dir"]).expanduser().resolve()
    env = os.environ.copy()
    devices = settings.get("devices")
    if devices:
        if isinstance(devices, list):
            devices = ",".join(str(device) for device in devices)
        env["CUDA_VISIBLE_DEVICES"] = str(devices)

    if not dry_run:
        if devices:
            available = len([device for device in str(devices).split(",") if device.strip()])
        else:
            import torch

            available = torch.cuda.device_count()
        requested = max(settings["gpu_counts"])
        if available < requested:
            raise RuntimeError(f"Requested up to {requested} GPUs, but only {available} are visible")
        output_dir.mkdir(parents=True, exist_ok=True)

    # Prime model-forward/Liger/FA3 kernels in a discarded run. Without this,
    # the first (normally 1-GPU) measurement pays Triton JIT compilation while
    # later entries reuse its disk cache and scaling is overstated.
    warmup_settings = copy.deepcopy(settings)
    warmup_settings["max_steps"] = settings["warmup_steps"]
    warmup_settings["run_evaluation"] = False
    warmup_config = resolve_accelerate_config(config_path, settings, 1)
    warmup_result = output_dir / "warmup" / "result.json"
    warmup_command = worker_command(
        config_path, warmup_settings, 1, warmup_result, warmup_config
    )
    print(f"# Warm-up: 1 GPU, {settings['warmup_steps']} discarded optimizer steps", flush=True)
    print("$ " + " ".join(warmup_command), flush=True)
    if not dry_run:
        subprocess.run(warmup_command, cwd=PROJECT_ROOT, env=env, check=True)

    results = []
    for gpu_count in settings["gpu_counts"]:
        accumulation_steps = gradient_accumulation_steps(settings, gpu_count)
        accelerate_config = resolve_accelerate_config(config_path, settings, gpu_count)
        result_path = output_dir / f"{gpu_count}_gpu" / "result.json"
        command = worker_command(
            config_path, settings, gpu_count, result_path, accelerate_config
        )
        print(
            f"# {gpu_count} GPU(s): per_device={settings['per_device_batch_size']} "
            f"accumulation={accumulation_steps} "
            f"effective={settings['effective_batch_size']}",
            flush=True,
        )
        print("$ " + " ".join(command), flush=True)
        if dry_run:
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
        results.append(json.loads(result_path.read_text(encoding="utf-8")))

    if dry_run:
        return
    add_scaling_metrics(results)
    raw_config = config_path.read_bytes()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "benchmark": settings,
        "training_config": {
            "model": config.get("model"),
            "lora": config.get("lora"),
            "training": config.get("training"),
        },
        "results": results,
    }
    report_path = output_dir / settings["report_filename"]
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = report_path.with_suffix(".csv")
    write_csv_report(csv_path, results)
    print_report(results)
    print(f"\nJSON report: {report_path}\nCSV report:  {csv_path}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    settings = resolve_benchmark(config, args)
    if args.worker:
        if args.gpu_count is None or args.result_path is None or args.accelerate_config is None:
            raise ValueError(
                "Internal worker mode requires --gpu-count, --result-path, and "
                "--accelerate-config"
            )
        run_worker(
            config_path,
            settings,
            args.gpu_count,
            Path(args.result_path).resolve(),
            Path(args.accelerate_config).resolve(),
        )
    else:
        run_parent(config_path, config, settings, args.dry_run)


if __name__ == "__main__":
    main()
