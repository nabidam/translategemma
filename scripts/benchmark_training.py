#!/usr/bin/env python3
"""Benchmark SFT throughput across model, GPU, batch, and training-option matrices.

The parent process expands the configured benchmark matrix and launches one
isolated Accelerate job per entry. Model profiles provide size-specific model
and batch settings. Benchmark types independently measure GPU scaling,
micro-batch size at a constant effective global batch, and the effects of
gradient checkpointing and sequence packing.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_TYPES = ("gpu_count", "batch_size", "training_options")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PACKAGE_NAMES = (
    "torch",
    "transformers",
    "accelerate",
    "deepspeed",
    "peft",
    "trl",
    "datasets",
    "flash-attn-3",
    "liger-kernel",
    "bitsandbytes",
)
NVIDIA_TELEMETRY_FIELDS = (
    "index",
    "uuid",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
)
NUMERIC_TELEMETRY_FIELDS = NVIDIA_TELEMETRY_FIELDS[3:]

# H200F size profiles from docs/refs/multi_gpu_h200f.md. User-provided
# benchmark.model_profiles entries are recursively merged over these defaults.
PREDEFINED_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "4b": {
        "model": {"base_model_id": "google/translategemma-4b-it"},
        "training": {"max_length": 2048},
        "per_device_batch_size": 12,
        "effective_batch_size": 96,
        "batch_sizes": [3, 6, 12],
    },
    "12b": {
        "model": {"base_model_id": "google/translategemma-12b-it"},
        "training": {"max_length": 2048},
        "per_device_batch_size": 6,
        "effective_batch_size": 48,
        "batch_sizes": [2, 3, 6],
    },
    "27b": {
        "model": {"base_model_id": "google/translategemma-27b-it"},
        "training": {"max_length": 2048},
        "per_device_batch_size": 2,
        "effective_batch_size": 16,
        "batch_sizes": [1, 2],
    },
}

DEFAULT_TRAINING_OPTION_PROFILES = [
    {"name": "checkpointing_packing", "gradient_checkpointing": True, "packing": True},
    {"name": "checkpointing_only", "gradient_checkpointing": True, "packing": False},
    {"name": "packing_only", "gradient_checkpointing": False, "packing": True},
    {"name": "neither", "gradient_checkpointing": False, "packing": False},
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config.yaml", help="Training configuration file."
    )
    parser.add_argument(
        "--benchmark-types",
        nargs="+",
        choices=BENCHMARK_TYPES,
        help="Benchmark dimensions to run; overrides benchmark.types.",
    )
    parser.add_argument(
        "--model-sizes",
        nargs="+",
        help="Model profile names to benchmark; overrides benchmark.model_sizes.",
    )
    parser.add_argument(
        "--gpu-counts",
        nargs="+",
        type=positive_int,
        help="GPU counts for the gpu_count benchmark; overrides benchmark.gpu_counts.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=positive_int,
        help="Per-device sizes for batch_size benchmarks; overrides every model profile.",
    )
    parser.add_argument(
        "--batch-gpu-count",
        type=positive_int,
        help="Fixed GPU count for batch_size benchmarks.",
    )
    parser.add_argument(
        "--training-options-gpu-count",
        type=positive_int,
        help="Fixed GPU count for training_options benchmarks.",
    )
    parser.add_argument(
        "--max-examples", type=positive_int, help="Override benchmark.max_examples."
    )
    parser.add_argument(
        "--max-steps", type=positive_int, help="Override benchmark.max_steps."
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=positive_int,
        help="Override the baseline per-device size in every selected model profile.",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=positive_int,
        help="Override the effective global batch in every selected model profile.",
    )
    parser.add_argument("--output-dir", help="Override benchmark.output_dir.")
    parser.add_argument(
        "--accelerate-config-pattern",
        help=("Accelerate profile path pattern; use {gpu_count} as the placeholder."),
    )
    parser.add_argument(
        "--devices",
        help="Comma-separated physical GPU IDs exposed to every run, for example 0,1,2,3.",
    )
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument(
        "--run-evaluation",
        dest="run_evaluation",
        action="store_true",
        help="Measure validation loss after each timed training run.",
    )
    evaluation.add_argument(
        "--no-run-evaluation",
        dest="run_evaluation",
        action="store_false",
        help="Skip the post-training validation pass.",
    )
    parser.set_defaults(run_evaluation=None)
    telemetry = parser.add_mutually_exclusive_group()
    telemetry.add_argument(
        "--collect-gpu-telemetry",
        dest="collect_gpu_telemetry",
        action="store_true",
        help="Sample device VRAM, utilization, power, clocks, and temperature.",
    )
    telemetry.add_argument(
        "--no-collect-gpu-telemetry",
        dest="collect_gpu_telemetry",
        action="store_false",
        help="Disable nvidia-smi telemetry sampling.",
    )
    parser.set_defaults(collect_gpu_telemetry=None)
    parser.add_argument(
        "--telemetry-interval-seconds",
        type=positive_float,
        help="Seconds between nvidia-smi samples.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and validate the complete matrix without loading a model.",
    )

    # Internal Accelerate worker arguments.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-path", help=argparse.SUPPRESS)
    parser.add_argument("--accelerate-config", help=argparse.SUPPRESS)
    parser.add_argument("--run-spec", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Return a recursive mapping merge without mutating either input."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def require_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def require_positive_int_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    for item in value:
        require_positive_int(item, label)
    return list(dict.fromkeys(value))


def validate_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SAFE_NAME.fullmatch(value):
        raise ValueError(f"{label} must use only letters, numbers, '.', '_' or '-'")
    return value


def normalize_training_option_profiles(raw_profiles: Any) -> list[dict]:
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("benchmark.training_option_profiles must be a non-empty list")
    profiles = []
    names = set()
    for index, raw_profile in enumerate(raw_profiles):
        label = f"benchmark.training_option_profiles[{index}]"
        if not isinstance(raw_profile, dict):
            raise ValueError(f"{label} must be a mapping")
        profile = copy.deepcopy(raw_profile)
        name = validate_name(profile.get("name"), f"{label}.name")
        if name in names:
            raise ValueError(f"Duplicate training option profile: {name}")
        names.add(name)
        for option in ("gradient_checkpointing", "packing"):
            if not isinstance(profile.get(option), bool):
                raise ValueError(f"{label}.{option} must be true or false")
        profiles.append(profile)
    return profiles


def resolve_benchmark(config: dict, args: argparse.Namespace) -> dict:
    benchmark = copy.deepcopy(config.get("benchmark") or {})
    overrides = {
        "types": args.benchmark_types,
        "model_sizes": args.model_sizes,
        "gpu_counts": args.gpu_counts,
        "batch_sizes": args.batch_sizes,
        "batch_gpu_count": args.batch_gpu_count,
        "training_options_gpu_count": args.training_options_gpu_count,
        "max_examples": args.max_examples,
        "max_steps": args.max_steps,
        "output_dir": args.output_dir,
        "accelerate_config_pattern": args.accelerate_config_pattern,
        "run_evaluation": args.run_evaluation,
        "collect_gpu_telemetry": getattr(args, "collect_gpu_telemetry", None),
        "telemetry_interval_seconds": getattr(args, "telemetry_interval_seconds", None),
        "devices": args.devices,
    }
    for key, value in overrides.items():
        if value is not None:
            benchmark[key] = value

    benchmark.setdefault("types", list(BENCHMARK_TYPES))
    benchmark.setdefault("model_sizes", ["12b"])
    benchmark.setdefault("gpu_counts", [1])
    benchmark.setdefault("batch_gpu_count", 1)
    benchmark.setdefault("training_options_gpu_count", 1)
    benchmark.setdefault("max_examples", 20_000)
    benchmark.setdefault("max_steps", 200)
    benchmark.setdefault("warmup_steps", 5)
    benchmark.setdefault("run_evaluation", True)
    benchmark.setdefault("collect_gpu_telemetry", True)
    benchmark.setdefault("telemetry_interval_seconds", 1.0)
    benchmark.setdefault("fail_fast", False)
    benchmark.setdefault("output_dir", "logs/training_benchmark")
    benchmark.setdefault("report_filename", "benchmark_results.json")
    benchmark.setdefault(
        "accelerate_config_pattern", "accelerate_configs/h200_{gpu_count}gpu.yaml"
    )
    benchmark.setdefault("devices", None)
    benchmark.setdefault("model_profiles", {})
    benchmark.setdefault("training_option_profiles", DEFAULT_TRAINING_OPTION_PROFILES)

    types = benchmark["types"]
    if not isinstance(types, list) or not types:
        raise ValueError("benchmark.types must be a non-empty list")
    unknown_types = set(types) - set(BENCHMARK_TYPES)
    if unknown_types:
        raise ValueError(f"Unknown benchmark.types: {', '.join(sorted(unknown_types))}")
    benchmark["types"] = list(dict.fromkeys(types))

    configured_profiles = benchmark["model_profiles"]
    if not isinstance(configured_profiles, dict):
        raise ValueError("benchmark.model_profiles must be a mapping")
    profiles = copy.deepcopy(PREDEFINED_MODEL_PROFILES)
    for name, overrides_for_model in configured_profiles.items():
        validate_name(name, "benchmark.model_profiles key")
        if not isinstance(overrides_for_model, dict):
            raise ValueError(f"benchmark.model_profiles.{name} must be a mapping")
        profiles[name] = deep_merge(profiles.get(name, {}), overrides_for_model)

    model_sizes = benchmark["model_sizes"]
    if not isinstance(model_sizes, list) or not model_sizes:
        raise ValueError("benchmark.model_sizes must be a non-empty list")
    benchmark["model_sizes"] = list(
        dict.fromkeys(
            validate_name(name, "benchmark.model_sizes entry") for name in model_sizes
        )
    )
    missing = [name for name in benchmark["model_sizes"] if name not in profiles]
    if missing:
        raise ValueError(
            f"No benchmark.model_profiles defined for: {', '.join(missing)}"
        )

    global_batch_sizes = benchmark.get("batch_sizes")
    if global_batch_sizes is not None:
        global_batch_sizes = require_positive_int_list(
            global_batch_sizes, "benchmark.batch_sizes"
        )

    selected_profiles = {}
    for name in benchmark["model_sizes"]:
        profile = profiles[name]
        if not isinstance(profile.get("model"), dict) or not profile["model"].get(
            "base_model_id"
        ):
            raise ValueError(
                f"benchmark.model_profiles.{name}.model.base_model_id is required"
            )
        if args.per_device_batch_size is not None:
            profile["per_device_batch_size"] = args.per_device_batch_size
        if args.effective_batch_size is not None:
            profile["effective_batch_size"] = args.effective_batch_size
        require_positive_int(
            profile.get("per_device_batch_size"),
            f"benchmark.model_profiles.{name}.per_device_batch_size",
        )
        require_positive_int(
            profile.get("effective_batch_size"),
            f"benchmark.model_profiles.{name}.effective_batch_size",
        )
        profile["batch_sizes"] = require_positive_int_list(
            global_batch_sizes
            if global_batch_sizes is not None
            else profile.get("batch_sizes"),
            f"benchmark.model_profiles.{name}.batch_sizes",
        )
        selected_profiles[name] = profile
    benchmark["model_profiles"] = selected_profiles

    benchmark["gpu_counts"] = require_positive_int_list(
        benchmark["gpu_counts"], "benchmark.gpu_counts"
    )
    for key in (
        "batch_gpu_count",
        "training_options_gpu_count",
        "max_examples",
        "max_steps",
        "warmup_steps",
    ):
        require_positive_int(benchmark[key], f"benchmark.{key}")
    if not isinstance(benchmark["run_evaluation"], bool):
        raise ValueError("benchmark.run_evaluation must be true or false")
    if not isinstance(benchmark["collect_gpu_telemetry"], bool):
        raise ValueError("benchmark.collect_gpu_telemetry must be true or false")
    if not isinstance(benchmark["fail_fast"], bool):
        raise ValueError("benchmark.fail_fast must be true or false")
    interval = benchmark["telemetry_interval_seconds"]
    if (
        not isinstance(interval, (int, float))
        or isinstance(interval, bool)
        or interval <= 0
    ):
        raise ValueError(
            "benchmark.telemetry_interval_seconds must be greater than zero"
        )
    if not benchmark["output_dir"] or not benchmark["report_filename"]:
        raise ValueError(
            "benchmark.output_dir and benchmark.report_filename must be configured"
        )
    pattern = benchmark["accelerate_config_pattern"]
    if not isinstance(pattern, str) or "{gpu_count}" not in pattern:
        raise ValueError("benchmark.accelerate_config_pattern must contain {gpu_count}")
    benchmark["training_option_profiles"] = normalize_training_option_profiles(
        benchmark["training_option_profiles"]
    )
    return benchmark


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def installed_package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def host_memory_gib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", meminfo.read_text(), re.MULTILINE)
    return int(match.group(1)) / 1024**2 if match else None


def cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return platform.processor() or None
    match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.MULTILINE)
    return match.group(1).strip() if match else platform.processor() or None


def cpu_topology() -> dict[str, int | None]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return {"physical_core_count": None, "socket_count": None}
    sockets = set()
    cores = set()
    for block in cpuinfo.read_text().split("\n\n"):
        socket_match = re.search(r"^physical id\s*:\s*(\d+)$", block, re.MULTILINE)
        core_match = re.search(r"^core id\s*:\s*(\d+)$", block, re.MULTILINE)
        if socket_match:
            socket_id = int(socket_match.group(1))
            sockets.add(socket_id)
            if core_match:
                cores.add((socket_id, int(core_match.group(1))))
    return {
        "physical_core_count": len(cores) or None,
        "socket_count": len(sockets) or None,
    }


def workspace_disk() -> dict[str, float | str]:
    stats = os.statvfs(PROJECT_ROOT)
    return {
        "path": str(PROJECT_ROOT),
        "total_gib": stats.f_blocks * stats.f_frsize / 1024**3,
        "available_gib": stats.f_bavail * stats.f_frsize / 1024**3,
    }


def parse_numeric(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def query_nvidia_telemetry(
    gpu_ids: list[str] | None = None,
) -> tuple[list[dict], str | None]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(NVIDIA_TELEMETRY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    if gpu_ids:
        command.insert(1, f"--id={','.join(gpu_ids)}")
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, check=True, timeout=10
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        detail = getattr(error, "stderr", None) or str(error)
        return [], detail.strip()

    rows = []
    for values in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if len(values) != len(NVIDIA_TELEMETRY_FIELDS):
            continue
        row = dict(zip(NVIDIA_TELEMETRY_FIELDS, (value.strip() for value in values)))
        for field in NUMERIC_TELEMETRY_FIELDS:
            row[field] = parse_numeric(row[field])
        rows.append(row)
    return rows, None


def query_host_telemetry(
    previous_cpu: tuple[int, int] | None,
) -> tuple[dict, tuple[int, int] | None]:
    reading: dict[str, float] = {}
    try:
        cpu_values = [
            int(value)
            for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        ]
        idle = cpu_values[3] + (cpu_values[4] if len(cpu_values) > 4 else 0)
        total = sum(cpu_values)
        current_cpu = (idle, total)
        if previous_cpu is not None:
            idle_delta = idle - previous_cpu[0]
            total_delta = total - previous_cpu[1]
            if total_delta > 0:
                reading["cpu_utilization_percent"] = 100 * (
                    1 - idle_delta / total_delta
                )
    except (FileNotFoundError, IndexError, ValueError):
        current_cpu = None

    try:
        memory_values = {
            match.group(1): int(match.group(2))
            for match in re.finditer(
                r"^(MemTotal|MemAvailable):\s+(\d+)\s+kB$",
                Path("/proc/meminfo").read_text(),
                re.MULTILINE,
            )
        }
        total_kib = memory_values["MemTotal"]
        available_kib = memory_values["MemAvailable"]
        reading["memory_used_gib"] = (total_kib - available_kib) / 1024**2
        reading["memory_utilization_percent"] = (
            100 * (total_kib - available_kib) / total_kib
        )
    except (FileNotFoundError, KeyError, ValueError):
        pass

    try:
        load_1m, load_5m, load_15m = os.getloadavg()
        reading.update(
            {
                "load_average_1m": load_1m,
                "load_average_5m": load_5m,
                "load_average_15m": load_15m,
            }
        )
    except OSError:
        pass
    return reading, current_cpu


def selected_physical_gpu_ids(devices: Any, gpu_count: int) -> list[str]:
    if devices:
        values = devices if isinstance(devices, list) else str(devices).split(",")
        return [str(value).strip() for value in values[:gpu_count]]
    return [str(index) for index in range(gpu_count)]


def summarize_gpu_telemetry(samples: list[dict], duration_seconds: float) -> dict:
    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        for gpu in sample["gpus"]:
            grouped.setdefault(gpu["uuid"], []).append(gpu)

    summaries = []
    for uuid, readings in grouped.items():
        first = readings[0]
        summary = {
            "index": first["index"],
            "uuid": uuid,
            "name": first["name"],
            "sample_count": len(readings),
        }
        for field in NUMERIC_TELEMETRY_FIELDS:
            values = [
                reading[field] for reading in readings if reading[field] is not None
            ]
            normalized = field.replace(".", "_")
            if values:
                summary[f"{normalized}_average"] = sum(values) / len(values)
                summary[f"{normalized}_maximum"] = max(values)
                summary[f"{normalized}_minimum"] = min(values)
        average_power = summary.get("power_draw_average")
        if average_power is not None:
            summary["estimated_energy_wh"] = average_power * duration_seconds / 3600
        summaries.append(summary)
    host_summary = {}
    host_fields = sorted(
        {field for sample in samples for field in sample.get("host", {})}
    )
    for field in host_fields:
        values = [
            sample["host"][field]
            for sample in samples
            if sample.get("host", {}).get(field) is not None
        ]
        if values:
            host_summary[f"{field}_average"] = sum(values) / len(values)
            host_summary[f"{field}_maximum"] = max(values)
            host_summary[f"{field}_minimum"] = min(values)

    return {
        "sample_count": len(samples),
        "duration_seconds": duration_seconds,
        "per_gpu": summaries,
        "host": host_summary,
        "estimated_total_energy_wh": sum(
            gpu.get("estimated_energy_wh", 0.0) for gpu in summaries
        ),
    }


def run_with_gpu_telemetry(
    command: list[str],
    env: dict,
    case: dict,
    settings: dict,
    telemetry_path: Path,
    log_path: Path,
) -> tuple[int, dict]:
    started_at = utc_now()
    started = time.perf_counter()
    samples = []
    errors = []
    gpu_ids = selected_physical_gpu_ids(settings.get("devices"), case["gpu_count"])
    interval = float(settings["telemetry_interval_seconds"])
    previous_cpu = None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def stream_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()

        output_thread = threading.Thread(target=stream_output, daemon=True)
        output_thread.start()
        while process.poll() is None:
            if settings["collect_gpu_telemetry"]:
                rows, error = query_nvidia_telemetry(gpu_ids)
                host, previous_cpu = query_host_telemetry(previous_cpu)
                if error and (not errors or errors[-1] != error):
                    errors.append(error)
                if rows or host:
                    samples.append(
                        {
                            "captured_at": utc_now(),
                            "elapsed_seconds": time.perf_counter() - started,
                            "gpus": rows,
                            "host": host,
                        }
                    )
            try:
                process.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                pass
        output_thread.join()

    duration = time.perf_counter() - started
    summary = summarize_gpu_telemetry(samples, duration)
    summary.update(
        {
            "enabled": settings["collect_gpu_telemetry"],
            "started_at": started_at,
            "finished_at": utc_now(),
            "errors": errors,
            "raw_path": str(telemetry_path)
            if settings["collect_gpu_telemetry"]
            else None,
            "log_path": str(log_path),
        }
    )
    if settings["collect_gpu_telemetry"]:
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "case_id": case["id"],
                    "interval_seconds": interval,
                    "gpu_ids": gpu_ids,
                    "summary": summary,
                    "samples": samples,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return process.returncode, summary


def collect_system_inventory(settings: dict) -> dict:
    import torch

    gpu_rows, gpu_error = query_nvidia_telemetry()
    environment_names = (
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_IB_HCA",
        "NCCL_SOCKET_IFNAME",
        "NCCL_P2P_LEVEL",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
    )
    revision = command_output(["git", "rev-parse", "HEAD"])
    dirty = command_output(["git", "status", "--porcelain"])
    topology = cpu_topology()
    return {
        "captured_at": utc_now(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "python": sys.version,
            "cpu_model": cpu_model(),
            "physical_core_count": topology["physical_core_count"],
            "logical_cpu_count": os.cpu_count(),
            "cpu_socket_count": topology["socket_count"],
            "memory_total_gib": host_memory_gib(),
            "workspace_disk": workspace_disk(),
        },
        "gpu": {
            "visible_count": torch.cuda.device_count(),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "driver_version": command_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ),
            "devices": gpu_rows,
            "inventory_error": gpu_error,
            "topology": command_output(["nvidia-smi", "topo", "-m"]),
        },
        "software": installed_package_versions(),
        "repository": {
            "revision": revision,
            "dirty": bool(dirty),
            "dirty_files": dirty.splitlines() if dirty else [],
        },
        "environment": {
            name: os.environ[name] for name in environment_names if name in os.environ
        },
        "telemetry": {
            "enabled": settings["collect_gpu_telemetry"],
            "interval_seconds": settings["telemetry_interval_seconds"],
        },
    }


def dataset_manifest(config_path: Path, config: dict, cases: list[dict]) -> list[dict]:
    paths: dict[str, set[str]] = {}
    benchmark_dataset_keys = (
        "train_sft_dataset_path",
        "validation_sft_dataset_path",
    )
    for case in cases:
        prepared = deep_merge(config, case["config_overrides"])
        data_config = prepared.get("data") or {}
        for key in benchmark_dataset_keys:
            value = data_config.get(key)
            if value:
                paths.setdefault(key, set()).add(str(value))

    manifest = []
    for key, values in sorted(paths.items()):
        for value in sorted(values):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = (config_path.parent / path).resolve()
            entry = {
                "config_key": key,
                "configured_path": value,
                "resolved_path": str(path),
            }
            if path.is_file():
                stat = path.stat()
                entry.update(
                    {
                        "exists": True,
                        "type": "file",
                        "size_bytes": stat.st_size,
                        "modified_at": datetime.fromtimestamp(
                            stat.st_mtime, timezone.utc
                        ).isoformat(),
                        "sha256": sha256_file(path),
                    }
                )
            elif path.is_dir():
                entry.update({"exists": True, "type": "directory", "sha256": None})
            else:
                entry.update({"exists": False, "type": None, "sha256": None})
            manifest.append(entry)
    return manifest


def gradient_accumulation_steps(case: dict) -> int:
    """Derive accumulation while holding the effective global batch constant."""
    denominator = case["per_device_batch_size"] * case["gpu_count"]
    effective_batch_size = case["effective_batch_size"]
    if effective_batch_size % denominator:
        raise ValueError(
            f"{case['id']}: effective batch {effective_batch_size} must be exactly divisible by "
            f"per-device batch {case['per_device_batch_size']} * {case['gpu_count']} GPUs"
        )
    accumulation = effective_batch_size // denominator
    if accumulation < 1:
        raise ValueError(
            f"{case['id']}: effective batch {effective_batch_size} is smaller than "
            f"the {denominator}-sample distributed micro-batch"
        )
    return accumulation


def profile_config_overrides(profile: dict) -> dict:
    return {
        section: copy.deepcopy(profile[section])
        for section in ("model", "data", "lora", "training")
        if section in profile
    }


def build_benchmark_cases(config: dict, settings: dict) -> list[dict]:
    """Expand all selected dimensions into validated, worker-ready run specs."""
    cases = []
    base_training = config.get("training") or {}
    for model_size in settings["model_sizes"]:
        profile = settings["model_profiles"][model_size]
        base_case = {
            "model_size": model_size,
            "per_device_batch_size": profile["per_device_batch_size"],
            "effective_batch_size": profile["effective_batch_size"],
            "gradient_checkpointing": profile.get("training", {}).get(
                "gradient_checkpointing",
                base_training.get("gradient_checkpointing", False),
            ),
            "packing": profile.get("training", {}).get(
                "packing", base_training.get("packing", False)
            ),
            "config_overrides": profile_config_overrides(profile),
            "max_examples": settings["max_examples"],
            "max_steps": settings["max_steps"],
            "run_evaluation": settings["run_evaluation"],
        }
        if "gpu_count" in settings["types"]:
            for gpu_count in settings["gpu_counts"]:
                cases.append(
                    {
                        **copy.deepcopy(base_case),
                        "id": f"gpu_count/{model_size}/{gpu_count}_gpu",
                        "benchmark_type": "gpu_count",
                        "variant": f"{gpu_count}_gpu",
                        "gpu_count": gpu_count,
                    }
                )
        if "batch_size" in settings["types"]:
            for batch_size in profile["batch_sizes"]:
                cases.append(
                    {
                        **copy.deepcopy(base_case),
                        "id": f"batch_size/{model_size}/per_device_{batch_size}",
                        "benchmark_type": "batch_size",
                        "variant": f"per_device_{batch_size}",
                        "gpu_count": settings["batch_gpu_count"],
                        "per_device_batch_size": batch_size,
                    }
                )
        if "training_options" in settings["types"]:
            for option_profile in settings["training_option_profiles"]:
                case = {
                    **copy.deepcopy(base_case),
                    "id": f"training_options/{model_size}/{option_profile['name']}",
                    "benchmark_type": "training_options",
                    "variant": option_profile["name"],
                    "gpu_count": settings["training_options_gpu_count"],
                    "gradient_checkpointing": option_profile["gradient_checkpointing"],
                    "packing": option_profile["packing"],
                }
                case["config_overrides"] = deep_merge(
                    case["config_overrides"],
                    {
                        "training": {
                            "gradient_checkpointing": case["gradient_checkpointing"],
                            "packing": case["packing"],
                        }
                    },
                )
                cases.append(case)

    for case in cases:
        case["gradient_accumulation_steps"] = gradient_accumulation_steps(case)
    return cases


def resolve_accelerate_config(
    config_path: Path, settings: dict, gpu_count: int
) -> Path:
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
            f"{accelerate_config} sets num_processes={profile.get('num_processes')!r}; expected {gpu_count}"
        )
    distributed_type = str(profile.get("distributed_type", "")).upper().strip("'\"")
    if gpu_count > 1 and distributed_type != "MULTI_GPU":
        raise ValueError(f"{accelerate_config} must use MULTI_GPU for {gpu_count} GPUs")
    return accelerate_config


def benchmark_config(config: dict, output_dir: Path, case: dict) -> dict:
    """Apply model/variant overrides and prepare an output-free timed run."""
    prepared = deep_merge(config, case["config_overrides"])
    if not prepared.get("training", {}).get("run_sft"):
        raise ValueError("Training benchmarks require training.run_sft: true")
    prepared["training"].update(
        {
            "run_dpo": False,
            "batch_size": case["per_device_batch_size"],
            "effective_batch_size": case["effective_batch_size"],
            "gradient_accumulation_steps": case["gradient_accumulation_steps"],
            "gradient_checkpointing": case["gradient_checkpointing"],
            "packing": case["packing"],
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
    config_path: Path, case: dict, result_path: Path, accelerate_config: Path
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
    config = benchmark_config(raw_config, run_dir, case)
    set_seed(config["training"]["seed"])
    processor = setup_processor(config)
    data = prepare_sft_dataset(
        config["data"]["train_sft_dataset_path"],
        processor,
        config,
        "benchmark train",
        case["max_examples"],
        packed=config["training"]["packing"],
    )
    validation = None
    validation_path = config["data"].get("validation_sft_dataset_path")
    if case["run_evaluation"] and validation_path:
        validation = prepare_sft_dataset(
            validation_path,
            processor,
            config,
            "benchmark validation",
            case["max_examples"],
            packed=False,
        )
    model, processor = setup_model_and_processor(config, processor=processor)

    class CountingTrainer(Trainer):
        """Count exactly the tensors consumed by forward passes without CPU syncs."""

        token_counts = None

        def compute_loss(self, model, inputs, *compute_args, **compute_kwargs):
            attention_mask = inputs.get("attention_mask")
            position_ids = inputs.get("position_ids")
            if model.training and (
                attention_mask is not None or position_ids is not None
            ):
                if self.token_counts is None:
                    device = (
                        attention_mask.device
                        if attention_mask is not None
                        else position_ids.device
                    )
                    self.token_counts = torch.zeros(
                        3, dtype=torch.float64, device=device
                    )
                if attention_mask is not None:
                    self.token_counts[0] += attention_mask.shape[0]
                    self.token_counts[1] += attention_mask.numel()
                    self.token_counts[2] += attention_mask.sum(dtype=torch.float64)
                else:
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
        max_steps=case["max_steps"],
        group_by_length=config["training"]["group_by_length"],
    )
    trainer = CountingTrainer(
        model=model,
        args=training_args,
        train_dataset=data,
        eval_dataset=validation,
        data_collator=make_sft_data_collator(processor, config),
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    training_started_at = utc_now()
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
    local_device = torch.cuda.current_device()
    device_stats = torch.tensor(
        [
            local_peak_allocated,
            local_peak_reserved,
            torch.cuda.get_device_properties(local_device).total_memory,
        ],
        dtype=torch.float64,
        device=counts.device,
    )
    if dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        gathered_device_stats = [
            torch.zeros_like(device_stats) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_device_stats, device_stats)
    else:
        gathered_device_stats = [device_stats]

    eval_metrics = trainer.evaluate() if validation is not None else {}
    if rank != 0:
        return

    sample_count, padded_tokens, non_padding_tokens = (
        float(value) for value in counts.cpu()
    )
    elapsed_seconds = elapsed.item()
    per_rank_memory = []
    for device_rank, stats in enumerate(gathered_device_stats):
        allocated, reserved, total = (float(value) for value in stats.cpu())
        per_rank_memory.append(
            {
                "rank": device_rank,
                "peak_allocated_gib": allocated / 1024**3,
                "peak_reserved_gib": reserved / 1024**3,
                "device_total_gib": total / 1024**3,
                "peak_allocated_percent": 100 * allocated / total,
                "peak_reserved_percent": 100 * reserved / total,
            }
        )
    peak_allocated_gib = max(item["peak_allocated_gib"] for item in per_rank_memory)
    peak_reserved_gib = max(item["peak_reserved_gib"] for item in per_rank_memory)
    result = {
        "status": "success",
        "id": case["id"],
        "benchmark_type": case["benchmark_type"],
        "model_size": case["model_size"],
        "model_id": config["model"]["base_model_id"],
        "variant": case["variant"],
        "gpu_count": case["gpu_count"],
        "accelerate_config": str(accelerate_config),
        "accelerate_config_sha256": hashlib.sha256(
            accelerate_config.read_bytes()
        ).hexdigest(),
        "gpu_name": torch.cuda.get_device_name(),
        "training_started_at": training_started_at,
        "training_finished_at": utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "optimizer_steps": trainer.state.global_step,
        "seconds_per_optimizer_step": elapsed_seconds / trainer.state.global_step
        if trainer.state.global_step
        else None,
        "total_gpu_seconds": elapsed_seconds * case["gpu_count"],
        "samples": int(sample_count),
        "padded_tokens": int(padded_tokens),
        "non_padding_tokens": int(non_padding_tokens),
        "samples_per_second": sample_count / elapsed_seconds,
        "padded_tokens_per_second": padded_tokens / elapsed_seconds,
        "non_padding_tokens_per_second": non_padding_tokens / elapsed_seconds,
        "non_padding_tokens_per_gpu_second": non_padding_tokens
        / elapsed_seconds
        / case["gpu_count"],
        "padding_efficiency": non_padding_tokens / padded_tokens
        if padded_tokens
        else 0.0,
        "peak_memory_allocated_gib": peak_allocated_gib,
        "peak_memory_reserved_gib": peak_reserved_gib,
        "per_rank_pytorch_memory": per_rank_memory,
        "train_loss": train_output.metrics.get("train_loss"),
        "validation_loss": eval_metrics.get("eval_loss"),
        "trainer_train_metrics": train_output.metrics,
        "trainer_evaluation_metrics": eval_metrics,
        "prepared_train_rows": len(data),
        "prepared_validation_rows": len(validation) if validation is not None else None,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_parameter_percent": 100 * trainable_parameters / total_parameters,
        "per_device_batch_size": case["per_device_batch_size"],
        "gradient_accumulation_steps": case["gradient_accumulation_steps"],
        "effective_global_batch_size": case["effective_batch_size"],
        "gradient_checkpointing": case["gradient_checkpointing"],
        "packing": case["packing"],
        "max_sequence_length": config["training"]["max_length"],
        "attention_implementation": config["model"]["attn_implementation"],
        "dtype": config["model"]["dtype"],
        "use_4bit": config["model"]["use_4bit"],
        "use_liger_kernel": config["training"]["use_liger_kernel"],
        "optimizer": config["training"]["optimizer"],
        "learning_rate": config["training"]["learning_rate"],
        "source_language": config["data"]["source_lang"],
        "target_language": config["data"]["target_lang"],
        "max_examples": case["max_examples"],
        "max_steps": case["max_steps"],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def worker_command(
    config_path: Path, case: dict, result_path: Path, accelerate_config: Path
) -> list[str]:
    return [
        "accelerate",
        "launch",
        "--config_file",
        str(accelerate_config),
        str(Path(__file__).resolve()),
        "--worker",
        "--config",
        str(config_path),
        "--result-path",
        str(result_path),
        "--accelerate-config",
        str(accelerate_config),
        "--run-spec",
        json.dumps(case, separators=(",", ":")),
    ]


def add_comparison_metrics(results: list[dict]) -> None:
    groups: dict[tuple[str, str], list[dict]] = {}
    for result in results:
        if result.get("status") != "success":
            continue
        groups.setdefault((result["benchmark_type"], result["model_size"]), []).append(
            result
        )
    for (benchmark_type, _model_size), group in groups.items():
        baseline = group[0]
        if benchmark_type == "gpu_count":
            baseline = min(group, key=lambda result: result["gpu_count"])
        baseline_rate = baseline["non_padding_tokens_per_second"]
        for result in group:
            relative = result["non_padding_tokens_per_second"] / baseline_rate
            result["throughput_vs_baseline"] = relative
            if benchmark_type == "gpu_count":
                ideal = result["gpu_count"] / baseline["gpu_count"]
                result["scaling_efficiency"] = relative / ideal


def build_benchmark_summary(results: list[dict]) -> dict:
    successful = [result for result in results if result.get("status") == "success"]
    failed = [result for result in results if result.get("status") != "success"]
    groups: dict[tuple[str, str], list[dict]] = {}
    for result in successful:
        groups.setdefault((result["benchmark_type"], result["model_size"]), []).append(
            result
        )

    comparisons = []
    for (benchmark_type, model_size), group in groups.items():
        fastest = max(group, key=lambda item: item["non_padding_tokens_per_second"])
        lowest_vram = min(group, key=lambda item: item["peak_memory_allocated_gib"])
        comparison = {
            "benchmark_type": benchmark_type,
            "model_size": model_size,
            "fastest_variant": fastest["variant"],
            "fastest_tokens_per_second": fastest["non_padding_tokens_per_second"],
            "fastest_elapsed_seconds": fastest["elapsed_seconds"],
            "lowest_vram_variant": lowest_vram["variant"],
            "lowest_peak_pytorch_allocated_gib": lowest_vram[
                "peak_memory_allocated_gib"
            ],
        }
        if benchmark_type == "gpu_count":
            largest = max(group, key=lambda item: item["gpu_count"])
            comparison.update(
                {
                    "largest_gpu_count": largest["gpu_count"],
                    "largest_gpu_count_speedup": largest.get("throughput_vs_baseline"),
                    "largest_gpu_count_scaling_efficiency": largest.get(
                        "scaling_efficiency"
                    ),
                }
            )
        comparisons.append(comparison)

    return {
        "total_runs": len(results),
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "measured_training_seconds": sum(
            result["elapsed_seconds"] for result in successful
        ),
        "total_gpu_seconds": sum(result["total_gpu_seconds"] for result in successful),
        "total_job_wall_seconds": sum(
            result.get("job_wall_seconds", 0.0) for result in results
        ),
        "total_job_gpu_seconds": sum(
            result.get("job_wall_seconds", 0.0) * result.get("gpu_count", 0)
            for result in results
        ),
        "estimated_gpu_energy_wh": sum(
            result.get("gpu_telemetry", {}).get("estimated_total_energy_wh", 0.0)
            for result in results
        ),
        "comparisons": comparisons,
        "failed_case_ids": [result["id"] for result in failed],
    }


def write_csv_report(path: Path, results: list[dict]) -> None:
    fields = list(dict.fromkeys(key for result in results for key in result))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in result.items()
                }
            )


def optional_number(value: Any, precision: int = 1, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}{suffix}"


def telemetry_metric(result: dict, key: str) -> float | None:
    values = [
        gpu[key]
        for gpu in result.get("gpu_telemetry", {}).get("per_gpu", [])
        if gpu.get(key) is not None
    ]
    return max(values) if values else None


def render_rich_report(report: dict, html_path: Path) -> None:
    console = Console(record=True, width=190)
    summary = report["summary"]
    status_style = "green" if not summary["failed_runs"] else "yellow"
    console.print(
        Panel.fit(
            f"[bold]TranslateGemma training benchmark[/bold]\n"
            f"Run ID: {report['run_id']}\n"
            f"Completed: {report['created_at']}\n"
            f"[{status_style}]{summary['successful_runs']} successful, "
            f"{summary['failed_runs']} failed[/{status_style}]",
            border_style="cyan",
        )
    )

    host = report["system"]["host"]
    gpu = report["system"]["gpu"]
    system_table = Table(title="System and software", show_lines=False)
    system_table.add_column("Category", style="cyan")
    system_table.add_column("Property", style="bold")
    system_table.add_column("Value")
    system_rows = (
        ("Host", "Hostname", host.get("hostname")),
        ("Host", "OS / kernel", f"{host.get('platform')} / {host.get('kernel')}"),
        ("Host", "CPU", host.get("cpu_model")),
        (
            "Host",
            "Sockets / physical / logical CPUs",
            f"{host.get('cpu_socket_count')} / {host.get('physical_core_count')} / {host.get('logical_cpu_count')}",
        ),
        ("Host", "RAM", optional_number(host.get("memory_total_gib"), 1, " GiB")),
        (
            "Host",
            "Workspace disk total / available",
            f"{optional_number(host.get('workspace_disk', {}).get('total_gib'), 1, ' GiB')} / "
            f"{optional_number(host.get('workspace_disk', {}).get('available_gib'), 1, ' GiB')}",
        ),
        ("GPU", "Visible devices", gpu.get("visible_count")),
        (
            "GPU",
            "Driver / CUDA / cuDNN",
            f"{gpu.get('driver_version')} / {gpu.get('cuda_runtime')} / {gpu.get('cudnn_version')}",
        ),
        (
            "Repository",
            "Revision / dirty",
            f"{report['system']['repository'].get('revision')} / {report['system']['repository'].get('dirty')}",
        ),
    )
    for category, property_name, value in system_rows:
        system_table.add_row(str(category), str(property_name), str(value))
    package_text = ", ".join(
        f"{name}={version}"
        for name, version in report["system"]["software"].items()
        if version
    )
    system_table.add_row("Software", "Packages", package_text)
    console.print(system_table)

    gpu_table = Table(title="GPU inventory")
    gpu_table.add_column("Index", justify="right")
    gpu_table.add_column("UUID")
    gpu_table.add_column("Name")
    gpu_table.add_column("VRAM", justify="right")
    gpu_table.add_column("Power limit", justify="right")
    for device in gpu.get("devices", []):
        gpu_table.add_row(
            str(device.get("index")),
            str(device.get("uuid")),
            str(device.get("name")),
            optional_number(device.get("memory.total"), 0, " MiB"),
            optional_number(device.get("power.limit"), 0, " W"),
        )
    if gpu.get("devices"):
        console.print(gpu_table)

    dataset_table = Table(title="Dataset artifacts")
    dataset_table.add_column("Config key")
    dataset_table.add_column("Path")
    dataset_table.add_column("Exists", justify="center")
    dataset_table.add_column("Size", justify="right")
    dataset_table.add_column("SHA-256")
    for artifact in report["datasets"]:
        dataset_table.add_row(
            artifact["config_key"],
            artifact["configured_path"],
            "yes" if artifact["exists"] else "no",
            optional_number(
                artifact.get("size_bytes", 0) / 1024**2
                if artifact.get("size_bytes") is not None
                else None,
                1,
                " MiB",
            ),
            artifact.get("sha256") or "-",
        )
    if report["datasets"]:
        console.print(dataset_table)

    results_table = Table(title="Measured runs", show_lines=True)
    columns = (
        ("Status", "center"),
        ("Type", "left"),
        ("Model", "left"),
        ("Variant", "left"),
        ("GPUs", "right"),
        ("Batch math", "right"),
        ("Time", "right"),
        ("s/step", "right"),
        ("tokens/s", "right"),
        ("Peak VRAM", "right"),
        ("Relative", "right"),
        ("Scale eff.", "right"),
        ("Train loss", "right"),
        ("Val loss", "right"),
    )
    for title, justify in columns:
        results_table.add_column(title, justify=justify, no_wrap=True)
    for result in report["results"]:
        if result.get("status") != "success":
            results_table.add_row(
                "[red]FAIL[/red]",
                result["benchmark_type"],
                result["model_size"],
                result["variant"],
                str(result["gpu_count"]),
                *(["-"] * 9),
            )
            continue
        results_table.add_row(
            "[green]OK[/green]",
            result["benchmark_type"],
            result["model_size"],
            result["variant"],
            str(result["gpu_count"]),
            f"{result['per_device_batch_size']}x{result['gradient_accumulation_steps']}x{result['gpu_count']}={result['effective_global_batch_size']}",
            optional_number(result["elapsed_seconds"], 1, "s"),
            optional_number(result["seconds_per_optimizer_step"], 3),
            optional_number(result["non_padding_tokens_per_second"], 0),
            optional_number(result["peak_memory_allocated_gib"], 1, " GiB"),
            optional_number(result.get("throughput_vs_baseline"), 2, "x"),
            optional_number(result.get("scaling_efficiency"), 1, "%")
            if result.get("scaling_efficiency") is None
            else optional_number(100 * result["scaling_efficiency"], 1, "%"),
            optional_number(result.get("train_loss"), 4),
            optional_number(result.get("validation_loss"), 4),
        )
    console.print(results_table)

    telemetry_table = Table(title="Device telemetry", show_lines=True)
    telemetry_columns = (
        ("Case", "left"),
        ("Samples", "right"),
        ("Device VRAM peak", "right"),
        ("GPU util avg / max", "right"),
        ("Memory util avg", "right"),
        ("Power avg / max", "right"),
        ("Temp max", "right"),
        ("Energy", "right"),
    )
    for title, justify in telemetry_columns:
        telemetry_table.add_column(title, justify=justify, no_wrap=True)
    for result in report["results"]:
        telemetry = result.get("gpu_telemetry", {})
        telemetry_table.add_row(
            result["id"],
            str(telemetry.get("sample_count", 0)),
            optional_number(telemetry_metric(result, "memory_used_maximum"), 0, " MiB"),
            f"{optional_number(telemetry_metric(result, 'utilization_gpu_average'), 1, '%')} / "
            f"{optional_number(telemetry_metric(result, 'utilization_gpu_maximum'), 1, '%')}",
            optional_number(
                telemetry_metric(result, "utilization_memory_average"), 1, "%"
            ),
            f"{optional_number(telemetry_metric(result, 'power_draw_average'), 1, ' W')} / "
            f"{optional_number(telemetry_metric(result, 'power_draw_maximum'), 1, ' W')}",
            optional_number(
                telemetry_metric(result, "temperature_gpu_maximum"), 0, " C"
            ),
            optional_number(telemetry.get("estimated_total_energy_wh"), 2, " Wh"),
        )
    console.print(telemetry_table)

    host_telemetry_table = Table(title="Host telemetry", show_lines=True)
    host_telemetry_table.add_column("Case")
    host_telemetry_table.add_column("CPU avg / max", justify="right")
    host_telemetry_table.add_column("RAM avg / max", justify="right")
    host_telemetry_table.add_column("RAM used max", justify="right")
    host_telemetry_table.add_column("Load 1m avg / max", justify="right")
    for result in report["results"]:
        host_metrics = result.get("gpu_telemetry", {}).get("host", {})
        host_telemetry_table.add_row(
            result["id"],
            f"{optional_number(host_metrics.get('cpu_utilization_percent_average'), 1, '%')} / "
            f"{optional_number(host_metrics.get('cpu_utilization_percent_maximum'), 1, '%')}",
            f"{optional_number(host_metrics.get('memory_utilization_percent_average'), 1, '%')} / "
            f"{optional_number(host_metrics.get('memory_utilization_percent_maximum'), 1, '%')}",
            optional_number(host_metrics.get("memory_used_gib_maximum"), 1, " GiB"),
            f"{optional_number(host_metrics.get('load_average_1m_average'), 2)} / "
            f"{optional_number(host_metrics.get('load_average_1m_maximum'), 2)}",
        )
    console.print(host_telemetry_table)

    comparison_table = Table(title="Comparison summary")
    comparison_table.add_column("Type")
    comparison_table.add_column("Model")
    comparison_table.add_column("Fastest variant")
    comparison_table.add_column("tokens/s", justify="right")
    comparison_table.add_column("Lowest PyTorch VRAM")
    comparison_table.add_column("Peak GiB", justify="right")
    comparison_table.add_column("Largest-GPU speedup / efficiency", justify="right")
    for item in summary["comparisons"]:
        scaling = "-"
        if item.get("largest_gpu_count_speedup") is not None:
            scaling = (
                f"{item['largest_gpu_count_speedup']:.2f}x / "
                f"{100 * item['largest_gpu_count_scaling_efficiency']:.1f}%"
            )
        comparison_table.add_row(
            item["benchmark_type"],
            item["model_size"],
            item["fastest_variant"],
            f"{item['fastest_tokens_per_second']:.0f}",
            item["lowest_vram_variant"],
            f"{item['lowest_peak_pytorch_allocated_gib']:.1f}",
            scaling,
        )
    console.print(comparison_table)

    totals = Table(title="Run totals")
    totals.add_column("Measured training", justify="right")
    totals.add_column("Training GPU time", justify="right")
    totals.add_column("Complete job time", justify="right")
    totals.add_column("Complete job GPU time", justify="right")
    totals.add_column("Estimated GPU energy", justify="right")
    totals.add_row(
        optional_number(summary["measured_training_seconds"] / 3600, 2, " h"),
        optional_number(summary["total_gpu_seconds"] / 3600, 2, " GPU-h"),
        optional_number(summary["total_job_wall_seconds"] / 3600, 2, " h"),
        optional_number(summary["total_job_gpu_seconds"] / 3600, 2, " GPU-h"),
        optional_number(summary["estimated_gpu_energy_wh"] / 1000, 3, " kWh"),
    )
    console.print(totals)
    console.save_html(str(html_path), clear=False)


def write_markdown_summary(path: Path, report: dict) -> None:
    summary = report["summary"]
    lines = [
        "# TranslateGemma training benchmark summary",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Created: `{report['created_at']}`",
        f"- Status: {summary['successful_runs']} successful, {summary['failed_runs']} failed",
        f"- Measured training time: {summary['measured_training_seconds'] / 3600:.2f} hours",
        f"- Training-loop GPU time: {summary['total_gpu_seconds'] / 3600:.2f} GPU-hours",
        f"- Complete-job time: {summary['total_job_wall_seconds'] / 3600:.2f} hours",
        f"- Complete-job GPU time: {summary['total_job_gpu_seconds'] / 3600:.2f} GPU-hours",
        f"- Estimated GPU energy: {summary['estimated_gpu_energy_wh'] / 1000:.3f} kWh",
        "",
        "## Results",
        "",
        "| Status | Type | Model | Variant | GPUs | Batch | Accum | Global | Seconds | tokens/s | PyTorch peak GiB | Device peak MiB | GPU util avg | Relative | Scaling efficiency |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        if result.get("status") != "success":
            lines.append(
                f"| failed | {result['benchmark_type']} | {result['model_size']} | "
                f"{result['variant']} | {result['gpu_count']} | - | - | - | - | - | - | - | - | - | - |"
            )
            continue
        device_peak = telemetry_metric(result, "memory_used_maximum")
        utilization = telemetry_metric(result, "utilization_gpu_average")
        scaling = result.get("scaling_efficiency")
        lines.append(
            f"| success | {result['benchmark_type']} | {result['model_size']} | "
            f"{result['variant']} | {result['gpu_count']} | {result['per_device_batch_size']} | "
            f"{result['gradient_accumulation_steps']} | {result['effective_global_batch_size']} | "
            f"{result['elapsed_seconds']:.1f} | {result['non_padding_tokens_per_second']:.0f} | "
            f"{result['peak_memory_allocated_gib']:.1f} | {optional_number(device_peak, 0)} | "
            f"{optional_number(utilization, 1)} | {optional_number(result.get('throughput_vs_baseline'), 2)} | "
            f"{optional_number(100 * scaling if scaling is not None else None, 1)} |"
        )
    lines.extend(["", "## Comparison summary", ""])
    for item in summary["comparisons"]:
        lines.append(
            f"- `{item['benchmark_type']}/{item['model_size']}`: fastest "
            f"`{item['fastest_variant']}` at {item['fastest_tokens_per_second']:.0f} tokens/s; "
            f"lowest PyTorch VRAM `{item['lowest_vram_variant']}` at "
            f"{item['lowest_peak_pytorch_allocated_gib']:.1f} GiB."
        )
    if summary["failed_case_ids"]:
        lines.extend(["", "## Failed runs", ""])
        lines.extend(f"- `{case_id}`" for case_id in summary["failed_case_ids"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_parent(config_path: Path, config: dict, settings: dict, dry_run: bool) -> None:
    output_dir = Path(settings["output_dir"]).expanduser().resolve()
    cases = build_benchmark_cases(config, settings)
    run_started_at = utc_now()
    run_started = time.perf_counter()
    env = os.environ.copy()
    devices = settings.get("devices")
    if devices:
        if isinstance(devices, list):
            devices = ",".join(str(device) for device in devices)
        env["CUDA_VISIBLE_DEVICES"] = str(devices)

    max_requested = max(case["gpu_count"] for case in cases)
    if not dry_run:
        if devices:
            available = len(
                [device for device in str(devices).split(",") if device.strip()]
            )
        else:
            import torch

            available = torch.cuda.device_count()
        if available < max_requested:
            raise RuntimeError(
                f"Requested up to {max_requested} GPUs, but only {available} are visible"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

    # Each model size has different kernels/shapes, so prime every selected
    # model once before comparing its matrix entries.
    for model_size in settings["model_sizes"]:
        representative = next(
            case for case in cases if case["model_size"] == model_size
        )
        warmup_case = copy.deepcopy(representative)
        warmup_case.update(
            {
                "id": f"warmup/{model_size}",
                "benchmark_type": "warmup",
                "variant": "warmup",
                "gpu_count": 1,
                "max_steps": settings["warmup_steps"],
                "run_evaluation": False,
            }
        )
        # The profile baseline is guaranteed to support every configured GPU count.
        profile = settings["model_profiles"][model_size]
        warmup_case["per_device_batch_size"] = profile["per_device_batch_size"]
        warmup_case["effective_batch_size"] = profile["effective_batch_size"]
        warmup_case["gradient_accumulation_steps"] = gradient_accumulation_steps(
            warmup_case
        )
        warmup_config = resolve_accelerate_config(config_path, settings, 1)
        warmup_result = output_dir / "warmup" / model_size / "result.json"
        command = worker_command(config_path, warmup_case, warmup_result, warmup_config)
        print(
            f"# Warm-up {model_size}: {settings['warmup_steps']} discarded optimizer steps",
            flush=True,
        )
        print("$ " + shlex.join(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)

    results = []
    for case in cases:
        accelerate_config = resolve_accelerate_config(
            config_path, settings, case["gpu_count"]
        )
        result_path = output_dir / Path(case["id"]) / "result.json"
        command = worker_command(config_path, case, result_path, accelerate_config)
        print(
            f"# {case['id']}: GPUs={case['gpu_count']} per_device={case['per_device_batch_size']} "
            f"accumulation={case['gradient_accumulation_steps']} "
            f"effective={case['effective_batch_size']} checkpointing={case['gradient_checkpointing']} "
            f"packing={case['packing']}",
            flush=True,
        )
        print("$ " + shlex.join(command), flush=True)
        if dry_run:
            continue
        telemetry_path = result_path.parent / "gpu_telemetry.json"
        log_path = result_path.parent / "run.log"
        job_started = time.perf_counter()
        return_code, telemetry = run_with_gpu_telemetry(
            command, env, case, settings, telemetry_path, log_path
        )
        job_wall_seconds = time.perf_counter() - job_started
        if return_code == 0 and result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result.update(
                {
                    "job_wall_seconds": job_wall_seconds,
                    "gpu_telemetry": telemetry,
                    "run_log": str(log_path),
                    "launch_command": shlex.join(command),
                }
            )
        else:
            result = {
                "status": "failed",
                "id": case["id"],
                "benchmark_type": case["benchmark_type"],
                "model_size": case["model_size"],
                "variant": case["variant"],
                "gpu_count": case["gpu_count"],
                "per_device_batch_size": case["per_device_batch_size"],
                "gradient_accumulation_steps": case["gradient_accumulation_steps"],
                "effective_global_batch_size": case["effective_batch_size"],
                "gradient_checkpointing": case["gradient_checkpointing"],
                "packing": case["packing"],
                "exit_code": return_code,
                "failure_reason": "Accelerate worker exited without a successful result",
                "job_wall_seconds": job_wall_seconds,
                "gpu_telemetry": telemetry,
                "run_log": str(log_path),
                "launch_command": shlex.join(command),
            }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results.append(result)
        if result["status"] == "failed" and settings["fail_fast"]:
            break

    if dry_run:
        print(
            f"\nValidated {len(cases)} benchmark runs across {len(settings['model_sizes'])} model profile(s)."
        )
        return
    add_comparison_metrics(results)
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
    )
    report_path = output_dir / settings["report_filename"]
    csv_path = report_path.with_suffix(".csv")
    markdown_path = report_path.with_name(f"{report_path.stem}_summary.md")
    html_path = report_path.with_name(f"{report_path.stem}_summary.html")
    report = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at": utc_now(),
        "run_started_at": run_started_at,
        "orchestration_wall_seconds": time.perf_counter() - run_started,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "system": collect_system_inventory(settings),
        "datasets": dataset_manifest(config_path, config, cases),
        "benchmark": settings,
        "training_config": {
            "model": config.get("model"),
            "data": config.get("data"),
            "lora": config.get("lora"),
            "training": config.get("training"),
        },
        "summary": build_benchmark_summary(results),
        "report_artifacts": {
            "json": str(report_path),
            "csv": str(csv_path),
            "markdown": str(markdown_path),
            "html": str(html_path),
        },
        "results": results,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv_report(csv_path, results)
    write_markdown_summary(markdown_path, report)
    render_rich_report(report, html_path)
    print(
        f"\nJSON report:     {report_path}\n"
        f"CSV report:      {csv_path}\n"
        f"Markdown summary: {markdown_path}\n"
        f"HTML summary:     {html_path}"
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    settings = resolve_benchmark(config, args)
    if args.worker:
        if (
            args.result_path is None
            or args.accelerate_config is None
            or args.run_spec is None
        ):
            raise ValueError(
                "Internal worker mode requires --result-path, --accelerate-config, and --run-spec"
            )
        case = json.loads(args.run_spec)
        run_worker(
            config_path,
            case,
            Path(args.result_path).resolve(),
            Path(args.accelerate_config).resolve(),
        )
    else:
        run_parent(config_path, config, settings, args.dry_run)


if __name__ == "__main__":
    main()
