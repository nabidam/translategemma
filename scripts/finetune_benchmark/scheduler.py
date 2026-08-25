"""One-GPU-per-job scheduler with per-job telemetry, logging, and resume.

Every unit of work in this sweep — a fine-tune, a candidate's generation pass —
is a single-GPU subprocess. That is what makes the matrix embarrassingly
parallel across the host's four GPUs, and it is also why each job must be
isolated: CUDA_VISIBLE_DEVICES is set per process, so a job can only ever see
the one device the pool handed it.

A finished job writes result.json. A later run reads it back instead of
repeating the work, so an interrupted sweep resumes at the cell it died on.
The nvidia-smi sampling helpers are imported from scripts/benchmark_training.py
rather than reimplemented, so telemetry numbers stay comparable with the
training benchmark's; the runner itself is separate because that one streams the
child's output to the shared stdout, which four concurrent jobs cannot share.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from logging_utils import console, logger  # noqa: E402  (path set above)
import benchmark_training as training_benchmark  # noqa: E402


@dataclass(frozen=True)
class Job:
    """A single-GPU subprocess plus where its artefacts live."""

    id: str
    stage: str
    command: list[str]
    result_path: Path
    log_path: Path
    telemetry_path: Path
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _peak_and_average(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten the per-GPU telemetry summary into the few numbers the report uses."""
    per_gpu = summary.get("per_gpu") or []
    if not per_gpu:
        return {}
    gpu = per_gpu[0]
    return {
        "gpu_name": gpu.get("name"),
        "vram_peak_mib": gpu.get("memory_used_maximum"),
        "vram_total_mib": gpu.get("memory_total_maximum"),
        "gpu_utilization_average_percent": gpu.get("utilization_gpu_average"),
        "power_draw_average_watts": gpu.get("power_draw_average"),
        "energy_wh": gpu.get("estimated_energy_wh"),
        "temperature_maximum_celsius": gpu.get("temperature_gpu_maximum"),
        "sm_clock_average_mhz": gpu.get("clocks_sm_average"),
    }


def _run_with_telemetry(job: Job, gpu: int, settings: dict[str, Any]) -> dict[str, Any]:
    """Run one job pinned to one GPU, sampling that GPU while it runs.

    Output goes to the job's own log file only. Four jobs interleaving their
    training logs on one terminal is unreadable, and the parent already reports
    start/finish and the failing tail.
    """
    enabled = bool(settings.get("enabled", True))
    interval = float(settings.get("interval_seconds", 2.0))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), **job.env}
    job.log_path.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    started_at, started = _utc_now(), time.perf_counter()
    with job.log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            job.command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            if enabled:
                rows, error = training_benchmark.query_nvidia_telemetry([str(gpu)])
                if error and (not errors or errors[-1] != error):
                    errors.append(error)
                if rows:
                    samples.append(
                        {
                            "captured_at": _utc_now(),
                            "elapsed_seconds": time.perf_counter() - started,
                            "gpus": rows,
                            # Host CPU/RAM is deliberately not sampled here: with
                            # four concurrent jobs it cannot be attributed to one.
                            "host": {},
                        }
                    )
            try:
                process.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                pass
    duration = time.perf_counter() - started

    summary = training_benchmark.summarize_gpu_telemetry(samples, duration)
    if enabled and samples:
        job.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        job.telemetry_path.write_text(
            json.dumps(
                {
                    "job_id": job.id,
                    "gpu": gpu,
                    "interval_seconds": interval,
                    "summary": summary,
                    "samples": samples,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return {
        "job_id": job.id,
        "stage": job.stage,
        "status": "ok" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "gpu": gpu,
        "command": job.command,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": duration,
        "log_path": str(job.log_path),
        "telemetry_path": str(job.telemetry_path) if enabled and samples else None,
        "telemetry_errors": errors,
        "hardware": _peak_and_average(summary),
        "metadata": job.metadata,
    }


def verify_gpus(gpus: list[int]) -> None:
    """Fail before any job starts if a configured GPU id is not addressable.

    The ids in sweep.gpus are physical nvidia-smi indices, and they are also
    what each job gets as CUDA_VISIBLE_DEVICES. Inside a container started with
    a partial device list (GPUS=4,5,6,7) the driver renumbers those devices to
    0..3, so the configured ids would address nothing. Run the container with
    every GPU visible (GPUS=all) and select here instead.
    """
    rows, error = training_benchmark.query_nvidia_telemetry()
    if error or not rows:
        logger.warning("Could not verify GPU ids (%s); proceeding with %s.", error or "no nvidia-smi output", gpus)
        return
    visible = {int(row["index"]) for row in rows if row.get("index") is not None}
    if missing := sorted(set(gpus) - visible):
        raise RuntimeError(
            f"sweep.gpus lists {missing}, but this process can only see GPU indices {sorted(visible)}. "
            "Start the container with all GPUs visible (GPUS=all docker compose run --rm trainer ...) "
            "and let the sweep select devices, or set sweep.gpus to the ids visible here."
        )
    logger.info("GPU pool verified: %s of visible %s", gpus, sorted(visible))


def _cached_result(job: Job) -> dict[str, Any] | None:
    if not job.result_path.exists():
        return None
    try:
        result = json.loads(job.result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return result if result.get("status") == "ok" else None


def _log_failure_tail(job: Job, lines: int = 25) -> None:
    if not job.log_path.exists():
        return
    tail = job.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    logger.error("Last %d log lines of %s:\n%s", len(tail), job.id, "\n".join(tail))


def run_jobs(
    jobs: list[Job],
    gpus: list[int],
    telemetry_settings: dict[str, Any],
    force: bool = False,
    fail_fast: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run jobs across a GPU pool, at most one job per GPU at a time."""
    results: dict[str, dict[str, Any]] = {}
    pending: list[Job] = []
    for job in jobs:
        cached = None if force else _cached_result(job)
        if cached:
            logger.info(
                "Reusing completed job [bold green]%s[/bold green] (%.1f min, %s)",
                job.id,
                cached.get("duration_seconds", 0) / 60,
                job.result_path,
            )
            results[job.id] = cached
        else:
            pending.append(job)
    if not pending:
        return results

    verify_gpus(gpus)
    available: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        available.put(gpu)
    stop = threading.Event()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("[dim]elapsed[/dim]"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=2,
    )

    def execute(job: Job) -> dict[str, Any]:
        if stop.is_set():
            return {"job_id": job.id, "stage": job.stage, "status": "skipped", "metadata": job.metadata}
        gpu = available.get()
        try:
            logger.info("[bold]%s[/bold] starting on GPU %d", job.id, gpu)
            result = _run_with_telemetry(job, gpu, telemetry_settings)
        finally:
            available.put(gpu)
        job.result_path.parent.mkdir(parents=True, exist_ok=True)
        job.result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        if result["status"] == "ok":
            logger.info(
                "[bold green]%s[/bold green] finished in %.1f min (peak VRAM %s MiB, %s Wh)",
                job.id,
                result["duration_seconds"] / 60,
                _format_number(result["hardware"].get("vram_peak_mib")),
                _format_number(result["hardware"].get("energy_wh")),
            )
        else:
            logger.error("[bold red]%s[/bold red] failed with exit code %s", job.id, result["returncode"])
            _log_failure_tail(job)
            if fail_fast:
                stop.set()
        return result

    with progress:
        task = progress.add_task(f"{pending[0].stage} jobs", total=len(pending))
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            for result in pool.map(execute, pending):
                results[result["job_id"]] = result
                progress.advance(task)

    failed = [job_id for job_id, result in results.items() if result["status"] != "ok"]
    if failed and fail_fast:
        raise RuntimeError(f"Jobs failed: {failed}. Inspect their log_path in {jobs[0].result_path.parent.parent}.")
    if failed:
        logger.warning("Continuing with %d failed job(s): %s", len(failed), failed)
    return results


def _format_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):,.0f}"
