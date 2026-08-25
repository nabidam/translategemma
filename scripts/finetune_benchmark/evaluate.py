"""Evaluation stage: every system on every test set, through translation_benchmark.

Nothing about scoring is reimplemented here. For each test set the sweep renders
a benchmark_config.yaml whose candidate list is the ten systems (two base models
plus four volumes each), then drives the repository's own three-phase pipeline:

  generate  one candidate per GPU, four at a time
  score     transparent metrics + XCOMET + MetricX + paired bootstrap CIs
  report    the per-test-set HTML explorer

Generation is the only phase that parallelises across candidates; scoring and
report rendering are single-process by design in translation_benchmark, so the
sweep parallelises them across TEST SETS instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .config import (
    BASE_BENCHMARK_CONFIG,
    CAUSAL_LORA,
    SweepConfig,
    System,
    deep_merge,
    load_yaml,
    write_yaml,
)
from .scheduler import Job, logger, run_jobs

GENERATION_PROFILE = "sweep"


def _candidate(config: SweepConfig, system: System) -> dict[str, Any]:
    evaluation = system.model["evaluation"]
    candidate: dict[str, Any] = {
        "id": system.id,
        "label": system.label,
        "family": system.model.get("family", system.model_key),
        "size": system.model.get("size"),
        "type": "generated",
        "runner": evaluation["runner"],
        "model": system.model["base_model_id"],
        "source_lang": evaluation["source_lang"],
        "target_lang": evaluation["target_lang"],
        "dtype": evaluation.get("dtype", "bfloat16"),
        "generation_profile": GENERATION_PROFILE,
        "enabled": True,
    }
    if system.kind == CAUSAL_LORA and evaluation.get("attn_implementation"):
        candidate["attn_implementation"] = evaluation["attn_implementation"]
    if not system.is_base:
        candidate["adapter"] = str(config.adapter_path(system))
    return candidate


def _available_systems(config: SweepConfig) -> list[System]:
    """Systems that can actually be evaluated: base models plus trained adapters."""
    available: list[System] = []
    for system in config.systems:
        if system.is_base or config.adapter_path(system).exists():
            available.append(system)
        else:
            logger.warning(
                "Skipping %s: no adapter at %s (its fine-tune cell has not run or failed).",
                system.id, config.adapter_path(system),
            )
    return available


def benchmark_config_path(config: SweepConfig, test_set: dict[str, Any]) -> Path:
    return config.derived_config_dir / f"benchmark_{test_set['id']}.yaml"


def write_benchmark_config(config: SweepConfig, test_set: dict[str, Any], systems: list[System]) -> Path:
    dataset_path = config.work_dir / "testsets" / f"{test_set['id']}.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} is missing. Run the data stage before the evaluate stage.")
    payload = deep_merge(
        load_yaml(BASE_BENCHMARK_CONFIG),
        deep_merge(
            config.evaluation.get("overrides") or {},
            {
                "benchmark": {
                    "title": f"{test_set['label']} — fine-tuning volume sweep",
                    "output_dir": str(config.evaluation_dir / test_set["id"]),
                    "dataset": {
                        "path": str(dataset_path),
                        "source_lang": config.corpus["source_lang"],
                        "target_lang": config.corpus["target_lang"],
                        # The prepared copy is always written in this shape.
                        "columns": {"id": "id", "source": "en", "reference": "fa", "domain": "domain"},
                    },
                },
                "generation_profiles": {GENERATION_PROFILE: dict(config.evaluation["generation"])},
            },
        ),
    )
    # Replaced, not merged: the template's example candidates must not survive.
    payload["candidates"] = [_candidate(config, system) for system in systems]
    return write_yaml(benchmark_config_path(config, test_set), payload)


def _benchmark_command(config_path: Path, command: str, extra: list[str] | None = None) -> list[str]:
    return [sys.executable, "benchmark_translations.py", "--config", str(config_path), command, *(extra or [])]


def _job(config: SweepConfig, test_set_id: str, phase: str, name: str, command: list[str], metadata: dict) -> Job:
    directory = config.jobs_dir / "evaluate" / test_set_id / phase / name
    return Job(
        id=f"{phase}-{test_set_id}-{name}",
        stage=phase,
        command=command,
        result_path=directory / "result.json",
        log_path=directory / f"{phase}.log",
        telemetry_path=directory / "telemetry.json",
        metadata={"test_set": test_set_id, "phase": phase, **metadata},
    )


def run(config: SweepConfig, force: bool = False, test_set_ids: list[str] | None = None) -> dict:
    systems = _available_systems(config)
    if not systems:
        raise RuntimeError("No evaluable system: neither base models nor adapters resolved.")
    test_sets = [item for item in config.test_sets if not test_set_ids or item["id"] in test_set_ids]
    if not test_sets:
        raise ValueError(f"No enabled test set matches {test_set_ids}")

    config_paths = {item["id"]: write_benchmark_config(config, item, systems) for item in test_sets}
    telemetry = config.sweep.get("telemetry", {})
    fail_fast = bool(config.sweep.get("fail_fast", False))
    results: dict[str, dict] = {}

    generation_jobs = [
        _job(
            config,
            test_set["id"],
            "generate",
            system.id,
            _benchmark_command(
                config_paths[test_set["id"]], "generate", ["--candidates", system.id] + (["--force"] if force else [])
            ),
            {"system_id": system.id, "system_label": system.label, "volume": system.volume,
             "volume_label": system.volume_label, "model_key": system.model_key},
        )
        for test_set in test_sets
        for system in systems
    ]
    logger.info(
        "Generating %d candidate output(s) (%d systems x %d test sets) across GPUs %s",
        len(generation_jobs), len(systems), len(test_sets), config.gpus,
    )
    results.update(run_jobs(generation_jobs, config.gpus, telemetry, force=force, fail_fast=fail_fast))

    failed = [job.id for job in generation_jobs if results.get(job.id, {}).get("status") != "ok"]
    if failed:
        logger.warning(
            "%d generation job(s) failed; scoring a test set requires all of its candidates: %s",
            len(failed), failed,
        )

    # Scoring loads XCOMET and MetricX, so it wants a GPU of its own; one job per
    # test set keeps the three of them running side by side.
    scored_test_sets = [
        test_set for test_set in test_sets
        if not any(job.metadata["test_set"] == test_set["id"] and job.id in failed for job in generation_jobs)
    ]
    for phase in ("score", "report"):
        jobs = [
            _job(config, test_set["id"], phase, "all", _benchmark_command(config_paths[test_set["id"]], phase), {})
            for test_set in scored_test_sets
        ]
        if not jobs:
            continue
        logger.info("Running %s for %d test set(s)", phase, len(jobs))
        results.update(run_jobs(jobs, config.gpus, telemetry, force=force, fail_fast=fail_fast))
        if any(results.get(job.id, {}).get("status") != "ok" for job in jobs):
            logger.error("A %s job failed; the report stage will only cover completed test sets.", phase)

    path = config.output_dir / "evaluate_stage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"config_paths": {key: str(value) for key, value in config_paths.items()}, "jobs": results},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Evaluation stage complete: %s", path)
    return results
