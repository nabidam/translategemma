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
    BASE_TESTSET_CONFIG,
    BASE_TRAINING_CONFIG,
    CAUSAL_LORA,
    SweepConfig,
    System,
    deep_merge,
    load_yaml,
    write_yaml,
)
from . import preflight
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
    # Pinning the checkpoint is the difference between a reproducible published
    # number and one that silently moves when the Hub repository is updated
    # (docs/TRANSLATION_BENCHMARK.md, fair-comparison checklist).
    for key in ("revision", "processor", "tokenizer"):
        if value := evaluation.get(key):
            candidate[key] = value
    # Per-arm decoding settings, merged over the shared profile by the benchmark.
    # A 3.3B seq2seq and a 12B decoder do not want the same batch size on the
    # same card. NOTE: this lands in the candidate, whose hash decides output
    # reuse, so adding or changing it forces that candidate to regenerate.
    if generation := evaluation.get("generation"):
        candidate["generation"] = dict(generation)
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
                # Per-test-set overrides live in the profile rather than in the
                # candidates, so raising a batch size for one test set does not
                # invalidate outputs already collected for the others.
                "generation_profiles": {
                    GENERATION_PROFILE: {
                        **config.evaluation["generation"],
                        **(test_set.get("generation") or {}),
                    }
                },
            },
        ),
    )
    # Replaced, not merged: the template's example candidates must not survive.
    payload["candidates"] = [_candidate(config, system) for system in systems]
    return write_yaml(benchmark_config_path(config, test_set), payload)


def write_staging_configs(config: SweepConfig) -> dict[str, Path]:
    """Emit the configs scripts/fetch_offline_assets.py reads, for this sweep.

    Asset staging happens on the ONLINE machine, before the sweep's own derived
    configs exist (those need the prepared test sets). This renders the same
    model and metric choices into the three files the staging script expects, so
    nothing the sweep will load offline is missed: both base checkpoints, XCOMET
    with the encoder it pulls in, MetricX with its mT5 tokenizer, and the test-set
    builder's sentence embedding model.

    Local adapters are deliberately absent — they do not exist yet, and they are
    produced on the offline host anyway.
    """
    metrics = (config.evaluation.get("overrides") or {}).get("metrics") or {}
    comet, metricx = metrics.get("comet") or {}, metrics.get("metricx") or {}
    training_config = deep_merge(
        load_yaml(BASE_TRAINING_CONFIG),
        {
            "model": {
                "base_model_id": next(
                    (model["base_model_id"] for model in config.models.values()
                     if model["kind"] == CAUSAL_LORA),
                    next(iter(config.models.values()))["base_model_id"],
                )
            },
            "evaluation": {
                "metricx_enabled": bool(metricx.get("enabled", False)),
                "metricx_model_id": metricx.get("model") or load_yaml(BASE_TRAINING_CONFIG)["evaluation"]["metricx_model_id"],
                "metricx_tokenizer_id": metricx.get("tokenizer") or load_yaml(BASE_TRAINING_CONFIG)["evaluation"]["metricx_tokenizer_id"],
                "comet_enabled": bool(comet.get("enabled", False)),
                "comet_model_id": comet.get("model") or load_yaml(BASE_TRAINING_CONFIG)["evaluation"]["comet_model_id"],
            },
        },
    )
    # Base models only, and every one marked enabled: the staging script skips
    # candidates that are disabled or imported.
    base_systems = [system for system in config.systems if system.is_base]
    benchmark_config = deep_merge(
        load_yaml(BASE_BENCHMARK_CONFIG),
        deep_merge(config.evaluation.get("overrides") or {}, {}),
    )
    benchmark_config["candidates"] = [_candidate(config, system) for system in base_systems]
    testset_config = deep_merge(
        load_yaml(BASE_TESTSET_CONFIG), config.data["in_domain_test"].get("overrides") or {}
    )
    paths = {
        "config": write_yaml(config.derived_config_dir / "staging_config.yaml", training_config),
        "benchmark_config": write_yaml(config.derived_config_dir / "staging_benchmark.yaml", benchmark_config),
        "testset_config": write_yaml(config.derived_config_dir / "staging_testset.yaml", testset_config),
    }
    logger.info("Staging configs written. On the ONLINE machine, with HF_TOKEN exported, run:")
    logger.info(
        "  uv run --no-project --with huggingface_hub --with pyyaml python scripts/fetch_offline_assets.py "
        "--config %s --testset-config %s --benchmark-config %s --dest offline_assets/models",
        paths["config"], paths["testset_config"], paths["benchmark_config"],
    )
    required = {system.model["base_model_id"] for system in config.systems}
    required.update(
        value for value in (comet.get("model"), metricx.get("model"), metricx.get("tokenizer")) if value
    )
    if embedding_model := (testset_config.get("embeddings") or {}).get("model"):
        required.add(embedding_model)
    logger.info("Repositories this sweep needs: %s", sorted(required))
    return paths


def _benchmark_command(config_path: Path, command: str, extra: list[str] | None = None) -> list[str]:
    if command == "score":
        # The sweep's own scorer: identical to benchmark_translations.py score
        # except that MetricX runs with use_cache=False, without which its
        # forward pass crashes. See scripts/finetune_benchmark/score.py.
        return [
            sys.executable, "-m", "scripts.finetune_benchmark.score",
            "--config", str(config_path), *(extra or []),
        ]
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
    preflight.enforce(config, "evaluate")
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

    # A failed candidate must not take its whole test set down with it: score
    # the candidates that did produce output, and say which ones are missing.
    # Scoring every candidate named in the config would fail on the absent file.
    scorable: dict[str, list[str]] = {}
    for test_set in test_sets:
        succeeded = [
            job.metadata["system_id"]
            for job in generation_jobs
            if job.metadata["test_set"] == test_set["id"] and results.get(job.id, {}).get("status") == "ok"
        ]
        missing = [system.id for system in systems if system.id not in succeeded]
        if missing:
            logger.warning(
                "Test set %s: scoring %d of %d candidates; no output for %s.",
                test_set["id"], len(succeeded), len(systems), missing,
            )
        if len(succeeded) < 2:
            logger.error(
                "Test set %s has %d usable candidate(s); pairwise statistics need at least two. Skipping "
                "its scoring — fix the failed generation job(s) and re-run this stage.",
                test_set["id"], len(succeeded),
            )
            continue
        scorable[test_set["id"]] = succeeded

    # Scoring loads XCOMET and MetricX, so it wants a GPU of its own; one job per
    # test set keeps them running side by side.
    score_jobs = [
        _job(
            config, test_set_id, "score", "all",
            _benchmark_command(config_paths[test_set_id], "score", ["--candidates", *candidate_ids]),
            {"candidates": candidate_ids},
        )
        for test_set_id, candidate_ids in scorable.items()
    ]
    if score_jobs:
        logger.info("Running score for %d test set(s)", len(score_jobs))
        results.update(run_jobs(score_jobs, config.gpus, telemetry, force=force, fail_fast=fail_fast))

    # Only test sets that actually scored get a report: the benchmark's report
    # stage reads scores.csv and friends, so running it after a failed scoring
    # job produces nothing but a second, more confusing traceback.
    scored = [job.metadata["test_set"] for job in score_jobs if results.get(job.id, {}).get("status") == "ok"]
    if unscored := [test_set_id for test_set_id in scorable if test_set_id not in scored]:
        logger.error(
            "Scoring failed for %s; skipping their reports. Fix the cause and re-run this stage — "
            "the collected translations are reused.", unscored,
        )
    report_jobs = [
        _job(config, test_set_id, "report", "all", _benchmark_command(config_paths[test_set_id], "report"), {})
        for test_set_id in scored
    ]
    if report_jobs:
        logger.info("Running report for %d test set(s)", len(report_jobs))
        results.update(run_jobs(report_jobs, config.gpus, telemetry, force=force, fail_fast=fail_fast))

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
