"""Fine-tune stage: one LoRA job per (model, data volume) cell.

TranslateGemma cells run the repository's train.py against a generated config,
so the validated SFT path (packing, Liger fused CE, FlashAttention 3, the
graceful-stop callback) is used exactly as it is in production. NLLB cells run
train_nllb_lora.py from this directory. Both are single-GPU subprocesses, so the
eight cells of the default matrix occupy the four GPUs two rounds deep.

Cost is measured, not assumed: wall clock comes from the scheduler, and peak
VRAM, mean utilisation, mean power and estimated energy come from per-job
nvidia-smi sampling.
"""

from __future__ import annotations

import json
import math
import sys

from .config import (
    BASE_TRAINING_CONFIG,
    CAUSAL_LORA,
    SEQ2SEQ_LORA,
    SweepConfig,
    System,
    deep_merge,
    load_yaml,
    write_yaml,
)
from .scheduler import Job, logger, run_jobs


def _job_dir(config: SweepConfig, system: System):
    return config.jobs_dir / "finetune" / system.id


def _schedule_overrides(config: SweepConfig, model_key: str) -> dict:
    """Eval/checkpoint cadence for the active budget mode.

    In fixed_steps mode an epoch boundary may never be reached, so the
    epoch-based cadence configured for fixed_epochs mode would produce no
    evaluation and no checkpoint — and `load_best_model_at_end` would then have
    nothing to load. Derive a step interval from the step budget instead.
    """
    max_steps = config.max_steps_for(model_key)
    if max_steps is None:
        return {}
    interval = max(1, math.ceil(max_steps / config.budget["evals_per_run"]))
    return {"eval_steps": interval, "save_steps": interval}


def _causal_command(config: SweepConfig, system: System) -> list[str]:
    """Render a per-cell config.yaml and return the trainer invocation."""
    splits = config.subset_split_paths(system.volume)
    output_dir = config.finetune_output_dir(system)
    validation = str(splits["validation"]) if float(config.data["validation_ratio"]) > 0 else None
    max_steps = config.max_steps_for(system.model_key)
    schedule = _schedule_overrides(config, system.model_key)
    if schedule:
        schedule.update(evaluation_strategy="steps", save_strategy="steps")
    overrides = deep_merge(
        system.model.get("overrides") or {},
        {
            "model": {"base_model_id": system.model["base_model_id"], "output_dir": str(output_dir)},
            "data": {
                "train_sft_dataset_path": str(splits["train"]),
                "validation_sft_dataset_path": validation,
                # The sweep evaluates through translation_benchmark, never
                # through train.py's own post-training evaluation.
                "test_dataset_path": None,
                # Per-cell so the eight concurrent jobs cannot race on one
                # tokenized/packed Arrow cache directory.
                "prepared_cache_dir": str(output_dir / "prepared_cache"),
                "source_lang": config.corpus["source_lang"],
                "target_lang": config.corpus["target_lang"],
            },
            "training": {
                "run_sft": True,
                "run_dpo": False,
                # With max_steps set, Trainer stops at that step count and this
                # is only the upper bound on how often the data may be revisited.
                "epochs": config.epochs if max_steps is None else max(config.epochs, 1000),
                "seed": int(config.sweep["seed"]),
                "load_best_model_at_end": validation is not None,
                **schedule,
            },
            "evaluation": {"run_after_training": False},
            "logging": {"logs_dir": str(output_dir / "logs")},
        },
    )
    derived = write_yaml(
        config.derived_config_dir / f"train_{system.id}.yaml",
        deep_merge(load_yaml(BASE_TRAINING_CONFIG), overrides),
    )
    if max_steps is None:
        # The plain repository entry point, with nothing between it and the cell.
        return [sys.executable, "train.py", "--config", str(derived)]
    # train.py's CLI exposes a step cap only through --canary, which also caps
    # rows; the wrapper calls its run_pipeline with just the cap.
    return [
        sys.executable, "-m", "scripts.finetune_benchmark.train_causal_lora",
        "--config", str(derived), "--max-steps", str(max_steps),
    ]


def _seq2seq_command(config: SweepConfig, system: System) -> list[str]:
    splits = config.subset_split_paths(system.volume)
    max_steps = config.max_steps_for(system.model_key)
    command = [
        sys.executable,
        "-m",
        "scripts.finetune_benchmark.train_nllb_lora",
        "--config",
        str(config.path),
        "--model-key",
        system.model_key,
        "--train",
        str(splits["train"]),
        "--output-dir",
        str(config.finetune_output_dir(system)),
        "--epochs",
        # An epoch count large enough not to bind before the step cap does.
        str(config.epochs if max_steps is None else max(config.epochs, 1000)),
    ]
    if float(config.data["validation_ratio"]) > 0:
        command += ["--validation", str(splits["validation"])]
    if max_steps is not None:
        command += ["--max-steps", str(max_steps)]
        interval = max(1, math.ceil(max_steps / config.budget["evals_per_run"]))
        command += ["--eval-save-steps", str(interval)]
    return command


def build_jobs(config: SweepConfig, systems: list[System] | None = None) -> list[Job]:
    jobs: list[Job] = []
    for system in systems or config.finetune_systems:
        splits = config.subset_split_paths(system.volume)
        if not splits["train"].exists():
            raise FileNotFoundError(
                f"{splits['train']} is missing. Run the data stage before the finetune stage."
            )
        builders = {CAUSAL_LORA: _causal_command, SEQ2SEQ_LORA: _seq2seq_command}
        directory = _job_dir(config, system)
        jobs.append(
            Job(
                id=f"finetune-{system.id}",
                stage="finetune",
                command=builders[system.kind](config, system),
                result_path=directory / "result.json",
                log_path=directory / "train.log",
                telemetry_path=directory / "telemetry.json",
                metadata={
                    "system_id": system.id,
                    "system_label": system.label,
                    "model_key": system.model_key,
                    "kind": system.kind,
                    "base_model_id": system.model["base_model_id"],
                    "volume": system.volume,
                    "volume_label": system.volume_label,
                    "budget_mode": config.budget["mode"],
                    "epochs": config.epochs,
                    "max_steps": config.max_steps_for(system.model_key),
                    "train_path": str(splits["train"]),
                    "adapter_path": str(config.adapter_path(system)),
                },
            )
        )
    return jobs


def run(config: SweepConfig, force: bool = False, systems: list[System] | None = None) -> dict:
    jobs = build_jobs(config, systems)
    budget = config.budget
    logger.info(
        "Fine-tuning %d cell(s) across GPUs %s, budget %s (%s)",
        len(jobs), config.gpus, budget["mode"],
        f"{budget['epochs']} epoch(s)" if budget["mode"] == "fixed_epochs"
        else ", ".join(f"{key}={config.max_steps_for(key)} steps" for key in config.models),
    )
    results = run_jobs(
        jobs,
        config.gpus,
        config.sweep.get("telemetry", {}),
        force=force,
        fail_fast=bool(config.sweep.get("fail_fast", False)),
    )
    # An exit code of zero with no adapter on disk is a silent failure that
    # would surface much later as an unresolvable evaluation candidate.
    for job in jobs:
        result = results.get(job.id, {})
        adapter = config.adapter_path(config.system(job.metadata["system_id"]))
        if result.get("status") == "ok" and not adapter.exists():
            result["status"] = "failed"
            result["error"] = f"trainer exited 0 but wrote no adapter at {adapter}"
            job.result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.error("%s: %s", job.id, result["error"])
    path = config.output_dir / "finetune_stage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Fine-tune stage complete: %s", path)
    return results
