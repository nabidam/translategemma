#!/usr/bin/env python3
"""Fine-tuning data-volume benchmark: TranslateGemma vs NLLB-200.

Four stages, each resumable and independently runnable:

    plan      print the matrix, the job counts and every path, run nothing
    preflight verify staged checkpoints, test sets, GPUs and disk (offline, seconds)
    assets    (online machine) emit configs for scripts/fetch_offline_assets.py
    data      normalize the corpus, carve out the in-domain test set, build the
              nested training subsets and their train/validation splits
    finetune  one LoRA job per (model, volume) cell, one GPU each
    evaluate  every system on every test set through translation_benchmark
    report    cross-test-set tables, statistics, and the HTML conclusion

Inside the offline image:

    docker compose run --rm trainer \
        python -m scripts.finetune_benchmark.run_sweep all \
            --config scripts/finetune_benchmark/sweep_config.yaml

Re-running a stage reuses every completed job; pass --force to redo them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):  # Allow `python scripts/finetune_benchmark/run_sweep.py`.
    sys.path.insert(0, str(PROJECT_ROOT))
    __package__ = "scripts.finetune_benchmark"

from scripts.finetune_benchmark import data_prep, evaluate, finetune, preflight, report  # noqa: E402
from scripts.finetune_benchmark.config import load_sweep_config, volume_label  # noqa: E402
from scripts.finetune_benchmark.scheduler import console, logger  # noqa: E402

from logging_utils import setup_logging  # noqa: E402
from rich.table import Table  # noqa: E402

STAGES = ("data", "finetune", "evaluate", "report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=[*STAGES, "all", "plan", "assets", "preflight"])
    parser.add_argument("--config", default="scripts/finetune_benchmark/sweep_config.yaml")
    parser.add_argument("--force", action="store_true", help="Redo completed work instead of reusing it.")
    parser.add_argument("--systems", nargs="+", help="Limit the finetune stage to these system ids.")
    parser.add_argument("--test-sets", nargs="+", help="Limit the evaluate stage to these test set ids.")
    return parser.parse_args()


def show_plan(config) -> None:
    table = Table(title="Sweep matrix", header_style="bold cyan", border_style="green")
    table.add_column("System", style="cyan")
    table.add_column("Model")
    table.add_column("Rows", justify="right")
    table.add_column("Adapter / base")
    for system in config.systems:
        table.add_row(
            system.id,
            system.model["base_model_id"],
            "—" if system.is_base else f"{system.volume:,}",
            "base model" if system.is_base else str(config.adapter_path(system)),
        )
    console.print(table)

    plan = Table(title="Work units", header_style="bold cyan", border_style="green")
    plan.add_column("Stage", style="cyan")
    plan.add_column("Jobs", justify="right")
    plan.add_column("Detail")
    systems, test_sets = len(config.systems), len(config.test_sets)
    budget = config.budget
    detail = (
        f"{budget['epochs']} epoch(s) each"
        if budget["mode"] == "fixed_epochs"
        else "; ".join(f"{key} capped at {config.max_steps_for(key)} steps" for key in config.models)
    )
    plan.add_row("finetune", str(len(config.finetune_systems)),
                 f"{len(config.models)} model(s) x {len(config.volumes)} volume(s), "
                 f"budget {budget['mode']}: {detail}")
    plan.add_row("generate", str(systems * test_sets), f"{systems} systems x {test_sets} test sets")
    plan.add_row("score+report", str(2 * test_sets), "single-process per test set (XCOMET + MetricX)")
    plan.add_row("GPU pool", str(len(config.gpus)), f"physical ids {config.gpus}")
    console.print(plan)

    data = Table(title="Data", header_style="bold cyan", border_style="green")
    data.add_column("Artefact", style="cyan")
    data.add_column("Path")
    data.add_row("corpus", str(config.resolve(config.corpus["csv_path"])))
    data.add_row("in-domain test", str(config.in_domain_test_path))
    data.add_row("train pool", str(config.train_pool_csv))
    for volume in config.volumes:
        data.add_row(f"train {volume_label(volume)}", str(config.subset_split_paths(volume)["train"]))
    for test_set in config.test_sets:
        data.add_row(f"test set {test_set['id']}", str(config.test_set_path(test_set)))
    console.print(data)


def main() -> None:
    args = parse_args()
    config = load_sweep_config(args.config)
    setup_logging(config.raw, run_name="finetune_benchmark")
    logger.info("Sweep config: [bold]%s[/bold]", config.path)
    logger.info(
        "Run [bold]%s[/bold] (budget %s) -> [bold]%s[/bold]",
        config.run_id, config.budget["mode"], config.output_dir,
    )

    if args.stage == "plan":
        show_plan(config)
        return
    if args.stage == "preflight":
        failures = [check for check in preflight.run(config, "all") if check.status == preflight.FAIL]
        if failures:
            raise SystemExit(f"{len(failures)} preflight check(s) failed.")
        logger.info("Preflight clean.")
        return
    if args.stage == "assets":
        # Run this on the ONLINE staging machine; it writes configs for
        # scripts/fetch_offline_assets.py and prints the command to run.
        evaluate.write_staging_configs(config)
        return

    stages = STAGES if args.stage == "all" else (args.stage,)
    systems = [config.system(system_id) for system_id in args.systems] if args.systems else None
    for stage in stages:
        logger.info("=== stage: [bold magenta]%s[/bold magenta] ===", stage)
        if stage == "data":
            data_prep.run(config, force=args.force)
        elif stage == "finetune":
            finetune.run(config, force=args.force, systems=systems)
        elif stage == "evaluate":
            evaluate.run(config, force=args.force, test_set_ids=args.test_sets)
        elif stage == "report":
            report.run(config)


if __name__ == "__main__":
    main()
