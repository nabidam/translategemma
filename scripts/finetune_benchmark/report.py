"""Report stage: one cross-test-set answer with the statistics to defend it.

Reads what the earlier stages produced — per-test-set scores and paired
bootstrap comparisons from translation_benchmark, per-job wall clock and
nvidia-smi telemetry from the scheduler — and turns them into:

  master_summary.csv   every system x test set x metric, plus eval throughput
  training_cost.csv    wall clock, GPU-hours, peak VRAM, energy per fine-tune
  deltas.csv           each system vs its own family's base, with 95% CIs
  marginal_gains.csv   what each step up in data volume actually bought
  <report.filename>    a self-contained HTML report with the conclusion

The deltas are paired bootstrap intervals over the same examples, not a
comparison of two independent means: with 500-2000 segment test sets, a raw
COMET difference of a few thousandths is noise, and only the interval says so.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import SweepConfig, System, volume_label
from .scheduler import PROJECT_ROOT, logger  # noqa: F401  (PROJECT_ROOT sets sys.path)

from translation_benchmark.metrics import METRIC_DIRECTIONS  # noqa: E402
from degeneration import audit_outputs  # noqa: E402

PERCENT_METRICS = frozenset({"empty_output", "source_copy", "hit_max_new_tokens",
                             "number_preservation", "acronym_preservation", "formula_preservation"})


# ---------------------------------------------------------------- loading
def _load_job_results(directory: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for path in sorted(directory.rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Ignoring unreadable job result %s", path)
            continue
        results[payload.get("job_id", str(path))] = payload
    return results


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("Missing %s; the affected tables will be incomplete.", path)
        return None
    return pd.read_csv(path)


def _metric_columns(config: SweepConfig, frame: pd.DataFrame) -> list[str]:
    requested = config.report.get("metrics") or list(METRIC_DIRECTIONS)
    return [metric for metric in requested if metric in frame.columns]


def _budget_label(config: SweepConfig) -> str:
    budget = config.budget
    if budget["mode"] == "fixed_epochs":
        return f"{budget['epochs']} epoch(s) per cell"
    caps = ", ".join(f"{key} {config.max_steps_for(key)}" for key in config.models)
    return f"compute-matched ({caps} optimizer steps)"


def _budget_note(config: SweepConfig) -> str:
    if config.budget["mode"] == "fixed_epochs":
        return (
            f"Every cell trained for {config.epochs} epoch(s), so a larger volume also received "
            "proportionally more optimizer steps. Volume and compute are deliberately not separated; "
            "the training-cost table shows what each cell actually spent."
        )
    return (
        "Every cell was capped at the same number of optimizer steps "
        f"({_budget_label(config)}), so compute is held roughly constant and the small volumes revisit "
        "their data many times. A flat curve here means added diversity did not help at equal compute; "
        "it does not mean added data never helps, which is what the fixed_epochs run answers. Overfitting "
        "at the small volumes is expected — read it in the eval_loss column, not as a bug."
    )


# ---------------------------------------------------------------- tables
def build_master_summary(config: SweepConfig, generate_jobs: dict[str, dict]) -> pd.DataFrame:
    """One row per (test set, system): metric means plus generation throughput."""
    rows: list[pd.DataFrame] = []
    for test_set in config.test_sets:
        summary = _read_csv(config.evaluation_dir / test_set["id"] / "system_summary.csv")
        if summary is None:
            continue
        summary = summary.copy()
        summary.insert(0, "test_set", test_set["id"])
        summary.insert(1, "test_set_label", test_set["label"])
        summary["model_key"] = summary["candidate_id"].map(lambda value: value.rsplit("-", 1)[0])
        summary["volume_label"] = summary["candidate_id"].map(lambda value: value.rsplit("-", 1)[1])
        summary["volume"] = summary["volume_label"].map(
            lambda label: 0 if label == "base" else _volume_from_label(config, label)
        )
        durations, energies = [], []
        for candidate_id in summary["candidate_id"]:
            job = generate_jobs.get(f"generate-{test_set['id']}-{candidate_id}", {})
            durations.append(job.get("duration_seconds"))
            energies.append((job.get("hardware") or {}).get("energy_wh"))
        summary["generation_wall_seconds"] = durations
        summary["generation_energy_wh"] = energies
        summary["generation_rows_per_second"] = [
            (examples / duration) if duration else None
            for examples, duration in zip(summary["examples"], durations)
        ]
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    return combined.sort_values(["test_set", "model_key", "volume"]).reset_index(drop=True)


def _volume_from_label(config: SweepConfig, label: str) -> int:
    for volume in config.volumes:
        if volume_label(volume) == label:
            return volume
    return 0


def build_training_cost(config: SweepConfig, finetune_jobs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for system in config.finetune_systems:
        job = finetune_jobs.get(f"finetune-{system.id}")
        if not job:
            continue
        hardware = job.get("hardware") or {}
        duration = job.get("duration_seconds") or 0.0
        metrics = _training_metrics(config, system)
        rows.append(
            {
                "system_id": system.id,
                "system_label": system.label,
                "model_key": system.model_key,
                "volume": system.volume,
                "volume_label": system.volume_label,
                "status": job.get("status"),
                "budget_mode": config.budget["mode"],
                "epochs": config.epochs,
                "max_steps": config.max_steps_for(system.model_key),
                "steps_run": metrics.get("global_step"),
                "train_rows": metrics.get("train_rows"),
                "trainable_parameters": metrics.get("trainable_parameters"),
                "train_loss": metrics.get("train_loss"),
                "eval_loss": metrics.get("eval_loss"),
                "train_wall_seconds": duration,
                "train_wall_hours": duration / 3600,
                # One GPU per cell, so GPU-hours equal wall hours; the column is
                # kept explicit because the sweep runs four cells at once and the
                # total below is GPU-hours, not elapsed time.
                "gpu_hours": duration / 3600,
                "seconds_per_1k_rows": (duration / (metrics["train_rows"] / 1000))
                if metrics.get("train_rows") else None,
                "vram_peak_gib": (hardware.get("vram_peak_mib") or 0) / 1024 or None,
                "gpu_utilization_average_percent": hardware.get("gpu_utilization_average_percent"),
                "power_draw_average_watts": hardware.get("power_draw_average_watts"),
                "energy_wh": hardware.get("energy_wh"),
                "gpu_name": hardware.get("gpu_name"),
            }
        )
    return pd.DataFrame(rows)


def _training_metrics(config: SweepConfig, system: System) -> dict[str, Any]:
    """Read whichever metrics the cell's trainer left behind.

    The two trainers report differently: train_nllb_lora.py writes
    training_metrics.json, while train.py writes run_metadata.json (split sizes,
    no losses) and leaves the losses in Trainer's own trainer_state.json. Both
    are read so the cost table has the same columns for both arms.
    """
    directory = config.finetune_output_dir(system)
    metrics: dict[str, Any] = {"train_rows": system.volume}
    for name in ("run_metadata.json", "training_metrics.json"):
        path = directory / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics.update({key: value for key, value in payload.items() if not isinstance(value, (dict, list))})
        if splits := payload.get("split_sizes"):
            metrics["train_rows"] = splits.get("train", metrics["train_rows"])
    metrics.update(_trainer_state_metrics(directory))
    return metrics


def _trainer_state_metrics(directory: Path) -> dict[str, Any]:
    """Last train loss and best eval loss from the newest Trainer state file."""
    states = sorted(directory.rglob("trainer_state.json"), key=lambda path: path.stat().st_mtime)
    if not states:
        return {}
    try:
        state = json.loads(states[-1].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    history = state.get("log_history") or []
    metrics: dict[str, Any] = {}
    for entry in reversed(history):
        if "loss" in entry and "train_loss" not in metrics:
            metrics["train_loss"] = entry["loss"]
        if "eval_loss" in entry and "eval_loss" not in metrics:
            metrics["eval_loss"] = entry["eval_loss"]
        if len(metrics) == 2:
            break
    if state.get("best_metric") is not None:
        metrics["eval_loss"] = state["best_metric"]
    if state.get("global_step") is not None:
        metrics["global_step"] = state["global_step"]
    return metrics


def _pairwise_lookup(frame: pd.DataFrame) -> dict[tuple[str, str, str], dict]:
    return {
        (row["metric"], row["candidate_a"], row["candidate_b"]): row
        for row in frame.to_dict("records")
    }


def _oriented_delta(lookup: dict, metric: str, reference: str, candidate: str) -> dict[str, Any] | None:
    """Delta of `candidate` minus `reference`, whichever way the pair was stored."""
    if row := lookup.get((metric, reference, candidate)):
        return {
            "delta": row["delta_b_minus_a"], "ci95_low": row["ci95_low"], "ci95_high": row["ci95_high"],
            "win_rate": row["b_win_rate"], "paired_examples": row["paired_examples"],
            "significant_95": bool(row["significant_95"]),
        }
    if row := lookup.get((metric, candidate, reference)):
        return {
            "delta": -row["delta_b_minus_a"], "ci95_low": -row["ci95_high"], "ci95_high": -row["ci95_low"],
            "win_rate": row["a_win_rate"], "paired_examples": row["paired_examples"],
            "significant_95": bool(row["significant_95"]),
        }
    return None


def build_deltas(config: SweepConfig, master: pd.DataFrame) -> pd.DataFrame:
    """Every fine-tuned system against its own family's base model."""
    rows = []
    for test_set in config.test_sets:
        pairwise = _read_csv(config.evaluation_dir / test_set["id"] / "pairwise_comparisons.csv")
        if pairwise is None or pairwise.empty:
            continue
        lookup = _pairwise_lookup(pairwise)
        present = set(master.loc[master["test_set"] == test_set["id"], "candidate_id"])
        metrics = _metric_columns(config, master)
        for model_key in sorted({system.model_key for system in config.systems}):
            base_id = f"{model_key}-base"
            if base_id not in present:
                continue
            for system in config.finetune_systems:
                if system.model_key != model_key or system.id not in present:
                    continue
                for metric in metrics:
                    if not (delta := _oriented_delta(lookup, metric, base_id, system.id)):
                        continue
                    rows.append(
                        {
                            "test_set": test_set["id"], "model_key": model_key, "reference": base_id,
                            "system_id": system.id, "volume": system.volume, "metric": metric,
                            "direction": METRIC_DIRECTIONS.get(metric, "higher"),
                            "improved": (delta["delta"] > 0) == (METRIC_DIRECTIONS.get(metric, "higher") == "higher"),
                            **delta,
                        }
                    )
    return pd.DataFrame(rows)


def build_marginal_gains(config: SweepConfig, master: pd.DataFrame) -> pd.DataFrame:
    """Each step along the volume curve, priced in GPU-hours.

    The interesting question is not "does fine-tuning help" (it does) but "where
    does the curve flatten". A step whose interval spans zero bought nothing
    measurable, whatever its GPU-hours were.
    """
    metric = config.report.get("primary_metric", "comet")
    cost = build_training_cost(config, _load_job_results(config.jobs_dir / "finetune"))
    cost_by_system = cost.set_index("system_id")["gpu_hours"].to_dict() if not cost.empty else {}
    rows = []
    for test_set in config.test_sets:
        pairwise = _read_csv(config.evaluation_dir / test_set["id"] / "pairwise_comparisons.csv")
        if pairwise is None or pairwise.empty:
            continue
        lookup = _pairwise_lookup(pairwise)
        present = set(master.loc[master["test_set"] == test_set["id"], "candidate_id"])
        for model_key in sorted({system.model_key for system in config.systems}):
            ladder = [f"{model_key}-base"] + [
                system.id for system in config.finetune_systems if system.model_key == model_key
            ]
            ladder = [system_id for system_id in ladder if system_id in present]
            for previous, current in zip(ladder, ladder[1:]):
                if not (delta := _oriented_delta(lookup, metric, previous, current)):
                    continue
                added_hours = (cost_by_system.get(current) or 0) - (cost_by_system.get(previous) or 0)
                rows.append(
                    {
                        "test_set": test_set["id"], "model_key": model_key, "metric": metric,
                        "from": previous, "to": current, **delta,
                        "added_gpu_hours": added_hours,
                        "gpu_hours_per_metric_point": (added_hours / delta["delta"])
                        if delta["delta"] and delta["delta"] > 0 else None,
                    }
                )
    return pd.DataFrame(rows)


def build_degeneration(config: SweepConfig) -> pd.DataFrame:
    """Classify decoding failures per system, from the generated text itself.

    translation_benchmark scores corpus averages, and a corpus average cannot
    say "this system stopped translating and filled the token budget" — the
    2026-08-10 adapter posted its best eval_loss while 87% of its output was
    unusable. This audit is the only check in the sweep that can see that, so it
    runs over every candidate's translations.csv regardless of scoring.
    """
    rows = []
    for test_set in config.test_sets:
        dataset_path = config.work_dir / "testsets" / f"{test_set['id']}.csv"
        if not dataset_path.exists():
            continue
        dataset = pd.read_csv(dataset_path, dtype={"id": str})
        references = dataset.set_index("id")["fa"].astype(str)
        candidates_dir = config.evaluation_dir / test_set["id"] / "candidates"
        for path in sorted(candidates_dir.glob("*/translations.csv")):
            frame = pd.read_csv(path, dtype={"example_id": str})
            frame["translation"] = frame["translation"].fillna("").astype(str)
            aligned = frame.join(references.rename("reference"), on="example_id")
            if aligned["reference"].isna().any():
                logger.warning("%s has ids absent from %s; auditing the joined rows only.", path, dataset_path)
                aligned = aligned.dropna(subset=["reference"])
            audit = audit_outputs(aligned["translation"].tolist(), aligned["reference"].tolist())
            rows.append(
                {
                    "test_set": test_set["id"],
                    "candidate_id": path.parent.name,
                    "rows": audit["rows"],
                    "clean_rate": audit["clean_rate"],
                    "failure_rate": audit["failure_rate"],
                    "mean_chars": audit["mean_chars"],
                    "mean_trailing_chars": audit["mean_trailing_chars"],
                    "max_trailing_chars": audit["max_trailing_chars"],
                    **{f"failure_{name}": values["rate"] for name, values in audit["failures"].items()},
                }
            )
    return pd.DataFrame(rows)


def build_conclusion(config: SweepConfig, master: pd.DataFrame, deltas: pd.DataFrame,
                     marginal: pd.DataFrame, degeneration: pd.DataFrame) -> dict:
    """The claims the report is willing to make, each tied to a number."""
    metric = config.report.get("primary_metric", "comet")
    higher_is_better = METRIC_DIRECTIONS.get(metric, "higher") == "higher"
    conclusion: dict[str, Any] = {"primary_metric": metric, "direction": "higher" if higher_is_better else "lower",
                                  "per_test_set": {}, "notes": [], "degeneration_gate": None}
    if master.empty or metric not in master.columns:
        conclusion["notes"].append(f"No {metric} column available; the ranking sections are empty.")
        return conclusion

    for test_set_id, group in master.groupby("test_set"):
        ordered = group.sort_values(metric, ascending=not higher_is_better)
        winner = ordered.iloc[0]
        entry: dict[str, Any] = {
            "winner": winner["candidate_id"],
            "winner_label": winner.get("candidate_label"),
            "winner_score": float(winner[metric]),
            "examples": int(winner["examples"]),
            "ranking": [
                {"candidate_id": row["candidate_id"], metric: float(row[metric])}
                for _, row in ordered.iterrows()
            ],
        }
        if not deltas.empty:
            family_best = {}
            subset = deltas[(deltas["test_set"] == test_set_id) & (deltas["metric"] == metric)]
            for model_key, family in subset.groupby("model_key"):
                improved = family[family["improved"] & family["significant_95"]]
                best = (improved if not improved.empty else family).copy()
                best = best.sort_values("delta", ascending=not higher_is_better).iloc[0]
                family_best[model_key] = {
                    "system_id": best["system_id"],
                    "delta_vs_base": float(best["delta"]),
                    "ci95": [float(best["ci95_low"]), float(best["ci95_high"])],
                    "significant_95": bool(best["significant_95"]),
                    "any_significant_gain": bool(not improved.empty),
                }
            entry["best_per_family"] = family_best
        if not marginal.empty:
            steps = marginal[(marginal["test_set"] == test_set_id)]
            plateau = {}
            for model_key, family in steps.groupby("model_key"):
                # The first step whose interval spans zero is where more data
                # stopped paying, on this test set, at this sample size.
                insignificant = family[~family["significant_95"]]
                plateau[model_key] = None if insignificant.empty else insignificant.iloc[0]["to"]
            entry["plateau_after"] = plateau
        conclusion["per_test_set"][test_set_id] = entry

    threshold = config.report.get("max_degeneration_rate")
    if threshold is not None and not degeneration.empty:
        breached = degeneration[degeneration["failure_rate"] > float(threshold)]
        conclusion["degeneration_gate"] = {
            "threshold": float(threshold),
            "breached": [
                {"test_set": row["test_set"], "candidate_id": row["candidate_id"],
                 "failure_rate": float(row["failure_rate"])}
                for _, row in breached.iterrows()
            ],
        }
        if not breached.empty:
            # A system can top the metric table while a large share of its output
            # is structurally broken. Say so next to the ranking, not in a log.
            conclusion["notes"].append(
                "Decoding audit: "
                + ", ".join(
                    f"{row['candidate_id']} on {row['test_set']} fails {row['failure_rate']:.1%} of rows"
                    for _, row in breached.iterrows()
                )
                + f" (above report.max_degeneration_rate={float(threshold):.1%}). Treat those scores as "
                "unreliable until the decoding failure is fixed."
            )

    conclusion["notes"] += [
        _budget_note(config),
        "Training subsets are nested (each volume is a superset of the smaller ones) and were drawn "
        "from a pool with every test document, and its near-duplicates, already removed.",
        "Deltas are paired bootstrap estimates over identical examples; an interval spanning zero "
        "means the difference is not resolvable at this test-set size.",
        "Energy is integrated from sampled power draw on a shared host; with four concurrent jobs it "
        "is an estimate of the job's own draw, not an isolated measurement.",
        "COMET and MetricX correlations were established on WMT MQM language pairs, which do not include "
        "Persian (docs/EVALUATION_BACKLOG.md). Read these scores as ordinal within one test set: compare "
        "systems on the same rows, never a score on one test set against a score on another, and do not "
        "read an absolute quality threshold into them.",
    ]
    return conclusion


def build_human_review(config: SweepConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blinded side-by-side rows for human judgement, plus the key to unblind.

    Automatic metrics identify patterns; a blinded human read decides whether
    they matter (docs/TRANSLATION_BENCHMARK.md, human evaluation protocol). The
    export carries randomized per-row labels and no scores, so a reviewer cannot
    infer the system from the column order or from a metric.
    """
    settings = config.report.get("human_review") or {}
    if not settings.get("enabled", True):
        return pd.DataFrame(), pd.DataFrame()
    rows_per_test_set = int(settings.get("rows_per_test_set", 40))
    rng = np.random.default_rng(int(settings.get("seed", config.sweep["seed"])))
    review_rows, key_rows = [], []
    for test_set in config.test_sets:
        outputs_path = config.evaluation_dir / test_set["id"] / "all_model_outputs.csv"
        if not outputs_path.exists():
            continue
        frame = pd.read_csv(outputs_path, dtype={"example_id": str})
        columns = [column for column in frame.columns if column.startswith("translation__")]
        if not columns:
            continue
        group_column = "domain" if "domain" in frame.columns else None
        # Stratified by domain when the test set has one, so a rare domain is not
        # missing from the review entirely.
        if group_column:
            groups = [group for _, group in frame.groupby(group_column, sort=True)]
            per_group = max(1, rows_per_test_set // max(len(groups), 1))
            sampled = pd.concat(
                [group.sample(n=min(len(group), per_group), random_state=int(rng.integers(1 << 31)))
                 for group in groups],
                ignore_index=True,
            )
        else:
            sampled = frame.sample(n=min(len(frame), rows_per_test_set),
                                   random_state=int(rng.integers(1 << 31)))
        for _, row in sampled.iterrows():
            order = list(columns)
            rng.shuffle(order)
            for index, column in enumerate(order):
                label = chr(ord("A") + index)
                candidate_id = column[len("translation__"):]
                review_rows.append({
                    "test_set": test_set["id"], "example_id": row["example_id"],
                    "domain": row.get(group_column) if group_column else None,
                    "source": row.get("source"), "reference": row.get("reference"),
                    "system_label": label, "translation": row[column],
                    "adequacy_1_5": "", "fluency_1_5": "", "terminology_1_5": "", "notes": "",
                })
                key_rows.append({"test_set": test_set["id"], "example_id": row["example_id"],
                                 "system_label": label, "candidate_id": candidate_id})
    return pd.DataFrame(review_rows), pd.DataFrame(key_rows)


# ---------------------------------------------------------------- rendering
def _format_value(metric: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if metric in PERCENT_METRICS:
        return f"{100 * float(value):.1f}%"
    if metric in {"sentence_bleu", "sentence_chrf"}:
        return f"{float(value):.2f}"
    return f"{float(value):.4f}"


def _table(frame: pd.DataFrame, float_format: str = "{:,.4f}") -> str:
    if frame is None or frame.empty:
        return "<p class='empty'>No data.</p>"
    return frame.to_html(index=False, escape=True, border=0, float_format=float_format.format, na_rep="—")


def _svg_curve(config: SweepConfig, master: pd.DataFrame, test_set_id: str) -> str:
    """Metric-versus-volume curve per model family, drawn inline.

    Hand-rolled SVG rather than a plotting dependency: the offline image ships
    no chart library, and a curve here is worth more than a fourth table.
    """
    metric = config.report.get("primary_metric", "comet")
    group = master[(master["test_set"] == test_set_id)]
    if group.empty or metric not in group.columns:
        return ""
    higher_is_better = METRIC_DIRECTIONS.get(metric, "higher") == "higher"
    labels = ["base"] + [volume_label(volume) for volume in config.volumes]
    width, height, pad = 720, 300, 52
    values = group[metric].dropna().astype(float)
    if values.empty:
        return ""
    low, high = float(values.min()), float(values.max())
    span = (high - low) or max(abs(high), 1.0) * 0.1
    low, high = low - span * 0.15, high + span * 0.15
    palette = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]

    def position(index: int, value: float) -> tuple[float, float]:
        x = pad + index * (width - 2 * pad) / max(len(labels) - 1, 1)
        y = height - pad - (value - low) / (high - low) * (height - 2 * pad)
        return x, y

    parts = [f"<svg viewBox='0 0 {width} {height}' class='curve' role='img'>"]
    parts.append(
        f"<line x1='{pad}' y1='{height - pad}' x2='{width - pad}' y2='{height - pad}' class='axis'/>"
        f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height - pad}' class='axis'/>"
    )
    for index, label in enumerate(labels):
        x, _ = position(index, low)
        parts.append(f"<text x='{x:.1f}' y='{height - pad + 18}' class='tick' text-anchor='middle'>{label}</text>")
    for fraction in (0.0, 0.5, 1.0):
        value = low + fraction * (high - low)
        _, y = position(0, value)
        parts.append(f"<text x='{pad - 8}' y='{y + 4:.1f}' class='tick' text-anchor='end'>{value:.3f}</text>")
    for color_index, (model_key, family) in enumerate(group.groupby("model_key")):
        by_label = family.set_index("volume_label")[metric].to_dict()
        points = [
            position(index, float(by_label[label]))
            for index, label in enumerate(labels)
            if label in by_label and pd.notna(by_label[label])
        ]
        if not points:
            continue
        color = palette[color_index % len(palette)]
        path = " ".join(f"{'M' if index == 0 else 'L'}{x:.1f},{y:.1f}" for index, (x, y) in enumerate(points))
        parts.append(f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        for x, y in points:
            parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}'/>")
        parts.append(
            f"<text x='{width - pad + 6}' y='{points[-1][1] + 4:.1f}' class='legend' fill='{color}'>"
            f"{html.escape(str(model_key))}</text>"
        )
    arrow = "higher is better" if higher_is_better else "lower is better"
    parts.append(f"<text x='{pad}' y='{pad - 18}' class='legend'>{metric} ({arrow})</text></svg>")
    return "".join(parts)


CSS = """
:root { --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --good:#047857; --bad:#b91c1c; --bg:#ffffff; --panel:#f9fafb; }
* { box-sizing:border-box; }
body { margin:0; padding:2.5rem clamp(1rem,4vw,4rem); background:var(--bg); color:var(--ink);
  font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
h1 { font-size:1.9rem; margin:0 0 .25rem; letter-spacing:-.02em; }
h2 { font-size:1.25rem; margin:2.5rem 0 .75rem; padding-bottom:.4rem; border-bottom:1px solid var(--line); }
h3 { font-size:1rem; margin:1.75rem 0 .5rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
p.lede { color:var(--muted); margin:0 0 1.5rem; }
table { border-collapse:collapse; width:100%; font-size:13.5px; margin:.5rem 0 1rem; }
th,td { padding:.45rem .6rem; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th { background:var(--panel); font-weight:600; text-align:right; position:sticky; top:0; }
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) { text-align:left; }
tbody tr:hover { background:#f3f4f6; }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:8px; }
.cards { display:flex; flex-wrap:wrap; gap:1rem; margin:1rem 0 1.5rem; }
.card { flex:1 1 220px; border:1px solid var(--line); border-radius:10px; padding:.9rem 1rem; background:var(--panel); }
.card .k { font-size:.72rem; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); }
.card .v { font-size:1.35rem; font-weight:650; margin-top:.2rem; }
.card .s { font-size:.8rem; color:var(--muted); }
.good { color:var(--good); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
ul.notes { color:var(--muted); font-size:13.5px; padding-left:1.1rem; }
svg.curve { width:100%; height:auto; background:var(--panel); border:1px solid var(--line); border-radius:8px; }
svg .axis { stroke:#9ca3af; stroke-width:1; } svg .tick { font-size:11px; fill:#6b7280; }
svg .legend { font-size:12px; font-weight:600; }
code { background:var(--panel); padding:.1rem .3rem; border-radius:4px; font-size:12.5px; }
"""


def _conclusion_html(config: SweepConfig, conclusion: dict, deltas: pd.DataFrame) -> str:
    metric = conclusion["primary_metric"]
    parts = ["<h2>Conclusion</h2>"]
    if not conclusion["per_test_set"]:
        parts.append("<p class='empty'>No scored test set yet.</p>")
        return "".join(parts)
    cards = []
    for test_set_id, entry in conclusion["per_test_set"].items():
        cards.append(
            f"<div class='card'><div class='k'>{html.escape(test_set_id)} — best system</div>"
            f"<div class='v'>{html.escape(str(entry['winner']))}</div>"
            f"<div class='s'>{metric} {_format_value(metric, entry['winner_score'])} over {entry['examples']} examples</div></div>"
        )
    parts.append(f"<div class='cards'>{''.join(cards)}</div>")
    for test_set_id, entry in conclusion["per_test_set"].items():
        parts.append(f"<h3>{html.escape(test_set_id)}</h3><ul>")
        for model_key, best in (entry.get("best_per_family") or {}).items():
            marker = "good" if best["significant_95"] else "bad"
            verdict = "significant at 95%" if best["significant_95"] else "not resolvable at 95%"
            parts.append(
                f"<li><b>{html.escape(model_key)}</b>: best cell <code>{html.escape(best['system_id'])}</code>, "
                f"Δ{metric} {best['delta_vs_base']:+.4f} vs its base "
                f"(95% CI {best['ci95'][0]:+.4f} … {best['ci95'][1]:+.4f}, <span class='{marker}'>{verdict}</span>)</li>"
            )
        for model_key, plateau in (entry.get("plateau_after") or {}).items():
            if plateau:
                parts.append(
                    f"<li><b>{html.escape(model_key)}</b>: the volume curve stops paying at "
                    f"<code>{html.escape(str(plateau))}</code> — that step's interval spans zero.</li>"
                )
            else:
                parts.append(
                    f"<li><b>{html.escape(model_key)}</b>: every step up in data volume was still "
                    "significant, so the curve had not flattened at the largest volume tested.</li>"
                )
        parts.append("</ul>")
    parts.append("<h3>Read these numbers with</h3><ul class='notes'>")
    parts.extend(f"<li>{html.escape(note)}</li>" for note in conclusion["notes"])
    parts.append("</ul>")
    return "".join(parts)


def render_html(config: SweepConfig, master: pd.DataFrame, cost: pd.DataFrame, deltas: pd.DataFrame,
                marginal: pd.DataFrame, degeneration: pd.DataFrame, conclusion: dict) -> str:
    metric = conclusion["primary_metric"]
    metrics = _metric_columns(config, master) if not master.empty else []
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = [
        "<h1>Fine-tuning data-volume benchmark</h1>",
        f"<p class='lede'>{html.escape(', '.join(model.get('label', key) for key, model in config.models.items()))}"
        f" · volumes {', '.join(volume_label(volume) for volume in config.volumes)}"
        f" · {len(config.test_sets)} test set(s) · {_budget_label(config)}"
        f" · generated {generated}</p>",
        _conclusion_html(config, conclusion, deltas),
    ]
    sections.append("<h2>Quality by test set</h2>")
    for test_set in config.test_sets:
        group = master[master["test_set"] == test_set["id"]] if not master.empty else pd.DataFrame()
        if group.empty:
            continue
        sections.append(f"<h3>{html.escape(test_set['label'])}</h3>")
        sections.append(_svg_curve(config, master, test_set["id"]))
        columns = ["candidate_id", "candidate_label", "examples", *metrics,
                   "latency_seconds", "output_tokens", "generation_wall_seconds", "generation_rows_per_second"]
        view = group[[column for column in columns if column in group.columns]]
        sections.append(f"<div class='scroll'>{_table(view)}</div>")
    sections.append("<h2>Gain over the untuned base model</h2>")
    if not deltas.empty:
        view = deltas[deltas["metric"] == metric][
            ["test_set", "system_id", "reference", "delta", "ci95_low", "ci95_high",
             "win_rate", "paired_examples", "significant_95"]
        ]
        sections.append(f"<div class='scroll'>{_table(view)}</div>")
        sections.append("<h3>All metrics</h3>")
        sections.append(f"<div class='scroll'>{_table(deltas)}</div>")
    else:
        sections.append("<p class='empty'>No pairwise comparisons available.</p>")
    sections.append("<h2>What each step up in data volume bought</h2>")
    sections.append(f"<div class='scroll'>{_table(marginal)}</div>")
    sections.append("<h2>Decoding audit</h2>")
    sections.append(
        "<p class='lede'>Failure classes measured on the generated text: empty output, unstopped "
        "whitespace, repeated n-grams, leaked boilerplate, length blowup. A high failure rate invalidates "
        "that system's metric scores, however good they look.</p>"
    )
    sections.append(f"<div class='scroll'>{_table(degeneration)}</div>")
    sections.append("<h2>Fine-tuning cost</h2>")
    sections.append(f"<div class='scroll'>{_table(cost, '{:,.3f}')}</div>")
    if not cost.empty:
        total_hours = float(cost["gpu_hours"].sum())
        total_energy = float(pd.to_numeric(cost["energy_wh"], errors="coerce").fillna(0).sum())
        sections.append(
            f"<div class='cards'><div class='card'><div class='k'>Total fine-tune cost</div>"
            f"<div class='v'>{total_hours:,.1f} GPU-hours</div>"
            f"<div class='s'>≈ {total_energy / 1000:,.1f} kWh estimated</div></div></div>"
        )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Fine-tuning data-volume benchmark</title>"
        f"<style>{CSS}</style></head><body>{''.join(sections)}</body></html>"
    )


def run(config: SweepConfig) -> dict[str, Path]:
    finetune_jobs = _load_job_results(config.jobs_dir / "finetune")
    generate_jobs = _load_job_results(config.jobs_dir / "evaluate")
    master = build_master_summary(config, generate_jobs)
    cost = build_training_cost(config, finetune_jobs)
    deltas = build_deltas(config, master) if not master.empty else pd.DataFrame()
    marginal = build_marginal_gains(config, master) if not master.empty else pd.DataFrame()
    degeneration = build_degeneration(config)
    conclusion = build_conclusion(config, master, deltas, marginal, degeneration)

    directory = config.report_dir
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "master_summary": directory / "master_summary.csv",
        "training_cost": directory / "training_cost.csv",
        "deltas": directory / "deltas.csv",
        "marginal_gains": directory / "marginal_gains.csv",
        "degeneration": directory / "degeneration_audit.csv",
        "conclusion": directory / "conclusion.json",
        "html": directory / config.report.get("filename", "finetune_benchmark_report.html"),
    }
    master.to_csv(paths["master_summary"], index=False)
    cost.to_csv(paths["training_cost"], index=False)
    deltas.to_csv(paths["deltas"], index=False)
    marginal.to_csv(paths["marginal_gains"], index=False)
    degeneration.to_csv(paths["degeneration"], index=False)
    review, review_key = build_human_review(config)
    if not review.empty:
        paths["human_review"] = directory / "human_review_blind.csv"
        paths["human_review_key"] = directory / "human_review_key.csv"
        review.to_csv(paths["human_review"], index=False)
        review_key.to_csv(paths["human_review_key"], index=False)
    paths["conclusion"].write_text(json.dumps(conclusion, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["html"].write_text(
        render_html(config, master, cost, deltas, marginal, degeneration, conclusion), encoding="utf-8"
    )
    logger.info("Report written to [bold]%s[/bold]", paths["html"])
    for name, path in paths.items():
        logger.info("  %-16s %s", name, path)
    if master.empty:
        # The artefacts are still written -- an empty report is a legitimate
        # thing to look at -- but "Report written" must not read as success when
        # no test set has scores behind it.
        raise RuntimeError(
            "No scored test set: every table in the report is empty. The evaluate stage's score phase "
            f"has not produced {config.evaluation_dir}/<test_set>/system_summary.csv. Re-run the "
            "evaluate stage; collected translations are reused."
        )
    return paths
