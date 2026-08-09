from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BenchmarkConfig, stable_hash
from .generation import generate_candidate
from .io import candidate_dir, candidate_output_path, file_sha256, import_candidate, load_candidate_output, load_dataset
from .metrics import pairwise_comparisons, score_candidates, slice_summary, summarize_scores
from .report import write_reports


def validate(config: BenchmarkConfig) -> dict[str, Any]:
    dataset, manifest = load_dataset(config)
    return {"dataset": manifest, "candidate_ids": [item["id"] for item in config.candidates], "rows": len(dataset)}


def _should_reuse(config: BenchmarkConfig, candidate: dict[str, Any], dataset_manifest: dict[str, Any], force: bool) -> bool:
    output = candidate_output_path(config, candidate["id"])
    manifest_path = candidate_dir(config, candidate["id"]) / "manifest.json"
    if force or not output.exists():
        return False
    if not manifest_path.exists():
        raise ValueError(f"Existing output for {candidate['id']} has no manifest; use --force to replace it.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_dataset = manifest.get("dataset", {})
    dataset_identity = ("sha256", "selected_ids_sha256", "rows")
    if any(stored_dataset.get(key) != dataset_manifest.get(key) for key in dataset_identity):
        raise ValueError(f"Existing output for {candidate['id']} belongs to a different dataset; use --force.")
    if manifest.get("candidate_config_sha256") != stable_hash(candidate):
        raise ValueError(f"Candidate {candidate['id']} configuration changed; use --force to regenerate/reimport.")
    return True


def collect(config: BenchmarkConfig, requested: list[str] | None = None, kind: str | None = None, force: bool = False) -> list[Path]:
    dataset, dataset_manifest = load_dataset(config)
    outputs: list[Path] = []
    for candidate in config.selected_candidates(requested, kind):
        if _should_reuse(config, candidate, dataset_manifest, force):
            outputs.append(candidate_output_path(config, candidate["id"]))
            continue
        if candidate["type"] == "imported":
            outputs.append(import_candidate(config, candidate, dataset, dataset_manifest))
        else:
            outputs.append(generate_candidate(config, candidate, dataset, dataset_manifest))
    return outputs


def score(config: BenchmarkConfig, requested: list[str] | None = None) -> dict[str, Path]:
    dataset, dataset_manifest = load_dataset(config)
    candidates = config.selected_candidates(requested)
    frames: list[pd.DataFrame] = []
    for candidate in candidates:
        frame = load_candidate_output(config, candidate["id"], dataset)
        frame.insert(0, "candidate_id", candidate["id"])
        frame.insert(1, "candidate_label", candidate.get("label", candidate["id"]))
        frame.insert(2, "candidate_family", candidate.get("family", candidate.get("runner", "external")))
        frame.insert(3, "candidate_size", candidate.get("size"))
        frames.append(frame)
    scored = score_candidates(frames, config.raw.get("metrics", {}))
    summary = summarize_scores(scored)
    statistics = config.raw.get("statistics", {})
    pairwise = pairwise_comparisons(
        scored, samples=int(statistics.get("bootstrap_samples", 2000)), seed=int(statistics.get("seed", 42))
    )
    slices = slice_summary(scored, config.raw.get("report", {}).get("slices", ["domain"]))
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "scores": output / "scores.csv",
        "summary": output / "system_summary.csv",
        "pairwise": output / "pairwise_comparisons.csv",
        "slices": output / "slice_summary.csv",
    }
    scored.to_csv(paths["scores"], index=False)
    summary.to_csv(paths["summary"], index=False)
    pairwise.to_csv(paths["pairwise"], index=False)
    slices.to_csv(paths["slices"], index=False)
    outputs = scored.pivot(index="example_id", columns="candidate_id", values="translation").add_prefix("translation__")
    dataset.set_index("example_id").join(outputs).reset_index().to_csv(output / "all_model_outputs.csv", index=False)
    paths["all_outputs"] = output / "all_model_outputs.csv"
    score_manifest = {
        "dataset": dataset_manifest,
        "config_path": str(config.path),
        "config_sha256": file_sha256(config.path),
        "resolved_config": config.raw,
        "candidates": {
            candidate["id"]: {
                "config_sha256": stable_hash(candidate),
                "output_sha256": file_sha256(candidate_output_path(config, candidate["id"])),
            }
            for candidate in candidates
        },
        "metrics": config.raw.get("metrics", {}),
        "statistics": statistics,
    }
    (output / "score_manifest.json").write_text(json.dumps(score_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["manifest"] = output / "score_manifest.json"
    return paths


def report(config: BenchmarkConfig) -> tuple[Path, Path]:
    _, dataset_manifest = load_dataset(config)
    output = config.output_dir
    score_manifest_path = output / "score_manifest.json"
    required = [
        output / name
        for name in ["scores.csv", "system_summary.csv", "pairwise_comparisons.csv", "slice_summary.csv", "score_manifest.json"]
    ]
    if missing := [str(path) for path in required if not path.exists()]:
        raise FileNotFoundError(f"Score first; missing artifacts: {missing}")
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    identity_keys = ("sha256", "selected_ids_sha256", "rows")
    if any(score_manifest.get("dataset", {}).get(key) != dataset_manifest.get(key) for key in identity_keys):
        raise ValueError("The evaluation dataset changed after scoring; run the score command again.")
    scored, summary, pairwise, slices = [pd.read_csv(path) for path in required[:4]]
    return write_reports(config, dataset_manifest, scored, summary, pairwise, slices)
