from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BenchmarkConfig, stable_hash


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    raise ValueError(f"Unsupported table format {path.suffix!r}: {path}")


def load_dataset(config: BenchmarkConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = config.benchmark["dataset"]
    path = config.resolve_path(spec["path"])
    frame = read_table(path)
    columns = spec.get("columns", {})
    rename = {
        columns.get("id", "id"): "example_id",
        columns.get("source", "source_text"): "source",
        columns.get("reference", "target_text"): "reference",
    }
    optional = {source: target for target, source in {
        "domain": columns.get("domain"),
        "document_id": columns.get("document_id"),
    }.items() if source}
    frame = frame.rename(columns={**rename, **optional})
    required = {"example_id", "source", "reference"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Evaluation dataset is missing columns: {sorted(missing)}")
    if frame["example_id"].isna().any() or frame["example_id"].astype(str).duplicated().any():
        raise ValueError("Evaluation example IDs must be non-null and unique.")
    if frame[["source", "reference"]].isna().any().any():
        raise ValueError("Evaluation source and reference values must be non-null.")
    frame["example_id"] = frame["example_id"].astype(str)
    frame["source"] = frame["source"].astype(str)
    frame["reference"] = frame["reference"].astype(str)
    if maximum := spec.get("max_examples"):
        frame = frame.head(int(maximum)).copy()
    manifest = {
        "path": str(path),
        "sha256": file_sha256(path),
        "rows": len(frame),
        "columns": columns,
        "source_lang": spec.get("source_lang"),
        "target_lang": spec.get("target_lang"),
        "selected_ids_sha256": stable_hash(frame["example_id"].tolist()),
    }
    return frame.reset_index(drop=True), manifest


def candidate_dir(config: BenchmarkConfig, candidate_id: str) -> Path:
    return config.output_dir / "candidates" / candidate_id


def candidate_output_path(config: BenchmarkConfig, candidate_id: str) -> Path:
    return candidate_dir(config, candidate_id) / "translations.csv"


def write_candidate_output(
    config: BenchmarkConfig,
    candidate: dict[str, Any],
    frame: pd.DataFrame,
    dataset_manifest: dict[str, Any],
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    directory = candidate_dir(config, candidate["id"])
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "translations.csv"
    frame.to_csv(output, index=False)
    manifest = {
        "candidate": candidate,
        "candidate_config_sha256": stable_hash(candidate),
        "dataset": dataset_manifest,
        "output_sha256": file_sha256(output),
        "rows": len(frame),
        **(extra_manifest or {}),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return output


def load_candidate_output(config: BenchmarkConfig, candidate_id: str, dataset: pd.DataFrame) -> pd.DataFrame:
    path = candidate_output_path(config, candidate_id)
    if not path.exists():
        raise FileNotFoundError(f"No collected output for {candidate_id}: {path}")
    frame = pd.read_csv(path, dtype={"example_id": str})
    required = {"example_id", "translation"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Candidate {candidate_id} output is missing columns: {sorted(missing)}")
    if frame["example_id"].duplicated().any():
        raise ValueError(f"Candidate {candidate_id} has duplicate example IDs.")
    frame["translation"] = frame["translation"].fillna("").astype(str)
    expected, actual = set(dataset["example_id"]), set(frame["example_id"])
    if expected != actual:
        raise ValueError(
            f"Candidate {candidate_id} ID mismatch: {len(expected - actual)} missing, "
            f"{len(actual - expected)} unexpected."
        )
    return dataset[[column for column in dataset.columns]].merge(frame, on="example_id", how="left", validate="one_to_one")


def import_candidate(
    config: BenchmarkConfig,
    candidate: dict[str, Any],
    dataset: pd.DataFrame,
    dataset_manifest: dict[str, Any],
) -> Path:
    source_path = config.resolve_path(candidate["path"])
    imported = read_table(source_path)
    columns = candidate.get("columns", {})
    imported = imported.rename(columns={
        columns.get("id", "example_id"): "example_id",
        columns.get("translation", "translation"): "translation",
    })
    if not {"example_id", "translation"}.issubset(imported.columns):
        raise ValueError(f"Imported candidate {candidate['id']} needs ID and translation columns.")
    imported["example_id"] = imported["example_id"].astype(str)
    if imported["example_id"].duplicated().any():
        raise ValueError(f"Imported candidate {candidate['id']} has duplicate IDs.")
    expected, actual = set(dataset["example_id"]), set(imported["example_id"])
    missing_ids = expected - actual
    unexpected_ids = actual - expected
    allow_extra = bool(candidate.get("allow_extra_ids", False))
    if missing_ids or (unexpected_ids and not allow_extra):
        raise ValueError(
            f"Imported candidate {candidate['id']} ID mismatch: {len(missing_ids)} missing, "
            f"{len(unexpected_ids)} unexpected."
        )
    output = dataset[["example_id"]].merge(
        imported.loc[imported["example_id"].isin(expected), ["example_id", "translation"]],
        on="example_id", validate="one_to_one"
    )
    output["translation"] = output["translation"].fillna("").astype(str)
    output["status"] = output["translation"].map(lambda value: "ok" if value.strip() else "empty")
    return write_candidate_output(config, candidate, output, dataset_manifest, {
        "import_source": str(source_path), "import_source_sha256": file_sha256(source_path)
    })
