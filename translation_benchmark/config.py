from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _slug(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if not value or any(char not in allowed for char in value):
        raise ValueError(f"Invalid identifier {value!r}; use letters, numbers, '-', '_' or '.'.")
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BenchmarkConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def benchmark(self) -> dict[str, Any]:
        return self.raw["benchmark"]

    @property
    def output_dir(self) -> Path:
        value = Path(self.benchmark.get("output_dir", "benchmark_output"))
        return value if value.is_absolute() else self.root / value

    @property
    def candidates(self) -> list[dict[str, Any]]:
        return self.raw["candidates"]

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def selected_candidates(self, requested: list[str] | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        requested_set = set(requested or [])
        known = {candidate["id"] for candidate in self.candidates}
        if missing := requested_set - known:
            raise ValueError(f"Unknown candidate IDs: {sorted(missing)}")
        return [
            candidate
            for candidate in self.candidates
            if (candidate.get("enabled", True) if not requested_set else candidate["id"] in requested_set)
            and (kind is None or candidate["type"] == kind)
        ]


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw.get("benchmark"), dict):
        raise ValueError("Configuration must contain a 'benchmark' mapping.")
    if not isinstance(raw.get("candidates"), list) or not raw["candidates"]:
        raise ValueError("Configuration must contain a non-empty 'candidates' list.")
    dataset = raw["benchmark"].get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("path"):
        raise ValueError("benchmark.dataset.path is required.")
    ids: list[str] = []
    for candidate in raw["candidates"]:
        if not isinstance(candidate, dict):
            raise ValueError("Every candidate must be a mapping.")
        ids.append(_slug(str(candidate.get("id", ""))))
        if candidate.get("type") not in {"generated", "imported"}:
            raise ValueError(f"Candidate {candidate['id']} type must be 'generated' or 'imported'.")
        if candidate["type"] == "generated" and candidate.get("runner") not in {"translategemma", "nllb"}:
            raise ValueError(f"Candidate {candidate['id']} has unsupported runner {candidate.get('runner')!r}.")
        if candidate["type"] == "generated" and not candidate.get("model"):
            raise ValueError(f"Generated candidate {candidate['id']} requires model.")
        if candidate.get("runner") == "nllb" and not all(candidate.get(key) for key in ("source_lang", "target_lang")):
            raise ValueError(f"NLLB candidate {candidate['id']} requires source_lang and target_lang.")
        if candidate["type"] == "imported" and not candidate.get("path"):
            raise ValueError(f"Imported candidate {candidate['id']} requires path.")
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique.")
    profiles = raw.get("generation_profiles", {})
    for candidate in raw["candidates"]:
        profile = candidate.get("generation_profile")
        if profile and profile not in profiles:
            raise ValueError(f"Candidate {candidate['id']} references unknown generation profile {profile!r}.")
    statistics = raw.get("statistics", {})
    if int(statistics.get("bootstrap_samples", 2000)) <= 0:
        raise ValueError("statistics.bootstrap_samples must be positive.")
    return BenchmarkConfig(config_path, raw)
