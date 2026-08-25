"""Sweep configuration: loading, validation, and the derived per-cell configs.

The sweep owns no training or scoring logic. It owns the *identity* of every
cell in the matrix (model x data volume x test set) and the paths that cell
reads and writes, and it renders the derived config files that the repository's
own scripts consume. Keeping that in one module is what makes a cell resumable:
the same config always names the same files.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SWEEP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SWEEP_ROOT.parents[1]

BASE_TRAINING_CONFIG = PROJECT_ROOT / "config.yaml"
BASE_TESTSET_CONFIG = PROJECT_ROOT / "testset_config.yaml"
BASE_BENCHMARK_CONFIG = PROJECT_ROOT / "benchmark_config.yaml"

CAUSAL_LORA = "causal_lora"
SEQ2SEQ_LORA = "seq2seq_lora"
MODEL_KINDS = (CAUSAL_LORA, SEQ2SEQ_LORA)

_SLUG_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789-_.")


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` over a copy of `base`. Lists are replaced."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def volume_label(rows: int) -> str:
    """Compact, filesystem- and slug-safe label: 5000 -> '5k', 1500 -> '1500'."""
    if rows >= 1000 and rows % 1000 == 0:
        return f"{rows // 1000}k"
    return str(rows)


def _slug(value: str, label: str) -> str:
    if not value or any(char not in _SLUG_CHARS for char in value):
        raise ValueError(
            f"{label} must be lowercase letters, digits, '-', '_' or '.' (got {value!r}); "
            "it becomes a candidate id and a directory name."
        )
    return value


@dataclass(frozen=True)
class System:
    """One evaluated system: a base model, or that model at one data volume."""

    id: str
    label: str
    model_key: str
    model: dict[str, Any]
    volume: int | None  # None is the untuned base model.

    @property
    def volume_label(self) -> str:
        return "base" if self.volume is None else volume_label(self.volume)

    @property
    def kind(self) -> str:
        return self.model["kind"]

    @property
    def is_base(self) -> bool:
        return self.volume is None


@dataclass(frozen=True)
class SweepConfig:
    path: Path
    raw: dict[str, Any]

    # ---- sections -----------------------------------------------------
    @property
    def sweep(self) -> dict[str, Any]:
        return self.raw["sweep"]

    @property
    def corpus(self) -> dict[str, Any]:
        return self.raw["corpus"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def report(self) -> dict[str, Any]:
        return self.raw.get("report", {})

    @property
    def gpus(self) -> list[int]:
        return [int(value) for value in self.sweep["gpus"]]

    @property
    def volumes(self) -> list[int]:
        return [int(value) for value in self.sweep["volumes"]]

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        return {
            key: value
            for key, value in self.raw["models"].items()
            if value.get("enabled", True)
        }

    @property
    def test_sets(self) -> list[dict[str, Any]]:
        return [item for item in self.raw["test_sets"] if item.get("enabled", True)]

    # ---- paths --------------------------------------------------------
    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.sweep["output_dir"])

    @property
    def work_dir(self) -> Path:
        return self.resolve(self.data["work_dir"])

    @property
    def jobs_dir(self) -> Path:
        return self.output_dir / "jobs"

    @property
    def finetune_dir(self) -> Path:
        return self.output_dir / "finetune"

    @property
    def evaluation_dir(self) -> Path:
        return self.output_dir / "evaluation"

    @property
    def report_dir(self) -> Path:
        return self.output_dir / "report"

    @property
    def derived_config_dir(self) -> Path:
        return self.output_dir / "derived_configs"

    # Data-stage artefacts.
    @property
    def corpus_jsonl(self) -> Path:
        """The corpus in the SFT JSONL shape train.py and split_dataset.py read."""
        return self.work_dir / "corpus.jsonl"

    @property
    def testset_build_dir(self) -> Path:
        return self.work_dir / "in_domain_testset"

    @property
    def train_pool_csv(self) -> Path:
        return self.testset_build_dir / "train_pool.csv"

    @property
    def in_domain_test_path(self) -> Path:
        """The 500-row in-domain evaluation set (CSV, benchmark dataset shape)."""
        configured = self.data["in_domain_test"].get("existing_path")
        if self.data["in_domain_test"]["mode"] == "existing":
            return self.resolve(configured)
        return self.testset_build_dir / "test.csv"

    def subset_dir(self, volume: int) -> Path:
        return self.work_dir / "subsets" / volume_label(volume)

    def subset_split_paths(self, volume: int) -> dict[str, Path]:
        directory = self.subset_dir(volume)
        return {
            "train": directory / "train.jsonl",
            "validation": directory / "validation.jsonl",
            "manifest": directory / "split_manifest.json",
        }

    def test_set_path(self, test_set: dict[str, Any]) -> Path:
        if test_set["path"] == "auto":
            if test_set["id"] != "in_domain":
                raise ValueError(
                    f"test set {test_set['id']!r} uses path: auto, which only the "
                    "'in_domain' set (produced by the data stage) may do."
                )
            return self.in_domain_test_path
        return self.resolve(test_set["path"])

    # ---- systems ------------------------------------------------------
    @property
    def systems(self) -> list[System]:
        """Every evaluated system, base models first, then ascending volume."""
        systems: list[System] = []
        for key, model in self.models.items():
            label = model.get("label", key)
            systems.append(System(f"{key}-base", f"{label} — base", key, model, None))
            for volume in self.volumes:
                suffix = volume_label(volume)
                systems.append(
                    System(f"{key}-{suffix}", f"{label} — {suffix} rows", key, model, volume)
                )
        return systems

    @property
    def finetune_systems(self) -> list[System]:
        return [system for system in self.systems if not system.is_base]

    def system(self, system_id: str) -> System:
        for system in self.systems:
            if system.id == system_id:
                return system
        raise KeyError(f"Unknown system id {system_id!r}")

    def finetune_output_dir(self, system: System) -> Path:
        return self.finetune_dir / system.id

    def adapter_path(self, system: System) -> Path:
        """Where the trained adapter lands, per trainer convention.

        train.py writes <output_dir>/<sft_final_subdir>; the NLLB trainer in this
        directory writes <output_dir>/adapter.
        """
        directory = self.finetune_output_dir(system)
        if system.kind == CAUSAL_LORA:
            base = load_yaml(BASE_TRAINING_CONFIG)
            merged = deep_merge(base, system.model.get("overrides", {}))
            return directory / merged["model"]["sft_final_subdir"]
        return directory / "adapter"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path


def load_sweep_config(path: str | Path) -> SweepConfig:
    config = SweepConfig(Path(path).resolve(), load_yaml(path))
    _validate(config)
    return config


def _validate(config: SweepConfig) -> None:
    for section in ("sweep", "corpus", "data", "models", "test_sets", "evaluation"):
        if not config.raw.get(section):
            raise ValueError(f"sweep config is missing the {section!r} section")
    if not config.gpus:
        raise ValueError("sweep.gpus must list at least one physical GPU id")
    if len(set(config.gpus)) != len(config.gpus):
        raise ValueError(f"sweep.gpus contains duplicates: {config.gpus}")
    if not config.volumes or any(value <= 0 for value in config.volumes):
        raise ValueError("sweep.volumes must be a non-empty list of positive row counts")
    if sorted(config.volumes) != config.volumes:
        raise ValueError("sweep.volumes must be ascending; nested subsets depend on it")
    if int(config.sweep.get("epochs", 0)) <= 0:
        raise ValueError("sweep.epochs must be a positive integer")

    if not config.models:
        raise ValueError("no model in models: is enabled")
    for key, model in config.models.items():
        _slug(key, "models key")
        if model.get("kind") not in MODEL_KINDS:
            raise ValueError(f"models.{key}.kind must be one of {list(MODEL_KINDS)}")
        if not model.get("base_model_id"):
            raise ValueError(f"models.{key}.base_model_id is required")
        if not (model.get("evaluation") or {}).get("runner"):
            raise ValueError(f"models.{key}.evaluation.runner is required")
        if model["kind"] == SEQ2SEQ_LORA:
            for field in ("source_lang_token", "target_lang_token"):
                if not model.get(field):
                    raise ValueError(
                        f"models.{key}.{field} is required for a {SEQ2SEQ_LORA} model "
                        "(NLLB tags are model-specific, not ISO-639-1 codes)"
                    )

    test_ids = [_slug(str(item.get("id", "")), "test_sets[].id") for item in config.test_sets]
    if not test_ids:
        raise ValueError("no test set in test_sets: is enabled")
    if len(set(test_ids)) != len(test_ids):
        raise ValueError(f"test set ids must be unique: {test_ids}")
    for test_set in config.test_sets:
        if not test_set.get("path"):
            raise ValueError(f"test set {test_set['id']} needs a path (or 'auto')")
        columns = test_set.get("columns") or {}
        for field in ("id", "source", "reference"):
            if not columns.get(field):
                raise ValueError(f"test set {test_set['id']} needs columns.{field}")

    in_domain = config.data.get("in_domain_test") or {}
    if in_domain.get("mode") not in {"build", "existing"}:
        raise ValueError("data.in_domain_test.mode must be 'build' or 'existing'")
    if in_domain["mode"] == "existing" and not in_domain.get("existing_path"):
        raise ValueError("data.in_domain_test.existing_path is required in 'existing' mode")
    if in_domain["mode"] == "build" and int(in_domain.get("size", 0)) <= 0:
        raise ValueError("data.in_domain_test.size must be positive in 'build' mode")

    ratio = float(config.data.get("validation_ratio", 0.0))
    if not 0.0 <= ratio < 1.0:
        raise ValueError("data.validation_ratio must be in [0, 1)")

    # Duplicate ids across systems would silently overwrite candidate outputs.
    ids = [system.id for system in config.systems]
    if len(set(ids)) != len(ids):
        raise ValueError(f"system ids collide: {ids}")
    for system_id in ids:
        _slug(system_id, "system id")
