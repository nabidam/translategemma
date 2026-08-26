"""Sweep configuration: loading, validation, and the derived per-cell configs.

The sweep owns no training or scoring logic. It owns the *identity* of every
cell in the matrix (model x data volume x test set) and the paths that cell
reads and writes, and it renders the derived config files that the repository's
own scripts consume. Keeping that in one module is what makes a cell resumable:
the same config always names the same files.
"""

from __future__ import annotations

import copy
import math
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
    def excluded_test_domains(self) -> list[str]:
        """Domains withheld from the in-domain test set, hence fully trainable."""
        return [str(value) for value in (self.data["in_domain_test"].get("exclude_domains") or [])]

    @property
    def composition(self) -> dict[str, Any]:
        """Domain composition settings, with every optional key defaulted.

        An absent section means the historical behaviour: mirror the train
        pool's own domain shares.
        """
        raw = self.data.get("composition") or {}
        return {
            "mode": raw.get("mode") or "pool_proportional",
            "domain_shares": {key: float(value) for key, value in (raw.get("domain_shares") or {}).items()},
            "per_volume": {
                int(volume): {key: float(value) for key, value in (shares or {}).items()}
                for volume, shares in (raw.get("per_volume") or {}).items()
            },
            "on_shortfall": raw.get("on_shortfall") or "error",
            "unit": raw.get("unit") or "row",
        }

    def composition_shares(self, volume: int) -> dict[str, float] | None:
        """Target domain shares for one volume, or None when they are implicit.

        None means "derive from the pool" (pool_proportional) or "do not
        stratify at all" (random); the caller distinguishes those by mode.
        """
        composition = self.composition
        if override := composition["per_volume"].get(int(volume)):
            return override
        if composition["mode"] == "domain_shares":
            return composition["domain_shares"]
        return None

    @property
    def report(self) -> dict[str, Any]:
        return self.raw.get("report", {})

    @property
    def budget(self) -> dict[str, Any]:
        """Training-budget settings, with every optional key defaulted.

        Two modes, and the choice decides what a volume-to-volume delta means:

        fixed_epochs  every cell sees its data the same number of times, so a
                      larger volume also gets proportionally more optimizer
                      steps. Answers "what does more data buy me", and mixes
                      data volume with compute.
        fixed_steps   every cell gets the same number of optimizer steps, so the
                      5k cell repeats its data many times and the 100k cell sees
                      part of its own once. Isolates data diversity at equal
                      compute, and risks overfitting the small cells.
        """
        raw = self.sweep.get("budget") or {}
        return {
            "mode": raw.get("mode") or "fixed_epochs",
            # sweep.epochs is the pre-budget spelling; kept working on purpose.
            "epochs": int(raw.get("epochs", self.sweep.get("epochs", 1))),
            "max_steps": raw.get("max_steps", "auto"),
            "evals_per_run": int(raw.get("evals_per_run", 4)),
        }

    @property
    def epochs(self) -> int:
        return self.budget["epochs"]

    def effective_batch_size(self, model_key: str) -> int:
        """Rows (or packed blocks) per optimizer step for one arm.

        Single-GPU cells, so this is the arm's configured effective batch with no
        world-size term.
        """
        model = self.raw["models"][model_key]
        if model["kind"] == CAUSAL_LORA:
            merged = deep_merge(load_yaml(BASE_TRAINING_CONFIG), model.get("overrides") or {})
            training = merged["training"]
            if training.get("effective_batch_size"):
                return int(training["effective_batch_size"])
            return int(training["batch_size"]) * int(training["gradient_accumulation_steps"])
        training = model["training"]
        return int(training["per_device_batch_size"]) * int(training["gradient_accumulation_steps"])

    def max_steps_for(self, model_key: str) -> int | None:
        """Optimizer-step cap for one arm, or None in fixed_epochs mode.

        `auto` derives the cap from the SMALLEST volume at the configured epoch
        count, so every cell gets the compute the smallest cell would have had.
        For the packed TranslateGemma arm the estimate is an upper bound rather
        than exact: packing turns rows into fewer, longer blocks, so the same
        step count covers more epochs than the row arithmetic suggests. Set an
        explicit integer (or a per-arm mapping) when the budget must be exact.
        """
        budget = self.budget
        if budget["mode"] != "fixed_steps":
            return None
        configured = budget["max_steps"]
        if isinstance(configured, dict):
            if model_key not in configured:
                raise ValueError(
                    f"sweep.budget.max_steps has no entry for model {model_key!r}; "
                    f"it names {sorted(configured)}"
                )
            return int(configured[model_key])
        if isinstance(configured, int) and not isinstance(configured, bool):
            return int(configured)
        train_rows = min(self.volumes) * (1.0 - float(self.data.get("validation_ratio", 0.0)))
        steps = math.ceil(train_rows / self.effective_batch_size(model_key) * budget["epochs"])
        return max(1, steps)

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
    def run_id(self) -> str:
        """Directory name separating one budget from another under output_dir.

        The two budget modes produce different adapters and different scores from
        the same data. Without this they would share job paths, and a fixed_steps
        run would silently reuse the fixed_epochs run's completed cells. The data
        stage stays outside this namespace on purpose: both modes read the same
        subsets, which is what makes them comparable.
        """
        if configured := self.sweep.get("run_id"):
            return _slug(str(configured), "sweep.run_id")
        budget = self.budget
        if budget["mode"] == "fixed_epochs":
            return f"fixed_epochs_e{budget['epochs']}"
        steps = budget["max_steps"]
        suffix = steps if isinstance(steps, str) else ("per_model" if isinstance(steps, dict) else steps)
        return f"fixed_steps_{suffix}"

    @property
    def output_dir(self) -> Path:
        """Root for this budget's artefacts. `base_output_dir` is shared."""
        return self.base_output_dir / self.run_id

    @property
    def base_output_dir(self) -> Path:
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


BASE_CONFIG_PURPOSE = {
    BASE_TRAINING_CONFIG: "the TranslateGemma training defaults every cell's config is merged over",
    BASE_BENCHMARK_CONFIG: "the metric, statistics and report defaults every evaluation is merged over",
    BASE_TESTSET_CONFIG: "the test-set builder defaults the data stage is merged over",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        # These are repository files, not generated ones. Missing means the
        # checkout on this host is incomplete -- which on an air-gapped machine
        # usually means the source archive predates the file.
        if purpose := BASE_CONFIG_PURPOSE.get(path):
            raise FileNotFoundError(
                f"{path} is missing. It is a tracked repository file and holds {purpose}. "
                f"Restore it with `git checkout -- {path.name}`, or copy it from the repository."
            )
        raise FileNotFoundError(f"{path} does not exist")
    with path.open(encoding="utf-8") as handle:
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
    budget = config.budget
    if budget["mode"] not in BUDGET_MODES:
        raise ValueError(f"sweep.budget.mode must be one of {list(BUDGET_MODES)}")
    if budget["epochs"] <= 0:
        raise ValueError("sweep.budget.epochs must be a positive integer")
    if budget["evals_per_run"] <= 0:
        raise ValueError("sweep.budget.evals_per_run must be a positive integer")
    max_steps = budget["max_steps"]
    if isinstance(max_steps, dict):
        if unknown := sorted(set(max_steps) - set(config.raw["models"])):
            raise ValueError(f"sweep.budget.max_steps names unknown models: {unknown}")
        for key, value in max_steps.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"sweep.budget.max_steps[{key}] must be a positive integer")
    elif max_steps != "auto" and not (isinstance(max_steps, int) and not isinstance(max_steps, bool) and max_steps > 0):
        raise ValueError("sweep.budget.max_steps must be 'auto', a positive integer, or a per-model mapping")

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
    excluded = in_domain.get("exclude_domains")
    if excluded is not None and (
        not isinstance(excluded, list) or any(not isinstance(value, str) or not value.strip() for value in excluded)
    ):
        raise ValueError("data.in_domain_test.exclude_domains must be a list of non-empty domain names")

    ratio = float(config.data.get("validation_ratio", 0.0))
    if not 0.0 <= ratio < 1.0:
        raise ValueError("data.validation_ratio must be in [0, 1)")

    _validate_composition(config)

    # Duplicate ids across systems would silently overwrite candidate outputs.
    ids = [system.id for system in config.systems]
    if len(set(ids)) != len(ids):
        raise ValueError(f"system ids collide: {ids}")
    for system_id in ids:
        _slug(system_id, "system id")


BUDGET_MODES = ("fixed_epochs", "fixed_steps")
COMPOSITION_MODES = ("pool_proportional", "random", "domain_shares")


def _validate_shares(shares: dict[str, Any], label: str) -> None:
    if not shares:
        raise ValueError(f"{label} must name at least one domain")
    for domain, share in shares.items():
        if not isinstance(share, (int, float)) or isinstance(share, bool) or share < 0:
            raise ValueError(f"{label}[{domain!r}] must be a non-negative number (got {share!r})")
    total = float(sum(shares.values()))
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{label} must sum to 1.0 (got {total:.6f}): {shares}")


def _validate_composition(config: SweepConfig) -> None:
    composition = config.composition
    if composition["mode"] not in COMPOSITION_MODES:
        raise ValueError(f"data.composition.mode must be one of {list(COMPOSITION_MODES)}")
    if composition["on_shortfall"] not in {"error", "redistribute"}:
        raise ValueError("data.composition.on_shortfall must be 'error' or 'redistribute'")
    if composition["unit"] not in {"row", "document"}:
        raise ValueError("data.composition.unit must be 'row' or 'document'")
    if composition["mode"] == "domain_shares":
        _validate_shares(composition["domain_shares"], "data.composition.domain_shares")
    for volume, shares in composition["per_volume"].items():
        if int(volume) not in config.volumes:
            raise ValueError(
                f"data.composition.per_volume has an entry for {volume}, which is not in "
                f"sweep.volumes {config.volumes}"
            )
        _validate_shares(shares, f"data.composition.per_volume[{volume}]")
