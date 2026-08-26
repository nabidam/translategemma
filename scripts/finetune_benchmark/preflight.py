"""Offline preflight: verify everything a run needs before it queues a single job.

A corrupt or half-transferred checkpoint shard fails at model-load time, which on
this matrix is after the scheduler has already spent GPU minutes and, worse,
after other cells have started. Every check here is local and cheap — file
presence, size, and the few magic bytes that distinguish a real weight file from
a truncated download or an LFS pointer — so the whole sweep's inputs are checked
in seconds, with no network.

It deliberately does NOT load weights: the point is to be fast enough to run
automatically at the start of every stage.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rich.table import Table

from .config import CAUSAL_LORA, SweepConfig
from .scheduler import console, logger, training_benchmark

# Below this, a "weight file" is a pointer, a stub, or a truncated transfer.
MIN_WEIGHT_BYTES = 1 << 20
# Torch pickle archives are ZIPs; pre-zip checkpoints start with a pickle opcode.
TORCH_MAGIC = (b"PK\x03\x04", b"\x80")
WEIGHT_INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
SINGLE_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def _snapshot_dir(repo_id: str) -> Path:
    """Locate a cached repository without touching the network."""
    from huggingface_hub import snapshot_download

    if Path(repo_id).is_dir():
        return Path(repo_id)
    return Path(snapshot_download(repo_id, local_files_only=True))


def _weight_files(snapshot: Path) -> tuple[list[Path], int | None]:
    """Weight files a checkpoint declares, plus the total_size its index claims."""
    for name in WEIGHT_INDEX_FILES:
        index_path = snapshot / name
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            files = sorted({snapshot / value for value in index.get("weight_map", {}).values()})
            return files, (index.get("metadata") or {}).get("total_size")
    for name in SINGLE_WEIGHT_FILES:
        if (snapshot / name).is_file():
            return [snapshot / name], None
    # Sharded without an index, or a format this project does not load.
    found = sorted(snapshot.glob("*.safetensors")) + sorted(snapshot.glob("*.bin"))
    return found, None


def _check_weight_file(path: Path) -> str | None:
    """Return a problem description, or None when the file looks loadable."""
    if not path.exists():
        return "missing"
    resolved = path.resolve()  # The hub cache fills snapshots with symlinks to blobs.
    if not resolved.exists():
        return f"dangling symlink -> {resolved}"
    size = resolved.stat().st_size
    if size < MIN_WEIGHT_BYTES:
        head = resolved.open("rb").read(64)
        if head.startswith(b"version https://git-lfs"):
            return f"git-lfs pointer, not the weights ({size} bytes)"
        return f"only {size} bytes — truncated transfer or stub"
    with resolved.open("rb") as handle:
        header = handle.read(8)
    if path.suffix == ".safetensors":
        header_length = int.from_bytes(header, "little")
        if not 0 < header_length < size:
            return f"safetensors header length {header_length} is impossible for a {size}-byte file"
    elif path.suffix == ".bin" and not header.startswith(TORCH_MAGIC):
        # Exactly the 2026-08-26 failure: transformers unpickles the shard and
        # reports an unrelated-sounding "load from a TF 2.0 checkpoint?" error.
        return f"not a torch archive (starts with {header[:4]!r}) — re-stage this shard"
    return None


def check_checkpoint(label: str, repo_id: str) -> list[Check]:
    try:
        snapshot = _snapshot_dir(repo_id)
    except Exception as error:  # noqa: BLE001 - any resolution failure is the same fail
        return [Check(label, FAIL, f"{repo_id} is not staged locally: {type(error).__name__}: {error}")]

    checks: list[Check] = []
    if incomplete := sorted(snapshot.parent.parent.rglob("*.incomplete")):
        checks.append(
            Check(label, FAIL, f"unfinished download(s) in the cache: {[path.name for path in incomplete[:3]]}")
        )
    files, total_size = _weight_files(snapshot)
    if not files:
        return checks + [Check(label, FAIL, f"no weight files under {snapshot}")]
    problems = [(path.name, problem) for path in files if (problem := _check_weight_file(path))]
    if problems:
        return checks + [
            Check(label, FAIL, "; ".join(f"{name}: {problem}" for name, problem in problems[:4]))
        ]
    actual = sum(path.resolve().stat().st_size for path in files)
    if total_size and actual < 0.9 * total_size:
        return checks + [
            Check(label, FAIL, f"weights total {_human(actual)} but the index declares "
                               f"{_human(total_size)} — incomplete transfer")
        ]
    return checks + [Check(label, OK, f"{len(files)} weight file(s), {_human(actual)}, {snapshot}")]


def check_comet(label: str, repo_id: str) -> list[Check]:
    """COMET ships a Lightning checkpoint plus a separate encoder repository."""
    try:
        snapshot = _snapshot_dir(repo_id)
    except Exception as error:  # noqa: BLE001
        return [Check(label, FAIL, f"{repo_id} is not staged locally: {type(error).__name__}: {error}")]
    checkpoints = sorted(snapshot.rglob("*.ckpt"))
    if not checkpoints:
        return [Check(label, FAIL, f"no .ckpt under {snapshot}")]
    if problem := _check_weight_file(checkpoints[0]):
        return [Check(label, FAIL, f"{checkpoints[0].name}: {problem}")]
    checks = [Check(label, OK, f"{checkpoints[0].name}, {_human(checkpoints[0].resolve().stat().st_size)}")]
    # load_from_checkpoint builds a tokenizer from the encoder repo named in
    # hparams.yaml. Its absence surfaces as an AttributeError inside
    # transformers, which is why it is checked here (OFFLINE_DEPLOYMENT §6.4).
    hparams = next(iter(sorted(snapshot.rglob("hparams.yaml"))), None)
    if hparams is None:
        checks.append(Check(f"{label} encoder", WARN, "no hparams.yaml; cannot verify the encoder repository"))
        return checks
    import yaml

    encoder = (yaml.safe_load(hparams.read_text(encoding="utf-8")) or {}).get("pretrained_model")
    if not encoder:
        checks.append(Check(f"{label} encoder", WARN, "hparams.yaml names no pretrained_model"))
        return checks
    try:
        _snapshot_dir(str(encoder))
        checks.append(Check(f"{label} encoder", OK, str(encoder)))
    except Exception as error:  # noqa: BLE001
        checks.append(
            Check(f"{label} encoder", FAIL,
                  f"{encoder} is not staged; COMET loads its tokenizer from there ({type(error).__name__})")
        )
    return checks


def check_tokenizer(label: str, repo_id: str) -> list[Check]:
    try:
        snapshot = _snapshot_dir(repo_id)
    except Exception as error:  # noqa: BLE001
        return [Check(label, FAIL, f"{repo_id} is not staged locally: {type(error).__name__}: {error}")]
    wanted = ("tokenizer.json", "spiece.model", "sentencepiece.bpe.model", "tokenizer_config.json", "vocab.json")
    present = [name for name in wanted if (snapshot / name).is_file()]
    if not present:
        return [Check(label, FAIL, f"no tokenizer files under {snapshot}")]
    return [Check(label, OK, ", ".join(present))]


def check_base_configs() -> list[Check]:
    """The repository templates every derived config is merged over.

    Checked because the failure mode is late and confusing: the evaluate stage
    reads benchmark_config.yaml only when it renders its first candidate list,
    which is after the whole finetune stage has run.
    """
    from .config import BASE_CONFIG_PURPOSE

    return [
        Check(f"template {path.name}", OK if path.is_file() else FAIL,
              str(path) if path.is_file() else f"missing — holds {purpose}")
        for path, purpose in BASE_CONFIG_PURPOSE.items()
    ]


def check_data(config: SweepConfig, stage: str) -> list[Check]:
    checks: list[Check] = []
    corpus = config.resolve(config.corpus["csv_path"])
    checks.append(
        Check("corpus", OK if corpus.is_file() else FAIL,
              str(corpus) if corpus.is_file() else f"missing: {corpus}")
    )
    for test_set in config.test_sets:
        source = config.test_set_path(test_set)
        label = f"test set {test_set['id']}"
        if not source.exists():
            checks.append(Check(label, FAIL if stage != "data" else WARN, f"missing: {source}"))
            continue
        try:
            frame = pd.read_csv(source, nrows=5, dtype=str) if source.suffix in {".csv", ".tsv"} else None
        except Exception as error:  # noqa: BLE001
            checks.append(Check(label, FAIL, f"unreadable: {type(error).__name__}: {error}"))
            continue
        columns = test_set["columns"]
        missing = [
            columns[field] for field in ("source", "reference")
            if frame is not None and columns[field] not in frame.columns
        ]
        checks.append(
            Check(label, FAIL if missing else OK, f"missing column(s) {missing}" if missing else str(source))
        )
    if stage in {"finetune", "evaluate"}:
        for volume in config.volumes:
            path = config.subset_split_paths(volume)["train"]
            checks.append(
                Check(f"subset {volume}", OK if path.is_file() else FAIL,
                      str(path) if path.is_file() else f"missing: {path} — run the data stage")
            )
    if stage == "evaluate":
        for system in config.finetune_systems:
            adapter = config.adapter_path(system)
            checks.append(
                Check(f"adapter {system.id}", OK if adapter.is_dir() else WARN,
                      str(adapter) if adapter.is_dir() else f"absent: {adapter} — that cell will be skipped")
            )
    return checks


def check_gpus(config: SweepConfig) -> list[Check]:
    rows, error = training_benchmark.query_nvidia_telemetry()
    if error or not rows:
        return [Check("gpus", WARN, f"cannot query nvidia-smi: {error or 'no output'}")]
    visible = {int(row["index"]) for row in rows if row.get("index") is not None}
    if missing := sorted(set(config.gpus) - visible):
        return [Check("gpus", FAIL, f"sweep.gpus wants {missing}; only {sorted(visible)} are visible. "
                                    "Start the container with GPUS=all.")]
    free = min(
        (row["memory.total"] or 0) - (row["memory.used"] or 0)
        for row in rows if int(row["index"]) in set(config.gpus)
    )
    status = OK if free > 20_000 else WARN
    return [Check("gpus", status, f"{sorted(config.gpus)} visible, least-free device has {free / 1024:.0f} GiB")]


def check_disk(config: SweepConfig) -> list[Check]:
    checks = []
    for label, path in (("disk output", config.base_output_dir), ("disk data", config.work_dir)):
        target = path if path.exists() else path.parent
        try:
            free_gb = shutil.disk_usage(target).free / 1e9
        except OSError as error:
            checks.append(Check(label, WARN, f"{target}: {error}"))
            continue
        checks.append(
            Check(label, OK if free_gb > 100 else WARN, f"{free_gb:.0f} GB free at {target}")
        )
    return checks


def run(config: SweepConfig, stage: str = "all") -> list[Check]:
    """Every check for `stage`, printed as a table. Never raises on a FAIL."""
    checks: list[Check] = [
        *check_base_configs(), *check_gpus(config), *check_disk(config), *check_data(config, stage),
    ]
    for key, model in config.models.items():
        checks.extend(check_checkpoint(f"model {key}", model["base_model_id"]))
    metrics = (config.evaluation.get("overrides") or {}).get("metrics") or {}
    if (comet := metrics.get("comet") or {}).get("enabled"):
        checks.extend(check_comet("comet", comet["model"]))
    if (metricx := metrics.get("metricx") or {}).get("enabled"):
        checks.extend(check_checkpoint("metricx", metricx["model"]))
        checks.extend(check_tokenizer("metricx tokenizer", metricx["tokenizer"]))
    if stage in {"data", "all"} and config.data["in_domain_test"]["mode"] == "build":
        overrides = config.data["in_domain_test"].get("overrides") or {}
        embeddings = (overrides.get("embeddings") or {}).get("model")
        if embeddings is None:
            from .config import BASE_TESTSET_CONFIG, load_yaml

            embeddings = (load_yaml(BASE_TESTSET_CONFIG).get("embeddings") or {}).get("model")
        if embeddings:
            checks.extend(check_checkpoint("testset embeddings", embeddings))

    table = Table(title=f"Preflight ({stage})", title_style="bold yellow",
                  header_style="bold yellow", border_style="green")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail", style="dim", overflow="fold")
    styles = {OK: "[green]ok[/green]", WARN: "[yellow]warn[/yellow]", FAIL: "[red]FAIL[/red]"}
    for check in checks:
        table.add_row(check.name, styles[check.status], check.detail)
    console.print(table)
    return checks


def enforce(config: SweepConfig, stage: str) -> None:
    """Run the checks for `stage` and refuse to start when one fails.

    Called at the top of the finetune and evaluate stages: a bad checkpoint costs
    seconds to detect here and a queue of half-run cells to detect later. Set
    sweep.preflight: false to skip.
    """
    if not config.sweep.get("preflight", True):
        logger.warning("Preflight disabled (sweep.preflight: false).")
        return
    failures = [check for check in run(config, stage) if check.status == FAIL]
    if failures:
        detail = "; ".join(f"{check.name}: {check.detail}" for check in failures)
        raise RuntimeError(
            f"Preflight failed for the {stage} stage: {detail}. Fix these, or set sweep.preflight: false "
            "to run anyway."
        )
