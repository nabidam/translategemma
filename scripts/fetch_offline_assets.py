"""Stage every Hugging Face artefact the pipeline needs into a portable tree.

Run this on the ONLINE machine. The resulting directory is mounted at /models
inside the container with HF_HOME pointing at it, so no code change is needed to
run fully offline.

    uv run --no-project --with huggingface_hub --with pyyaml \
        python scripts/fetch_offline_assets.py \
        --benchmark-config benchmark_config.yaml --dest offline_assets/models

Repository ids are read from config.yaml, benchmark_config.yaml, and
testset_config.yaml rather than hardcoded, because they are run choices. Two of
them are indirect and easy to miss, so they are resolved here rather than left
to fail at inference time:

  * MetricX checkpoints contain no tokenizer, so `metricx_tokenizer_id` (mT5)
    must be staged separately.
  * COMET's load_from_checkpoint() builds an XLM-R encoder and calls
    XLMRobertaTokenizerFast.from_pretrained(<hparams.pretrained_model>). The
    encoder *weights* are not fetched (load_pretrained_weights=False), but its
    tokenizer and config are. That repository id is read out of the COMET
    checkpoint's hparams.yaml after download.

google/translategemma-12b-it is manually gated: request access on huggingface.co,
wait for approval, then export HF_TOKEN=hf_... before running this.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

# Duplicate serialisations of weights already present as safetensors.
IGNORE_PATTERNS = ["*.h5", "*.msgpack", "*.onnx", "*.onnx_data", "*.tflite", "*.ot"]

# Enough to rebuild a tokenizer (and its config) without pulling the checkpoint.
# Used for repositories whose weights are never loaded: the mT5 tokenizer source
# (~15 GB of weights skipped) and COMET's XLM-R encoder.
TOKENIZER_ONLY_PATTERNS = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "merges.txt",
]


def _is_remote_repo_id(value: object) -> bool:
    """Distinguish Hugging Face repo IDs from local model/adapter paths."""
    if not isinstance(value, str) or "/" not in value:
        return False
    return not value.startswith(("/", "./", "../")) and not Path(value).exists()


RepoSpec = tuple[str, list[str] | None, str | None]


def _deduplicate_repos(repos: list[RepoSpec]) -> list[RepoSpec]:
    """Deduplicate while promoting tokenizer-only entries to full snapshots."""
    seen: dict[tuple[str, str | None], list[str] | None] = {}
    for repo_id, patterns, revision in repos:
        key = (repo_id, revision)
        if key not in seen or patterns is None:
            seen[key] = patterns
    return [(repo_id, patterns, revision) for (repo_id, revision), patterns in seen.items()]


def collect_repo_ids(
    config_path: Path,
    testset_config_path: Path | None,
    benchmark_config_path: Path | None = None,
) -> tuple[list[RepoSpec], set[str]]:
    """Return repositories to stage and the COMET checkpoint IDs among them."""
    config = yaml.safe_load(config_path.read_text())
    evaluation = config["evaluation"]
    comet_repos: set[str] = set()

    repos: list[RepoSpec] = [(config["model"]["base_model_id"], None, None)]

    if evaluation.get("metricx_enabled"):
        repos.append((evaluation["metricx_model_id"], None, None))
        repos.append((evaluation["metricx_tokenizer_id"], TOKENIZER_ONLY_PATTERNS, None))

    if evaluation.get("comet_enabled"):
        # unbabel-comet >= 2.0 resolves download_model() through huggingface_hub
        # with cache_dir=None, so the checkpoint lands in HF_HOME like any other
        # repository. Its encoder dependency is added later, in main().
        repos.append((evaluation["comet_model_id"], None, None))
        comet_repos.add(evaluation["comet_model_id"])

    if benchmark_config_path is not None and benchmark_config_path.exists():
        benchmark_config = yaml.safe_load(benchmark_config_path.read_text()) or {}
        for candidate in benchmark_config.get("candidates", []):
            if not candidate.get("enabled", True) or candidate.get("type") != "generated":
                continue
            for key in ("model", "processor", "tokenizer"):
                repo_id = candidate.get(key)
                if _is_remote_repo_id(repo_id):
                    repos.append((repo_id, None, candidate.get("revision")))
            # Local adapter directories travel with run outputs. A Hub-hosted
            # adapter is staged only when declared explicitly, avoiding a
            # relative local path being mistaken for a repository ID.
            adapter_repo = candidate.get("adapter_repo")
            if _is_remote_repo_id(adapter_repo):
                repos.append((adapter_repo, None, candidate.get("adapter_revision")))

        benchmark_metrics = benchmark_config.get("metrics", {})
        metricx = benchmark_metrics.get("metricx", {})
        if metricx.get("enabled"):
            metricx_model = metricx.get("model", "google/metricx-24-hybrid-large-v2p6")
            metricx_tokenizer = metricx.get("tokenizer", "google/mt5-xl")
            if _is_remote_repo_id(metricx_model):
                repos.append((metricx_model, None, None))
            if _is_remote_repo_id(metricx_tokenizer):
                repos.append((metricx_tokenizer, TOKENIZER_ONLY_PATTERNS, None))
        comet = benchmark_metrics.get("comet", {})
        if comet.get("enabled"):
            comet_repo = comet.get("model", "Unbabel/wmt22-comet-da")
            repos.append((comet_repo, None, None))
            comet_repos.add(comet_repo)

    if testset_config_path is not None and testset_config_path.exists():
        testset_config = yaml.safe_load(testset_config_path.read_text())
        embedding_model = testset_config.get("embeddings", {}).get("model")
        if embedding_model:
            repos.append((embedding_model, None, None))

    return _deduplicate_repos(repos), comet_repos


def comet_encoder_repo(snapshot_path: Path) -> str | None:
    """Read the encoder repository id out of a COMET checkpoint's hparams.yaml."""
    hparams_path = snapshot_path / "hparams.yaml"
    if not hparams_path.exists():
        return None
    hparams = yaml.safe_load(hparams_path.read_text())
    return hparams.get("pretrained_model")


def stage(repo_id: str, hub_cache: Path, allow_patterns: list[str] | None, revision: str | None = None) -> Path:
    scope = "tokenizer + config only" if allow_patterns else "full"
    revision_label = f" @ {revision}" if revision else ""
    print(f"==> {repo_id}{revision_label} ({scope})", flush=True)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=hub_cache,
            allow_patterns=allow_patterns,
            ignore_patterns=None if allow_patterns else IGNORE_PATTERNS,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--testset-config", type=Path, default=Path("testset_config.yaml"))
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=Path("benchmark_config.yaml"),
        help="Multi-model benchmark config; enabled generated candidates and evaluators are staged.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("offline_assets/models"),
        help="Mounted at /models and used as HF_HOME; snapshots go to <dest>/hub.",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Extra repository id to stage in full. May be repeated.",
    )
    args = parser.parse_args()

    hub_cache = args.dest / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)

    queue, comet_repos = collect_repo_ids(args.config, args.testset_config, args.benchmark_config)
    queue += [(repo_id, None, None) for repo_id in args.repo]
    queue = _deduplicate_repos(queue)

    print(f"Queue: {queue}")

    staged: list[str] = []
    failures: list[tuple[str, str]] = []

    while queue:
        repo_id, allow_patterns, revision = queue.pop(0)
        try:
            snapshot_path = stage(repo_id, hub_cache, allow_patterns, revision)
            staged.append(repo_id)
        except Exception as error:  # noqa: BLE001 - report all, fail at the end
            failures.append((repo_id, str(error)))
            print(f"    FAILED: {error}", file=sys.stderr, flush=True)
            continue

        if repo_id in comet_repos:
            encoder_repo = comet_encoder_repo(snapshot_path)
            if encoder_repo is None:
                failures.append((repo_id, "hparams.yaml missing or has no pretrained_model"))
            elif encoder_repo not in staged:
                # Queued with the exact id COMET will request. The hub cache is
                # keyed by that literal string, so staging a canonical alias
                # (FacebookAI/xlm-roberta-large) would miss when offline.
                queue.append((encoder_repo, TOKENIZER_ONLY_PATTERNS, None))

    print(f"\nStaged {len(staged)} repositories in {hub_cache}")
    for repo_id in staged:
        print(f"  - {repo_id}")
    if failures:
        print("\nFailed:")
        for repo_id, error in failures:
            print(f"  - {repo_id}: {error}")
        print("\nGated repositories need HF_TOKEN and an approved access request.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
