"""Stage every Hugging Face artefact the pipeline needs into a portable tree.

Run this on the ONLINE machine. The resulting directory is mounted at /models
inside the container with HF_HOME pointing at it, so no code change is needed to
run fully offline.

    uv run --no-project --with huggingface_hub --with pyyaml \
        python scripts/fetch_offline_assets.py --dest offline_assets/models

Repository ids are read from config.yaml / testset_config.yaml rather than
hardcoded, because they are run choices. Two of them are indirect and easy to
miss, so they are resolved here rather than left to fail at inference time:

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


def collect_repo_ids(
    config_path: Path, testset_config_path: Path | None
) -> list[tuple[str, list[str] | None]]:
    """Return (repo_id, allow_patterns) pairs to stage, in download order."""
    config = yaml.safe_load(config_path.read_text())
    evaluation = config["evaluation"]

    repos: list[tuple[str, list[str] | None]] = [(config["model"]["base_model_id"], None)]

    if evaluation.get("metricx_enabled"):
        repos.append((evaluation["metricx_model_id"], None))
        repos.append((evaluation["metricx_tokenizer_id"], TOKENIZER_ONLY_PATTERNS))

    if evaluation.get("comet_enabled"):
        # unbabel-comet >= 2.0 resolves download_model() through huggingface_hub
        # with cache_dir=None, so the checkpoint lands in HF_HOME like any other
        # repository. Its encoder dependency is added later, in main().
        repos.append((evaluation["comet_model_id"], None))

    if testset_config_path is not None and testset_config_path.exists():
        testset_config = yaml.safe_load(testset_config_path.read_text())
        embedding_model = testset_config.get("embeddings", {}).get("model")
        if embedding_model:
            repos.append((embedding_model, None))

    seen: dict[str, list[str] | None] = {}
    for repo_id, patterns in repos:
        seen.setdefault(repo_id, patterns)
    return list(seen.items())


def comet_encoder_repo(snapshot_path: Path) -> str | None:
    """Read the encoder repository id out of a COMET checkpoint's hparams.yaml."""
    hparams_path = snapshot_path / "hparams.yaml"
    if not hparams_path.exists():
        return None
    hparams = yaml.safe_load(hparams_path.read_text())
    return hparams.get("pretrained_model")


def stage(repo_id: str, hub_cache: Path, allow_patterns: list[str] | None) -> Path:
    scope = "tokenizer + config only" if allow_patterns else "full"
    print(f"==> {repo_id} ({scope})", flush=True)
    return Path(
        snapshot_download(
            repo_id=repo_id,
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

    config = yaml.safe_load(args.config.read_text())
    hub_cache = args.dest / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)

    queue = collect_repo_ids(args.config, args.testset_config)
    queue += [(repo_id, None) for repo_id in args.repo]

    print(f"Queue: {queue}")

    staged: list[str] = []
    failures: list[tuple[str, str]] = []

    while queue:
        repo_id, allow_patterns = queue.pop(0)
        try:
            snapshot_path = stage(repo_id, hub_cache, allow_patterns)
            staged.append(repo_id)
        except Exception as error:  # noqa: BLE001 - report all, fail at the end
            failures.append((repo_id, str(error)))
            print(f"    FAILED: {error}", file=sys.stderr, flush=True)
            continue

        if repo_id == config["evaluation"].get("comet_model_id"):
            encoder_repo = comet_encoder_repo(snapshot_path)
            if encoder_repo is None:
                failures.append((repo_id, "hparams.yaml missing or has no pretrained_model"))
            elif encoder_repo not in staged:
                # Queued with the exact id COMET will request. The hub cache is
                # keyed by that literal string, so staging a canonical alias
                # (FacebookAI/xlm-roberta-large) would miss when offline.
                queue.append((encoder_repo, TOKENIZER_ONLY_PATTERNS))

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
