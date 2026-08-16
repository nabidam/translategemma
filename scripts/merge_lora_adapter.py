#!/usr/bin/env python
"""Merge a LoRA adapter into the base TranslateGemma model and export an immutable checkpoint.

This script executes on the fine-tune machine. It produces a standalone,
offline-loadable model directory containing:
  1. Base model weights merged with the fine-tuned adapter (BF16, safetensors).
  2. Complete tokenizer and processor assets.
  3. Generation configuration with explicit stop token IDs ([1, 106]).
  4. merge_manifest.json with provenance metadata and SHA256 checksums.
  5. SHA256SUMS for artifact integrity verification on the serving host.

Usage:
  uv run python scripts/merge_lora_adapter.py \
      --base-model google/translategemma-12b-it \
      --adapter checkpoints/sft-translategemma-12b-it \
      --output-dir exports/translategemma-12b-it-merged-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)

# Shared generation & stop-token resolution contract
from model_loading import load_generation_safe_model_config, resolve_dtype
from prompting import CHAT_TURN_END_TOKEN, resolve_stop_token_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("merge_lora_adapter")

PROMPT_CONTRACT_VERSION = "2026-08-10"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model ID or local directory path (e.g. google/translategemma-12b-it).",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter directory containing adapter_config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Destination directory for the merged model export.",
    )
    parser.add_argument(
        "--release-id",
        type=str,
        default=None,
        help="Identifier for this release. Defaults to 'tg-merged-<timestamp>'.",
    )
    parser.add_argument(
        "--base-revision",
        type=str,
        default=None,
        help="Base model revision/commit hash for provenance tracking.",
    )
    parser.add_argument(
        "--adapter-revision",
        type=str,
        default=None,
        help="Adapter git commit or version tag for provenance tracking.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Torch data type for loading and saving (default: bfloat16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to place model on during merge ('auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="5GB",
        help="Maximum shard size for safetensors weights (default: 5GB).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--allow-base-mismatch",
        action="store_true",
        help="Allow merge even if adapter's configured base model name differs from --base-model.",
    )
    return parser.parse_args(argv)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hex digest of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_package_versions() -> Dict[str, str]:
    """Capture runtime package versions for audit trail."""
    versions = {}
    for pkg in ("torch", "transformers", "peft", "accelerate"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not-installed"
    return versions


def normalize_model_identifier(ident: str) -> str:
    """Normalize model string or local path for comparison."""
    p = Path(ident)
    if p.exists():
        return p.resolve().as_posix()
    return ident.strip().rstrip("/")


def validate_adapter_compatibility(
    base_model_id: str,
    adapter_path: Path,
    allow_mismatch: bool = False,
) -> Tuple[PeftConfig, bool]:
    """Ensure adapter exists and is configured for the given base model."""
    adapter_config_file = adapter_path / "adapter_config.json"
    if not adapter_config_file.is_file():
        raise FileNotFoundError(
            f"adapter_config.json not found in adapter directory: {adapter_path}"
        )

    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    logger.info("Loaded adapter config: %s", peft_config)

    configured_base = getattr(peft_config, "base_model_name_or_path", None)
    base_matched = False
    if configured_base:
        norm_conf = normalize_model_identifier(configured_base)
        norm_req = normalize_model_identifier(base_model_id)

        # Match exact string, normalized path, or matching trailing repo name
        if (
            norm_conf == norm_req
            or norm_conf.endswith(norm_req)
            or norm_req.endswith(norm_conf)
            or Path(norm_conf).name == Path(norm_req).name
        ):
            base_matched = True
            logger.info("Adapter base model verified: %s matches %s", configured_base, base_model_id)
        else:
            msg = (
                f"Adapter configured base model ({configured_base!r}) does not match "
                f"requested base model ({base_model_id!r})."
            )
            if not allow_mismatch:
                raise ValueError(f"{msg} Pass --allow-base-mismatch to force merge.")
            logger.warning("%s Proceeding because --allow-base-mismatch is enabled.", msg)
    else:
        logger.warning("Adapter config does not specify base_model_name_or_path.")

    return peft_config, base_matched


def build_deterministic_generation_config(
    model_config: Any,
    processor: Any,
    base_model_id: str,
) -> GenerationConfig:
    """Construct an explicit, warning-free generation configuration with verified stop IDs."""
    tokenizer = processor.tokenizer
    gen_config = GenerationConfig.from_model_config(model_config)
    gen_config.do_sample = False
    gen_config.temperature = 1.0
    gen_config.top_p = 1.0
    gen_config.top_k = 50

    if gen_config.bos_token_id is None:
        gen_config.bos_token_id = tokenizer.bos_token_id

    # Union all required stop tokens: EOS (1) and chat turn end (106)
    resolved_stop_ids = resolve_stop_token_ids(
        tokenizer, gen_config, base_model_id=base_model_id
    )
    gen_config.eos_token_id = resolved_stop_ids

    if gen_config.pad_token_id is None:
        gen_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    gen_config._from_model_config = False
    return gen_config


def merge_and_export(
    base_model_id: str,
    adapter_dir: str,
    output_dir: str,
    dtype_name: str = "bfloat16",
    device: str = "auto",
    release_id: Optional[str] = None,
    base_revision: Optional[str] = None,
    adapter_revision: Optional[str] = None,
    max_shard_size: str = "5GB",
    force: bool = False,
    allow_base_mismatch: bool = False,
    command_args: Optional[Dict[str, Any]] = None,
) -> Path:
    """Execute the merge, validation, and atomic directory export with safe backup rollback."""
    out_path = Path(output_dir).resolve()
    adapter_path = Path(adapter_dir).resolve()

    if out_path.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {out_path}. Pass --force to overwrite."
            )
        logger.warning("Output directory %s exists and --force was provided.", out_path)

    if release_id is None:
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        release_id = f"tg-merged-{timestamp_str}"

    # 1. Validate adapter compatibility
    _, base_matched = validate_adapter_compatibility(
        base_model_id,
        adapter_path,
        allow_mismatch=allow_base_mismatch,
    )

    # 2. Resolve torch dtype
    torch_dtype = resolve_dtype(dtype_name)
    logger.info("Using dtype: %s (%s)", dtype_name, torch_dtype)

    # 3. Load processor & tokenizer
    logger.info("Loading processor from %s...", base_model_id)
    processor = AutoProcessor.from_pretrained(base_model_id, fix_markdown=False)
    tokenizer = processor.tokenizer

    # Verify stop tokens at extraction time
    stop_token_ids = resolve_stop_token_ids(tokenizer, base_model_id=base_model_id)
    stop_tokens = [tokenizer.convert_ids_to_tokens(tid) for tid in stop_token_ids]
    logger.info("Resolved stop token IDs: %s (tokens: %s)", stop_token_ids, stop_tokens)

    # Ensure 106 (<end_of_turn>) is explicitly present
    turn_end_id = tokenizer.convert_tokens_to_ids(CHAT_TURN_END_TOKEN)
    if turn_end_id not in stop_token_ids:
        raise ValueError(
            f"Expected <end_of_turn> ({turn_end_id}) to be in stop tokens {stop_token_ids}."
        )

    # 4. Load base model configuration and weights
    logger.info("Loading model config from %s...", base_model_id)
    model_config = load_generation_safe_model_config(base_model_id)

    device_map = device if device in ("auto", "cpu", "cuda") else "auto"
    if device == "auto" and not torch.cuda.is_available():
        device_map = "cpu"

    logger.info("Loading base model %s (device_map=%s)...", base_model_id, device_map)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=model_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    # 5. Attach adapter via PEFT
    logger.info("Attaching LoRA adapter from %s...", adapter_path)
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        torch_dtype=torch_dtype,
    )

    # 6. Merge weights into base
    logger.info("Merging LoRA weights into base model (merge_and_unload)...")
    start_merge = time.time()
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()
    logger.info("Merge completed in %.2f seconds.", time.time() - start_merge)

    # 7. Build generation config
    gen_config = build_deterministic_generation_config(
        model_config=merged_model.config,
        processor=processor,
        base_model_id=base_model_id,
    )

    # 8. Write to atomic temporary directory
    parent_dir = out_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".tmp_merge_{out_path.name}_", dir=parent_dir))
    logger.info("Staging merged artifacts in temporary directory: %s", tmp_dir)

    try:
        # Save model weights in safetensors
        logger.info("Saving merged weights (safe_serialization=True, max_shard_size=%s)...", max_shard_size)
        merged_model.save_pretrained(
            str(tmp_dir),
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )

        # Save processor & tokenizer
        logger.info("Saving processor and tokenizer files...")
        processor.save_pretrained(str(tmp_dir))

        # Save generation configuration
        logger.info("Saving generation_config.json...")
        gen_config.save_pretrained(str(tmp_dir))

        # Build inventory and checksums
        logger.info("Computing artifact checksums and building manifest...")
        file_inventory = []
        checksum_lines = []

        for file_path in sorted(tmp_dir.iterdir()):
            if file_path.is_file():
                rel_name = file_path.name
                size_bytes = file_path.stat().st_size
                sha256_hex = compute_file_sha256(file_path)
                file_inventory.append({
                    "path": rel_name,
                    "size_bytes": size_bytes,
                    "sha256": sha256_hex,
                })
                checksum_lines.append(f"{sha256_hex}  {rel_name}")

        # Write SHA256SUMS
        sha256sums_path = tmp_dir / "SHA256SUMS"
        sha256sums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        sha256sums_hash = compute_file_sha256(sha256sums_path)
        file_inventory.append({
            "path": "SHA256SUMS",
            "size_bytes": sha256sums_path.stat().st_size,
            "sha256": sha256sums_hash,
        })

        # Write merge_manifest.json
        manifest = {
            "release_id": release_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_model": base_model_id,
            "base_revision": base_revision,
            "adapter": str(adapter_path),
            "adapter_revision": adapter_revision,
            "base_match_verified": base_matched,
            "dtype": dtype_name,
            "stop_token_ids": stop_token_ids,
            "stop_tokens": stop_tokens,
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "package_versions": get_package_versions(),
            "command_args": command_args or {},
            "file_inventory": file_inventory,
        }

        manifest_path = tmp_dir / "merge_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Atomic promotion with rollback preservation
        backup_dir: Optional[Path] = None
        if out_path.exists():
            backup_dir = parent_dir / f".backup_{out_path.name}_{os.getpid()}_{int(time.time())}"
            logger.info("Moving existing release to temporary backup: %s", backup_dir)
            out_path.rename(backup_dir)

        try:
            tmp_dir.rename(out_path)
            logger.info("Successfully exported merged checkpoint to: %s", out_path)
            if backup_dir and backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
        except Exception:
            if backup_dir and backup_dir.exists():
                logger.error("Promotion failed; restoring previous release from backup...")
                if out_path.exists():
                    shutil.rmtree(out_path, ignore_errors=True)
                backup_dir.rename(out_path)
            raise

    except Exception:
        logger.exception("Merge export failed. Cleaning up temporary directory %s...", tmp_dir)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return out_path


def main() -> int:
    args = parse_args()
    command_args = vars(args).copy()
    try:
        merge_and_export(
            base_model_id=args.base_model,
            adapter_dir=args.adapter,
            output_dir=args.output_dir,
            dtype_name=args.dtype,
            device=args.device,
            release_id=args.release_id,
            base_revision=args.base_revision,
            adapter_revision=args.adapter_revision,
            max_shard_size=args.max_shard_size,
            force=args.force,
            allow_base_mismatch=args.allow_base_mismatch,
            command_args=command_args,
        )
        return 0
    except Exception as e:
        logger.error("Failed to merge and export LoRA adapter: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
