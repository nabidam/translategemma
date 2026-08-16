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
        help="Base model revision/commit hash for model loading and provenance tracking.",
    )
    parser.add_argument(
        "--adapter-revision",
        type=str,
        default=None,
        help="Adapter git commit or version tag for adapter loading and provenance tracking.",
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
        help="Allow exporting to existing destination (retains existing release as backup).",
    )
    parser.add_argument(
        "--allow-base-mismatch",
        action="store_true",
        help="Allow merge even if adapter's configured base model differs from --base-model.",
    )
    parser.add_argument(
        "--allow-architecture-mismatch",
        action="store_true",
        help="Allow merge even if adapter target modules or architecture differ from base model.",
    )
    parser.add_argument(
        "--trusted-anchor-dir",
        type=str,
        default=None,
        help="External directory outside model folder to write detached manifest SHA256 anchor.",
    )
    parser.add_argument(
        "--manifest-signature-file",
        type=str,
        default=None,
        help="External path to write detached manifest SHA256 hash/signature for trusted verification.",
    )
    parser.add_argument(
        "--current-symlink",
        type=str,
        default=None,
        help="Optional path to 'current' release symlink to atomically update after all verification gates pass.",
    )
    parser.add_argument(
        "--previous-symlink",
        type=str,
        default=None,
        help="Optional path to 'previous' release symlink to update for rollback.",
    )
    return parser.parse_args(argv)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hex digest of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_directory_fingerprint(dir_path: Path) -> Dict[str, str]:
    """Compute SHA256 fingerprints of key configuration and weight files in a directory."""
    fingerprints = {}
    for item in sorted(dir_path.iterdir()):
        if item.is_file() and (
            item.name.endswith(".json")
            or item.name.endswith(".safetensors")
            or item.name.endswith(".bin")
        ):
            fingerprints[item.name] = compute_file_sha256(item)
    return fingerprints


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
    """Normalize model string or local path for exact comparison."""
    p = Path(ident)
    if p.exists():
        return p.resolve().as_posix()
    return ident.strip().rstrip("/")


def validate_adapter_compatibility(
    base_model_id: str,
    adapter_path: Path,
    allow_mismatch: bool = False,
    override_reason: Optional[str] = None,
    adapter_revision: Optional[str] = None,
) -> Tuple[PeftConfig, bool, Dict[str, Any]]:
    """Ensure adapter exists and matches the exact base model and architecture."""
    adapter_config_file = adapter_path / "adapter_config.json"
    if not adapter_config_file.is_file():
        raise FileNotFoundError(
            f"adapter_config.json not found in adapter directory: {adapter_path}"
        )

    peft_config = PeftConfig.from_pretrained(str(adapter_path), revision=adapter_revision)
    logger.info("Loaded adapter config: %s", peft_config)

    configured_base = getattr(peft_config, "base_model_name_or_path", None)
    base_matched = False
    if configured_base:
        norm_conf = normalize_model_identifier(configured_base)
        norm_req = normalize_model_identifier(base_model_id)

        # Exact normalized equality
        if norm_conf == norm_req:
            base_matched = True
            logger.info("Adapter base model verified: %s matches %s", configured_base, base_model_id)
        else:
            msg = (
                f"Adapter configured base model ({configured_base!r}) does not match "
                f"requested base model ({base_model_id!r})."
            )
            if not allow_mismatch:
                raise ValueError(f"{msg} Pass --allow-base-mismatch and --override-reason to force merge.")
            if not override_reason or not override_reason.strip():
                raise ValueError(f"{msg} --allow-base-mismatch requires an explicit --override-reason.")
            logger.warning("%s Proceeding under explicit override: %s", msg, override_reason)
    else:
        msg = "Adapter config does not specify base_model_name_or_path."
        if not allow_mismatch:
            raise ValueError(f"{msg} Rejecting merge. Pass --allow-base-mismatch and --override-reason to force.")
        if not override_reason or not override_reason.strip():
            raise ValueError(f"{msg} --allow-base-mismatch requires an explicit --override-reason.")
        logger.warning("%s Proceeding under explicit override: %s", msg, override_reason)

    arch_fingerprint = {
        "peft_type": str(getattr(peft_config, "peft_type", "unknown")),
        "r": getattr(peft_config, "r", None),
        "lora_alpha": getattr(peft_config, "lora_alpha", None),
        "target_modules": sorted(list(getattr(peft_config, "target_modules", [])))
        if isinstance(getattr(peft_config, "target_modules", None), (list, set, tuple))
        else getattr(peft_config, "target_modules", None),
    }

    return peft_config, base_matched, arch_fingerprint


def resolve_model_provenance(
    model_ident: str,
    requested_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve commit hash and deterministic fingerprint for local directory or remote HF repository."""
    p = Path(model_ident)
    if p.is_dir():
        fingerprint = compute_directory_fingerprint(p)
        return {
            "identifier": str(p.resolve()),
            "provenance_mode": "local_directory",
            "requested_revision": requested_revision,
            "resolved_revision": requested_revision if requested_revision else "local_unversioned",
            "fingerprint": fingerprint,
            "revision_type": "label" if requested_revision else "none",
        }

    # Remote repository: attempt to resolve commit SHA via huggingface_hub or config
    resolved_commit = None
    try:
        from huggingface_hub import model_info as hf_model_info
        info = hf_model_info(model_ident, revision=requested_revision)
        resolved_commit = getattr(info, "sha", None)
    except Exception as e:
        logger.debug("Could not query Hugging Face Hub metadata for %s (%s)", model_ident, e)

    # If an immutable 40-char commit hash was requested, verify match
    if requested_revision and len(requested_revision) == 40 and all(c in "0123456789abcdefABCDEF" for c in requested_revision):
        if resolved_commit and resolved_commit.lower() != requested_revision.lower():
            raise ValueError(
                f"Requested commit SHA {requested_revision} does not match remote resolved commit {resolved_commit} for {model_ident}."
            )

    return {
        "identifier": model_ident,
        "provenance_mode": "hf_hub",
        "requested_revision": requested_revision,
        "resolved_revision": resolved_commit or requested_revision,
        "fingerprint": None,
        "revision_type": "commit_sha" if (resolved_commit or (requested_revision and len(requested_revision) == 40)) else "tag_or_branch",
    }


def validate_adapter_architecture(
    base_model: torch.nn.Module,
    peft_config: PeftConfig,
    allow_mismatch: bool = False,
    override_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate that adapter target modules exist in the base model architecture."""
    target_modules = getattr(peft_config, "target_modules", None)
    if isinstance(target_modules, (list, set, tuple)):
        target_list = sorted(list(target_modules))
    elif isinstance(target_modules, str):
        target_list = [target_modules]
    else:
        target_list = []

    base_named_modules = dict(base_model.named_modules())
    matched_targets = []
    missing_targets = []

    for target in target_list:
        found = False
        for mod_name in base_named_modules:
            if mod_name == target or mod_name.endswith(f".{target}") or target in mod_name.split("."):
                found = True
                break
        if found:
            matched_targets.append(target)
        else:
            missing_targets.append(target)

    base_model_type = getattr(base_model.config, "model_type", getattr(base_model, "__class__", {}).__name__)

    if missing_targets:
        msg = (
            f"Adapter target modules {missing_targets} were not found in loaded base model ({base_model_type})."
        )
        if not allow_mismatch:
            raise ValueError(f"{msg} Pass --allow-architecture-mismatch and --override-reason to force merge.")
        if not override_reason or not override_reason.strip():
            raise ValueError(f"{msg} --allow-architecture-mismatch requires an explicit --override-reason.")
        logger.warning("%s Proceeding under explicit override: %s", msg, override_reason)

    validation_result = {
        "validated": len(missing_targets) == 0,
        "base_model_type": base_model_type,
        "target_modules_requested": target_list,
        "target_modules_matched": matched_targets,
        "target_modules_missing": missing_targets,
        "override_reason": override_reason if missing_targets else None,
    }
    logger.info("Architecture validation passed: matched target modules %s on %s", matched_targets, base_model_type)
    return validation_result


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
    allow_architecture_mismatch: bool = False,
    override_reason: Optional[str] = None,
    trusted_anchor_dir: Optional[str] = None,
    manifest_signature_file: Optional[str] = None,
    current_symlink: Optional[str] = None,
    previous_symlink: Optional[str] = None,
    command_args: Optional[Dict[str, Any]] = None,
) -> Path:
    """Execute the merge, validation, detached integrity anchoring, and verified export."""
    from verify_model_export import verify_export

    out_path = Path(output_dir).resolve()
    adapter_path = Path(adapter_dir).resolve()

    if out_path.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {out_path}. Pass --force to export and retain backup."
            )
        logger.warning("Output directory %s exists; existing release will be retained as backup upon verified promotion.", out_path)

    if release_id is None:
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        release_id = f"tg-merged-{timestamp_str}"

    # 1. Resolve provenance for base and adapter
    base_provenance = resolve_model_provenance(base_model_id, base_revision)
    adapter_provenance = resolve_model_provenance(str(adapter_path), adapter_revision)
    logger.info("Base provenance resolved: %s", base_provenance["resolved_revision"])
    logger.info("Adapter provenance resolved: %s", adapter_provenance["resolved_revision"])

    # 2. Validate adapter compatibility and capture architecture fingerprint
    peft_config, base_matched, arch_fingerprint = validate_adapter_compatibility(
        base_model_id,
        adapter_path,
        allow_mismatch=allow_base_mismatch,
        override_reason=override_reason,
        adapter_revision=adapter_revision,
    )

    # 3. Resolve torch dtype
    torch_dtype = resolve_dtype(dtype_name)
    logger.info("Using dtype: %s (%s)", dtype_name, torch_dtype)

    # 4. Load processor & tokenizer with base revision selection
    logger.info("Loading processor from %s (revision=%s)...", base_model_id, base_revision)
    processor = AutoProcessor.from_pretrained(base_model_id, revision=base_revision, fix_markdown=False)
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

    # 5. Load base model configuration and weights with base revision selection
    logger.info("Loading model config from %s (revision=%s)...", base_model_id, base_revision)
    model_config = load_generation_safe_model_config(base_model_id, revision=base_revision)

    device_map = device if device in ("auto", "cpu", "cuda") else "auto"
    if device == "auto" and not torch.cuda.is_available():
        device_map = "cpu"

    logger.info("Loading base model %s (revision=%s, device_map=%s)...", base_model_id, base_revision, device_map)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        revision=base_revision,
        config=model_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    resolved_base_commit = getattr(base_model.config, "_commit_hash", None) or base_provenance.get("resolved_revision")
    base_fingerprint = base_provenance.get("fingerprint")
    adapter_fingerprint = adapter_provenance.get("fingerprint")

    # 6. Validate adapter architecture against base model modules
    arch_validation = validate_adapter_architecture(
        base_model=base_model,
        peft_config=peft_config,
        allow_mismatch=allow_architecture_mismatch,
        override_reason=override_reason,
    )

    # 7. Attach adapter via PEFT with adapter revision selection
    logger.info("Attaching LoRA adapter from %s (revision=%s)...", adapter_path, adapter_revision)
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        revision=adapter_revision,
        torch_dtype=torch_dtype,
    )

    # 8. Merge weights into base
    logger.info("Merging LoRA weights into base model (merge_and_unload)...")
    start_merge = time.time()
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()
    logger.info("Merge completed in %.2f seconds.", time.time() - start_merge)

    # 9. Build generation config
    gen_config = build_deterministic_generation_config(
        model_config=merged_model.config,
        processor=processor,
        base_model_id=base_model_id,
    )

    # 10. Write to atomic temporary directory
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

        # Build inventory and checksums for all payload files
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

        # Write SHA256SUMS (covers all payload files)
        sha256sums_path = tmp_dir / "SHA256SUMS"
        sha256sums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

        # Write merge_manifest.json (contains payload file inventory and provenance)
        manifest = {
            "release_id": release_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_model": base_model_id,
            "base_revision": base_revision,
            "resolved_base_commit": resolved_base_commit,
            "base_fingerprint": base_fingerprint,
            "base_model_info": base_provenance,
            "adapter": str(adapter_path),
            "adapter_revision": adapter_revision,
            "resolved_adapter_commit": adapter_provenance.get("resolved_revision"),
            "adapter_fingerprint": adapter_fingerprint,
            "adapter_info": adapter_provenance,
            "base_match_verified": base_matched,
            "override_reason": override_reason if (allow_base_mismatch or allow_architecture_mismatch) else None,
            "architecture_fingerprint": arch_fingerprint,
            "architecture_validation": arch_validation,
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

        # Write detached manifest checksum anchor
        manifest_sha256 = compute_file_sha256(manifest_path)
        manifest_anchor_path = tmp_dir / "merge_manifest.sha256"
        manifest_anchor_path.write_text(f"{manifest_sha256}  merge_manifest.json\n", encoding="utf-8")

        # External trusted anchor distribution if configured
        if trusted_anchor_dir:
            ext_dir = Path(trusted_anchor_dir).resolve()
            ext_dir.mkdir(parents=True, exist_ok=True)
            ext_anchor_file = ext_dir / f"{release_id}.sha256"
            ext_anchor_file.write_text(f"{manifest_sha256}  merge_manifest.json\n", encoding="utf-8")
            logger.info("Wrote external trusted manifest anchor to: %s", ext_anchor_file)

        if manifest_signature_file:
            sig_p = Path(manifest_signature_file).resolve()
            sig_p.parent.mkdir(parents=True, exist_ok=True)
            sig_p.write_text(f"{manifest_sha256}  merge_manifest.json\n", encoding="utf-8")
            logger.info("Wrote external manifest signature to: %s", sig_p)

        # Verification gate BEFORE promotion
        logger.info("Executing verification gate on staged export before promotion...")
        is_valid = verify_export(str(tmp_dir), skip_checksums=False, expected_manifest_sha256=manifest_sha256)
        if not is_valid:
            raise RuntimeError(f"Export verification gate failed on staged directory {tmp_dir}. Promotion aborted.")

        # Atomic promotion: retain existing release as retained backup if present
        backup_dir: Optional[Path] = None
        if out_path.exists():
            backup_dir = out_path.with_name(f"{out_path.name}.backup_{int(time.time())}")
            logger.info("Moving existing release to retained backup: %s", backup_dir)
            out_path.rename(backup_dir)

        try:
            tmp_dir.rename(out_path)
            logger.info("Successfully exported and verified merged checkpoint at: %s", out_path)
            if backup_dir and backup_dir.exists():
                logger.info("Previous release retained at %s for rollback.", backup_dir)
        except Exception:
            if backup_dir and backup_dir.exists():
                logger.error("Promotion failed; restoring previous release from backup...")
                if out_path.exists():
                    shutil.rmtree(out_path, ignore_errors=True)
                backup_dir.rename(out_path)
            raise

        # Update symlinks if requested
        if current_symlink:
            curr_p = Path(current_symlink)
            prev_target = None
            if curr_p.is_symlink() or curr_p.exists():
                try:
                    prev_target = curr_p.resolve()
                except Exception:
                    pass
                curr_p.unlink(missing_ok=True)
            curr_p.symlink_to(out_path, target_is_directory=True)
            logger.info("Updated 'current' symlink %s -> %s", curr_p, out_path)

            if previous_symlink and prev_target and prev_target.exists():
                prev_p = Path(previous_symlink)
                prev_p.unlink(missing_ok=True)
                prev_p.symlink_to(prev_target, target_is_directory=True)
                logger.info("Updated 'previous' symlink %s -> %s for rollback", prev_p, prev_target)

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
            allow_architecture_mismatch=args.allow_architecture_mismatch,
            override_reason=args.override_reason,
            trusted_anchor_dir=args.trusted_anchor_dir,
            manifest_signature_file=args.manifest_signature_file,
            current_symlink=args.current_symlink,
            previous_symlink=args.previous_symlink,
            command_args=command_args,
        )
        return 0
    except Exception as e:
        logger.error("Failed to merge and export LoRA adapter: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())


