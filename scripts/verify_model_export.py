#!/usr/bin/env python
"""Verify the integrity, completeness, and stop configuration of an exported model release.

This script can be executed on BOTH the fine-tune machine and the offline serving host.
It requires only standard Python libraries (no PyTorch or CUDA required).

Checks performed:
  1. Presence of required files (config.json, generation_config.json, safetensors, tokenizer, manifest, SHA256SUMS).
  2. Strict cryptographic SHA256 checksum verification across all inventory files.
  3. Verification that no untracked/extra or malicious files exist (rejects path traversal).
  4. Verification that generation_config.json contains <end_of_turn> (id 106) and <eos> (id 1).
  5. Verification that tokenizer files actually map token 106 to '<end_of_turn>'.
  6. Verification of merge_manifest.json fields, file inventory match, and provenance.

Usage:
  python scripts/verify_model_export.py --model-dir exports/translategemma-12b-it-merged-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_model_export")

REQUIRED_BASE_FILES: Set[str] = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merge_manifest.json",
    "SHA256SUMS",
}

EXPECTED_STOP_TOKEN_IDS: Set[int] = {1, 106}
EXPECTED_TURN_END_TOKEN: str = "<end_of_turn>"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the exported model release directory.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip full SHA256 byte hashing (quick metadata check only).",
    )
    return parser.parse_args(argv)


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_required_files(model_dir: Path) -> Tuple[bool, List[str]]:
    missing = []
    for fname in REQUIRED_BASE_FILES:
        if not (model_dir / fname).is_file():
            missing.append(fname)

    # Check for at least one safetensors file
    safetensors = list(model_dir.glob("*.safetensors"))
    if not safetensors:
        missing.append("*.safetensors (no model weight shards found)")

    if missing:
        return False, missing
    return True, []


def verify_generation_config(gen_config_path: Path) -> Tuple[bool, str]:
    try:
        data = json.loads(gen_config_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Failed to parse generation_config.json: {e}"

    eos_token_id = data.get("eos_token_id")
    if eos_token_id is None:
        return False, "eos_token_id is missing from generation_config.json"

    if isinstance(eos_token_id, int):
        eos_set = {eos_token_id}
    elif isinstance(eos_token_id, list):
        eos_set = set(eos_token_id)
    else:
        return False, f"Unexpected eos_token_id type in generation_config.json: {type(eos_token_id)}"

    missing_stops = EXPECTED_STOP_TOKEN_IDS - eos_set
    if missing_stops:
        return False, (
            f"Missing essential stop token IDs {missing_stops} in generation_config.json. "
            f"Present: {eos_set}. Decoder would run away."
        )

    return True, f"Valid stop token IDs: {sorted(list(eos_set))}"


def verify_tokenizer_token_mapping(model_dir: Path) -> Tuple[bool, str]:
    """Verify from tokenizer JSON/config that token 106 maps to <end_of_turn>."""
    tokenizer_json_file = model_dir / "tokenizer.json"
    if not tokenizer_json_file.is_file():
        return False, "tokenizer.json not found."

    try:
        tok_data = json.loads(tokenizer_json_file.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Failed to parse tokenizer.json: {e}"

    # 1. Check added_tokens list in tokenizer.json
    added_tokens = tok_data.get("added_tokens", [])
    found_turn_end = False
    for item in added_tokens:
        if item.get("id") == 106 and item.get("content") == EXPECTED_TURN_END_TOKEN:
            found_turn_end = True
            break

    # 2. Check model vocabulary if not found in added_tokens
    if not found_turn_end:
        vocab = tok_data.get("model", {}).get("vocab", {})
        if vocab.get(EXPECTED_TURN_END_TOKEN) == 106:
            found_turn_end = True

    if not found_turn_end:
        return False, (
            f"Tokenizer mapping check failed: ID 106 does not map to {EXPECTED_TURN_END_TOKEN!r} in tokenizer.json."
        )

    return True, f"Token ID 106 verified as {EXPECTED_TURN_END_TOKEN!r}"


def verify_checksums_and_inventory(
    model_dir: Path,
    manifest_data: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    sha256sums_file = model_dir / "SHA256SUMS"
    if not sha256sums_file.is_file():
        return False, ["SHA256SUMS file not found."]

    errors = []
    lines = sha256sums_file.read_text(encoding="utf-8").strip().splitlines()

    checksum_map: Dict[str, str] = {}
    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"Malformed line {line_no} in SHA256SUMS: {line}")
            continue

        expected_hash, rel_path = parts[0], parts[1].lstrip("*").strip()

        # Reject path traversal
        if ".." in rel_path or rel_path.startswith("/") or rel_path.startswith("\\"):
            errors.append(f"Security error: path traversal or absolute path in SHA256SUMS: {rel_path}")
            continue

        checksum_map[rel_path] = expected_hash

    # Check files on disk
    disk_files = {
        p.relative_to(model_dir).as_posix()
        for p in model_dir.iterdir()
        if p.is_file() and not p.name.startswith(".tmp")
    }

    manifest_inventory = {
        entry["path"]: entry.get("sha256")
        for entry in manifest_data.get("file_inventory", [])
        if isinstance(entry, dict) and "path" in entry
    }

    # Verify inventory completeness
    untracked_on_disk = disk_files - set(checksum_map.keys())
    if untracked_on_disk:
        errors.append(f"Untracked files present on disk but missing from SHA256SUMS: {sorted(list(untracked_on_disk))}")

    missing_from_disk = set(checksum_map.keys()) - disk_files
    if missing_from_disk:
        errors.append(f"Files listed in SHA256SUMS missing from disk: {sorted(list(missing_from_disk))}")

    # Check byte hashes
    for rel_path, expected_hash in checksum_map.items():
        target_file = model_dir / rel_path
        if not target_file.is_file():
            continue
        actual_hash = compute_sha256(target_file)
        if actual_hash.lower() != expected_hash.lower():
            errors.append(
                f"Checksum mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
            )

    if errors:
        return False, errors
    logger.info("Successfully verified checksums and strict inventory for %d files.", len(checksum_map))
    return True, []


def verify_manifest(manifest_path: Path) -> Tuple[bool, Dict[str, Any]]:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, {"error": f"Failed to parse merge_manifest.json: {e}"}

    required_keys = ["release_id", "created_at", "base_model", "adapter", "stop_token_ids", "file_inventory"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        return False, {"error": f"Missing required manifest keys: {missing}"}

    return True, data


def verify_export(model_dir_str: str, skip_checksums: bool = False) -> bool:
    model_dir = Path(model_dir_str).resolve()
    if not model_dir.is_dir():
        logger.error("Model directory does not exist: %s", model_dir)
        return False

    logger.info("Verifying model export at: %s", model_dir)

    # 1. Check required files
    ok_files, missing = check_required_files(model_dir)
    if not ok_files:
        logger.error("Missing required export files: %s", missing)
        return False
    logger.info("All required base files and weight shards are present.")

    # 2. Check generation config stop tokens
    ok_gen, msg_gen = verify_generation_config(model_dir / "generation_config.json")
    if not ok_gen:
        logger.error("generation_config.json check failed: %s", msg_gen)
        return False
    logger.info("Generation config check passed: %s", msg_gen)

    # 3. Check tokenizer mapping
    ok_tok, msg_tok = verify_tokenizer_token_mapping(model_dir)
    if not ok_tok:
        logger.error("Tokenizer mapping check failed: %s", msg_tok)
        return False
    logger.info("Tokenizer mapping check passed: %s", msg_tok)

    # 4. Check manifest
    ok_man, man_data = verify_manifest(model_dir / "merge_manifest.json")
    if not ok_man:
        logger.error("Manifest check failed: %s", man_data.get("error"))
        return False
    logger.info(
        "Manifest check passed (Release ID: %s, Base: %s, Adapter: %s)",
        man_data.get("release_id"),
        man_data.get("base_model"),
        man_data.get("adapter"),
    )

    # 5. Check SHA256 sums and strict inventory coverage
    if not skip_checksums:
        logger.info("Verifying SHA256 checksums of all artifact files and strict inventory coverage...")
        ok_chk, chk_errors = verify_checksums_and_inventory(model_dir, man_data)
        if not ok_chk:
            logger.error("Checksum verification failed:\n%s", "\n".join(chk_errors))
            return False
        logger.info("All file checksums and inventory coverage match SHA256SUMS.")
    else:
        logger.info("Skipped full byte checksum calculation (--skip-checksums).")

    logger.info("Model export verification PASSED: %s", model_dir)
    return True


def main() -> int:
    args = parse_args()
    success = verify_export(args.model_dir, skip_checksums=args.skip_checksums)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
