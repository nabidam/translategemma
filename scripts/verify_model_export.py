#!/usr/bin/env python
"""Verify the integrity, completeness, and stop configuration of an exported model release.

This script can be executed on BOTH the fine-tune machine and the offline serving host.
It requires only standard Python libraries (no PyTorch or CUDA required).

Checks performed:
  1. Presence of required files (config.json, generation_config.json, safetensors, tokenizer, manifest, SHA256SUMS).
  2. Cryptographic SHA256 checksum verification against SHA256SUMS.
  3. Verification that generation_config.json contains <end_of_turn> (id 106) and <eos> (id 1) in eos_token_id.
  4. Verification of merge_manifest.json fields and provenance.

Usage:
  python scripts/verify_model_export.py --model-dir exports/translategemma-12b-it-merged-v1
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


def verify_checksums(model_dir: Path) -> Tuple[bool, List[str]]:
    sha256sums_file = model_dir / "SHA256SUMS"
    if not sha256sums_file.is_file():
        return False, ["SHA256SUMS file not found."]

    errors = []
    lines = sha256sums_file.read_text(encoding="utf-8").strip().splitlines()
    checked_count = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"Malformed line in SHA256SUMS: {line}")
            continue

        expected_hash, rel_path = parts[0], parts[1].lstrip("*").strip()
        target_file = model_dir / rel_path

        if not target_file.is_file():
            errors.append(f"File listed in SHA256SUMS does not exist: {rel_path}")
            continue

        actual_hash = compute_sha256(target_file)
        if actual_hash.lower() != expected_hash.lower():
            errors.append(
                f"Checksum mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
            )
        else:
            checked_count += 1

    if errors:
        return False, errors
    logger.info("Successfully verified checksums for %d files.", checked_count)
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

    # 3. Check manifest
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

    # 4. Check SHA256 sums
    if not skip_checksums:
        logger.info("Verifying SHA256 checksums of all artifact files...")
        ok_chk, chk_errors = verify_checksums(model_dir)
        if not ok_chk:
            logger.error("Checksum verification failed:\n%s", "\n".join(chk_errors))
            return False
        logger.info("All file checksums match SHA256SUMS.")
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
