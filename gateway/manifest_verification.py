"""Manifest authenticity and model payload verification for the serving gateway.

The gateway ships standalone (no repository on disk), so this module deliberately
duplicates the verification contract of scripts/verify_model_export.py using only
the standard library.

Two distinct properties are tracked separately, because conflating them is what
lets a tampered release look verified:

* **Integrity** — the manifest matches a checksum stored *beside it*. Anyone who
  can rewrite the manifest can rewrite that checksum, so integrity is a diagnostic,
  never an authorization.
* **Authenticity** — the manifest matches a hash supplied from *outside* the model
  release (an operator-configured digest or an anchor file on a separate mount).
  Only this can gate production readiness.

Authenticating the manifest still says nothing about the weights the engine loads:
payload verification hashes every mounted file against the authenticated manifest
inventory and SHA256SUMS, so post-promotion edits to config, tokenizer, or shards
are caught before the gateway reports ready.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gateway.manifest_verification")

# Files describing the release rather than being loaded by the engine.
METADATA_RELEASE_FILES = {"SHA256SUMS", "merge_manifest.json", "merge_manifest.sha256"}

AUTHENTICITY_TRUSTED = "trusted_external_anchor"
AUTHENTICITY_COLOCATED = "colocated_checksum_only"
AUTHENTICITY_UNVERIFIED = "unverified"

REQUIRED_STOP_TOKEN_IDS = {1, 106}
CHAT_TURN_END_TOKEN = "<end_of_turn>"


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_anchor_hash(anchor_path: Path) -> str:
    """Read the leading hex digest out of a `sha256  filename` style anchor file."""
    text = anchor_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Anchor file is empty: {anchor_path}")
    return text.split()[0].strip().lower()


def verify_payload_against_manifest(
    model_dir: Path,
    manifest_data: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Hash every mounted payload file against the manifest inventory and SHA256SUMS.

    Returns (ok, errors). Set equality is enforced in both directions so that an
    extra unlisted shard is as fatal as a modified one.
    """
    errors: List[str] = []

    inventory: Dict[str, Dict[str, Any]] = {}
    for entry in manifest_data.get("file_inventory", []):
        if isinstance(entry, dict) and "path" in entry:
            inventory[str(entry["path"])] = entry
    if not inventory:
        return False, ["merge_manifest.json contains no file_inventory; payload cannot be verified."]

    checksum_map: Dict[str, str] = {}
    sums_file = model_dir / "SHA256SUMS"
    if sums_file.is_file():
        for line_no, raw_line in enumerate(sums_file.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                errors.append(f"Malformed line {line_no} in SHA256SUMS: {line}")
                continue
            digest, rel_path = parts[0].lower(), parts[1].lstrip("*").strip()
            if ".." in rel_path or rel_path.startswith("/") or rel_path.startswith("\\"):
                errors.append(f"Path traversal or absolute path in SHA256SUMS: {rel_path}")
                continue
            checksum_map[rel_path] = digest
    else:
        errors.append("SHA256SUMS not found in mounted model directory.")

    disk_files = {
        p.name
        for p in model_dir.iterdir()
        if p.is_file() and p.name not in METADATA_RELEASE_FILES and not p.name.startswith(".tmp")
    }

    untracked = sorted(disk_files - set(inventory))
    if untracked:
        errors.append(f"Mounted payload files absent from the authenticated manifest inventory: {untracked}")
    missing = sorted(set(inventory) - disk_files)
    if missing:
        errors.append(f"Manifest inventory files missing from the mounted model directory: {missing}")
    if checksum_map:
        sums_mismatch = sorted(set(checksum_map) ^ set(inventory))
        if sums_mismatch:
            errors.append(f"SHA256SUMS and manifest inventory disagree on file set: {sums_mismatch}")

    verified = 0
    for rel_path, entry in sorted(inventory.items()):
        target = model_dir / rel_path
        if not target.is_file():
            continue
        actual_hash = compute_sha256(target)
        expected_hash = str(entry.get("sha256", "")).lower()
        if actual_hash.lower() != expected_hash:
            errors.append(
                f"Payload hash mismatch for {rel_path}: manifest expects {expected_hash}, mounted file is {actual_hash}"
            )
            continue
        expected_size = entry.get("size_bytes")
        actual_size = target.stat().st_size
        if expected_size is not None and expected_size != actual_size:
            errors.append(
                f"Payload size mismatch for {rel_path}: manifest expects {expected_size} bytes, mounted file is {actual_size}"
            )
            continue
        sums_hash = checksum_map.get(rel_path)
        if sums_hash and sums_hash != actual_hash.lower():
            errors.append(f"SHA256SUMS disagrees with mounted payload for {rel_path}.")
            continue
        verified += 1

    if errors:
        return False, errors
    logger.info("Verified %d mounted payload files against the authenticated manifest.", verified)
    return True, []


def resolve_runtime_stop_contract(model_dir: Path) -> Tuple[Optional[List[int]], Optional[List[str]], List[str]]:
    """Independently resolve stop token IDs from the mounted generation config and tokenizer.

    Returns (stop_token_ids, stop_tokens, errors). This is deliberately independent
    of merge_manifest.json: the manifest is provenance, while these are the bytes the
    engine actually loads, and a disagreement between them is a deployment defect.
    """
    errors: List[str] = []
    gen_config_path = model_dir / "generation_config.json"
    if not gen_config_path.is_file():
        return None, None, [f"generation_config.json not found in {model_dir}"]

    try:
        gen_data = json.loads(gen_config_path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, None, [f"Failed to parse generation_config.json: {e}"]

    eos = gen_data.get("eos_token_id")
    if isinstance(eos, int):
        stop_ids = {eos}
    elif isinstance(eos, list):
        stop_ids = {int(i) for i in eos if i is not None}
    else:
        return None, None, [f"generation_config.json has unusable eos_token_id: {eos!r}"]

    missing = REQUIRED_STOP_TOKEN_IDS - stop_ids
    if missing:
        errors.append(
            f"generation_config.json is missing required stop token IDs {sorted(missing)} "
            f"(present: {sorted(stop_ids)}); the decoder would run past the turn boundary."
        )

    id_to_token: Dict[int, str] = {}
    tokenizer_path = model_dir / "tokenizer.json"
    if tokenizer_path.is_file():
        try:
            tok_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"Failed to parse tokenizer.json: {e}")
            tok_data = {}
        for item in tok_data.get("added_tokens", []):
            if isinstance(item, dict) and isinstance(item.get("id"), int):
                id_to_token[item["id"]] = item.get("content", "")
        if id_to_token.get(106) != CHAT_TURN_END_TOKEN:
            vocab = tok_data.get("model", {}).get("vocab", {})
            if vocab.get(CHAT_TURN_END_TOKEN) == 106:
                id_to_token[106] = CHAT_TURN_END_TOKEN
            else:
                errors.append(f"Mounted tokenizer does not map ID 106 to {CHAT_TURN_END_TOKEN!r}.")
    else:
        errors.append("tokenizer.json not found in mounted model directory.")

    sorted_ids = sorted(stop_ids)
    stop_tokens = [id_to_token.get(tid, f"<id:{tid}>") for tid in sorted_ids]
    return sorted_ids, stop_tokens, errors


class ManifestVerificationResult:
    """Separated integrity / authenticity / payload state for one mounted release."""

    def __init__(
        self,
        is_loaded: bool = False,
        integrity_verified: bool = False,
        authenticity_verified: bool = False,
        authenticity_status: str = AUTHENTICITY_UNVERIFIED,
        payload_verified: bool = False,
        payload_verified_at: Optional[str] = None,
        manifest_sha256: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        runtime_stop_token_ids: Optional[List[int]] = None,
        runtime_stop_tokens: Optional[List[str]] = None,
        error: Optional[str] = None,
    ):
        self.is_loaded = is_loaded
        self.integrity_verified = integrity_verified
        self.authenticity_verified = authenticity_verified
        self.authenticity_status = authenticity_status
        self.payload_verified = payload_verified
        self.payload_verified_at = payload_verified_at
        self.manifest_sha256 = manifest_sha256
        self.data = data or {}
        self.runtime_stop_token_ids = runtime_stop_token_ids
        self.runtime_stop_tokens = runtime_stop_tokens
        self.error = error

    @property
    def is_trusted(self) -> bool:
        """Only an externally anchored manifest whose payload matches may drive runtime behavior."""
        return self.authenticity_verified and self.payload_verified


def verify_model_release(
    model_dir_str: Optional[str],
    trusted_manifest_sha256: Optional[str] = None,
    trusted_anchor_file: Optional[str] = None,
    verify_payload: bool = True,
) -> ManifestVerificationResult:
    """Load the manifest, classify its trust level, and verify the mounted payload."""
    if not model_dir_str:
        return ManifestVerificationResult(error="No model_dir configured.")

    model_dir = Path(model_dir_str)
    manifest_file = model_dir / "merge_manifest.json"
    if not manifest_file.is_file():
        return ManifestVerificationResult(error=f"merge_manifest.json not found in {model_dir}")

    try:
        content_bytes = manifest_file.read_bytes()
        manifest_sha256 = hashlib.sha256(content_bytes).hexdigest()
        data = json.loads(content_bytes.decode("utf-8"))
    except Exception as e:
        return ManifestVerificationResult(error=f"Failed to read/parse merge_manifest.json: {e}")

    def failure(message: str) -> ManifestVerificationResult:
        logger.error("%s", message)
        return ManifestVerificationResult(
            is_loaded=True,
            manifest_sha256=manifest_sha256,
            data=data,
            error=message,
        )

    integrity_verified = False
    authenticity_verified = False
    authenticity_status = AUTHENTICITY_UNVERIFIED

    if trusted_manifest_sha256:
        expected = trusted_manifest_sha256.strip().lower()
        if manifest_sha256.lower() != expected:
            return failure(
                f"Manifest SHA256 mismatch with TG_TRUSTED_MANIFEST_SHA256: computed {manifest_sha256}, expected {expected}"
            )
        integrity_verified = True
        authenticity_verified = True
        authenticity_status = AUTHENTICITY_TRUSTED
        logger.info("Manifest authenticated against operator-configured trusted SHA256 anchor.")
    elif trusted_anchor_file:
        anchor_path = Path(trusted_anchor_file)
        if not anchor_path.is_file():
            return failure(f"Configured TG_TRUSTED_ANCHOR_FILE not found: {anchor_path}")
        try:
            expected = read_anchor_hash(anchor_path)
        except Exception as e:
            return failure(f"Failed to read trusted anchor file {anchor_path}: {e}")
        if manifest_sha256.lower() != expected:
            return failure(
                f"Manifest SHA256 mismatch with trusted anchor file {anchor_path}: "
                f"computed {manifest_sha256}, expected {expected}"
            )
        integrity_verified = True
        authenticity_verified = True
        authenticity_status = AUTHENTICITY_TRUSTED
        logger.info("Manifest authenticated against external trusted anchor file %s.", anchor_path)
    else:
        local_anchor = model_dir / "merge_manifest.sha256"
        if local_anchor.is_file():
            try:
                expected = read_anchor_hash(local_anchor)
            except Exception as e:
                return failure(f"Could not read co-located merge_manifest.sha256: {e}")
            if manifest_sha256.lower() != expected:
                return failure(
                    f"Co-located merge_manifest.sha256 mismatch: computed {manifest_sha256}, expected {expected}"
                )
            integrity_verified = True
            authenticity_status = AUTHENTICITY_COLOCATED
            logger.warning(
                "Manifest matches only its co-located checksum. This is integrity evidence, NOT authenticity: "
                "configure TG_TRUSTED_MANIFEST_SHA256 or TG_TRUSTED_ANCHOR_FILE for production."
            )
        else:
            logger.warning("No integrity anchor found for merge_manifest.json.")

    payload_verified = False
    payload_verified_at: Optional[str] = None
    payload_error: Optional[str] = None
    if verify_payload:
        ok, payload_errors = verify_payload_against_manifest(model_dir, data)
        if ok:
            payload_verified = True
            payload_verified_at = datetime.now(timezone.utc).isoformat()
        else:
            payload_error = "Mounted payload does not match manifest: " + "; ".join(payload_errors)
            logger.error("%s", payload_error)

    stop_ids, stop_tokens, stop_errors = resolve_runtime_stop_contract(model_dir)
    stop_error = "; ".join(stop_errors) if stop_errors else None
    if stop_error:
        logger.error("Mounted stop contract problem: %s", stop_error)

    combined_error = "; ".join(e for e in (payload_error, stop_error) if e) or None

    return ManifestVerificationResult(
        is_loaded=True,
        integrity_verified=integrity_verified,
        authenticity_verified=authenticity_verified,
        authenticity_status=authenticity_status,
        payload_verified=payload_verified,
        payload_verified_at=payload_verified_at,
        manifest_sha256=manifest_sha256,
        data=data,
        runtime_stop_token_ids=stop_ids,
        runtime_stop_tokens=stop_tokens,
        error=combined_error,
    )
