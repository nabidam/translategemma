#!/usr/bin/env python
"""Atomic release promotion and rollback manager for TranslateGemma model releases.

Active serving pointers (/opt/models/translategemma/current) are switched ONLY after:

  1. Full export verification (payload checksums + strict inventory coverage).
  2. Manifest authenticity against an EXTERNAL trust anchor. A co-located
     merge_manifest.sha256 is integrity evidence only: anyone who can rewrite the
     manifest can rewrite the checksum beside it, so it can never authorize activation.
  3. A release attestation binding every required behavioral gate result
     (merged-vs-adapter quality, degeneration, vLLM stop-token smoke test, deployment
     compatibility preflight) to this release's manifest SHA256.

Every activation is recorded in a release index so that rollback verifies the target
against the manifest hash that was trusted when it was promoted, and so that repeated
rollback/roll-forward walks real history instead of ping-ponging one stale pointer.

Usage:
  # Promote a verified, attested release:
  python scripts/promote_model_release.py promote \
      --release-dir /opt/models/translategemma/releases/tg-merged-v1 \
      --current-symlink /opt/models/translategemma/current \
      --previous-symlink /opt/models/translategemma/previous \
      --trusted-anchor-file /opt/models/translategemma/anchors/tg-merged-v1.sha256 \
      --attestation-file /opt/models/translategemma/attestations/tg-merged-v1.json

  # Roll back to the previous known-good release:
  python scripts/promote_model_release.py rollback \
      --current-symlink /opt/models/translategemma/current \
      --previous-symlink /opt/models/translategemma/previous

  # Check active and previous release status:
  python scripts/promote_model_release.py status \
      --current-symlink /opt/models/translategemma/current \
      --previous-symlink /opt/models/translategemma/previous

Attestation file format (JSON):
  {
    "manifest_sha256": "<hex digest of merge_manifest.json>",
    "release_id": "tg-merged-v1",
    "vllm_image_digest": "sha256:...",
    "gates": {
      "merged_quality":       {"passed": true, "verified_at": "2026-08-16T10:00:00+00:00",
                                "evidence": "reports/merged_quality.json",
                                "evidence_sha256": "<hex>"},
      "degeneration":         {...},
      "vllm_smoke":           {...},
      "deployment_preflight": {...}
    }
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from verify_model_export import compute_sha256, verify_export

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("promote_model_release")

REQUIRED_GATES = (
    "merged_quality",
    "degeneration",
    "vllm_smoke",
    "deployment_preflight",
)
DEFAULT_MAX_EVIDENCE_AGE_DAYS = 30
RELEASE_INDEX_VERSION = 1


def atomic_symlink_switch(target_path: Path, symlink_path: Path) -> None:
    """Atomically switch a symlink to point to target_path using an atomic rename."""
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    temp_symlink = symlink_path.parent / f".tmp_symlink_{symlink_path.name}_{os.getpid()}"
    try:
        if temp_symlink.is_symlink() or temp_symlink.exists():
            temp_symlink.unlink()
        os.symlink(target_path, temp_symlink)
        os.replace(temp_symlink, symlink_path)
    finally:
        if temp_symlink.is_symlink() or temp_symlink.exists():
            try:
                temp_symlink.unlink()
            except Exception:
                pass


def default_index_path(current_symlink: Path) -> Path:
    """Release index lives beside the pointers it describes."""
    return current_symlink.parent / "release_index.json"


def load_release_index(index_path: Path) -> Dict[str, Any]:
    if not index_path.is_file():
        return {"version": RELEASE_INDEX_VERSION, "active": None, "history": []}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Release index {index_path} is unreadable: {e}") from e
    data.setdefault("version", RELEASE_INDEX_VERSION)
    data.setdefault("active", None)
    data.setdefault("history", [])
    return data


def write_release_index(index_path: Path, index: Dict[str, Any]) -> None:
    """Write the index atomically so a crash never leaves a truncated trust record."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.parent / f".tmp_{index_path.name}_{os.getpid()}"
    tmp_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, index_path)


def read_trusted_anchor_hash(trusted_anchor_file: Optional[str]) -> Optional[str]:
    if not trusted_anchor_file:
        return None
    anchor_path = Path(trusted_anchor_file).resolve()
    if not anchor_path.is_file():
        raise FileNotFoundError(f"Trusted anchor file not found: {anchor_path}")
    text = anchor_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Trusted anchor file is empty: {anchor_path}")
    return text.split()[0].strip().lower()


def validate_attestation(
    attestation_path: Path,
    manifest_sha256: str,
    expected_image_digest: Optional[str] = None,
    max_age_days: int = DEFAULT_MAX_EVIDENCE_AGE_DAYS,
) -> Dict[str, Any]:
    """Validate that every required gate passed for exactly this artifact, recently.

    Raises ValueError with the specific reason on any missing, stale, failed, or
    misbound gate. Returns the parsed attestation on success.
    """
    if not attestation_path.is_file():
        raise ValueError(f"Release attestation file not found: {attestation_path}")

    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Release attestation {attestation_path} is not valid JSON: {e}") from e

    attested_manifest = str(attestation.get("manifest_sha256", "")).strip().lower()
    if not attested_manifest:
        raise ValueError("Release attestation does not declare manifest_sha256.")
    if attested_manifest != manifest_sha256.lower():
        raise ValueError(
            f"Release attestation is bound to manifest {attested_manifest}, but the release directory's "
            f"merge_manifest.json hashes to {manifest_sha256}. Attestation belongs to a different artifact."
        )

    if expected_image_digest:
        attested_digest = str(attestation.get("vllm_image_digest", "")).strip().lower()
        if attested_digest != expected_image_digest.strip().lower():
            raise ValueError(
                f"Release attestation records vLLM image digest {attested_digest or '(none)'}, "
                f"but deployment expects {expected_image_digest}."
            )

    gates = attestation.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("Release attestation has no 'gates' object.")

    now = datetime.now(timezone.utc)
    max_age = timedelta(days=max_age_days)

    for gate_name in REQUIRED_GATES:
        gate = gates.get(gate_name)
        if not isinstance(gate, dict):
            raise ValueError(f"Release attestation is missing required gate {gate_name!r}.")
        if gate.get("passed") is not True:
            raise ValueError(f"Required gate {gate_name!r} did not pass: {gate.get('detail') or gate.get('passed')!r}.")

        verified_at_raw = gate.get("verified_at")
        if not verified_at_raw:
            raise ValueError(f"Required gate {gate_name!r} has no verified_at timestamp.")
        try:
            verified_at = datetime.fromisoformat(str(verified_at_raw))
        except ValueError as e:
            raise ValueError(f"Required gate {gate_name!r} has an unparsable verified_at {verified_at_raw!r}: {e}") from e
        if verified_at.tzinfo is None:
            verified_at = verified_at.replace(tzinfo=timezone.utc)
        if now - verified_at > max_age:
            raise ValueError(
                f"Required gate {gate_name!r} evidence is stale: verified {verified_at.isoformat()}, "
                f"older than the {max_age_days}-day limit."
            )

        evidence = gate.get("evidence")
        if not evidence:
            raise ValueError(f"Required gate {gate_name!r} references no evidence artifact.")
        evidence_path = Path(evidence)
        if not evidence_path.is_absolute():
            evidence_path = (attestation_path.parent / evidence_path).resolve()
        if not evidence_path.is_file():
            raise ValueError(f"Evidence artifact for gate {gate_name!r} not found: {evidence_path}")

        declared_hash = str(gate.get("evidence_sha256", "")).strip().lower()
        if not declared_hash:
            raise ValueError(f"Required gate {gate_name!r} does not declare evidence_sha256.")
        actual_hash = compute_sha256(evidence_path).lower()
        if declared_hash != actual_hash:
            raise ValueError(
                f"Evidence artifact for gate {gate_name!r} does not match its declared hash: "
                f"attestation says {declared_hash}, file hashes to {actual_hash}."
            )

    logger.info(
        "Release attestation validated: all required gates %s passed and bind to manifest %s.",
        list(REQUIRED_GATES),
        manifest_sha256,
    )
    return attestation


def summarize_attestation(attestation: Dict[str, Any], attestation_path: Path) -> Dict[str, Any]:
    gates = attestation.get("gates", {})
    return {
        "attestation_file": str(attestation_path.resolve()),
        "attestation_sha256": compute_sha256(attestation_path),
        "vllm_image_digest": attestation.get("vllm_image_digest"),
        "gates": {
            name: {
                "verified_at": gates.get(name, {}).get("verified_at"),
                "evidence_sha256": gates.get(name, {}).get("evidence_sha256"),
            }
            for name in REQUIRED_GATES
        },
    }


def record_activation(
    index_path: Path,
    release_dir: Path,
    manifest_sha256: str,
    attestation_summary: Optional[Dict[str, Any]],
    action: str,
) -> None:
    index = load_release_index(index_path)
    entry = {
        "release_dir": str(release_dir),
        "manifest_sha256": manifest_sha256.lower(),
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "attestation": attestation_summary,
    }
    index["history"].append(entry)
    index["active"] = entry
    write_release_index(index_path, index)


def find_index_entry(index: Dict[str, Any], release_dir: Path) -> Optional[Dict[str, Any]]:
    """Return the most recent index entry recorded for release_dir."""
    target = str(release_dir)
    for entry in reversed(index.get("history", [])):
        if entry.get("release_dir") == target:
            return entry
    return None


def promote_release(
    release_dir: Path,
    current_symlink: Path,
    previous_symlink: Path,
    attestation_file: Optional[str] = None,
    expected_manifest_sha256: Optional[str] = None,
    trusted_anchor_file: Optional[str] = None,
    expected_image_digest: Optional[str] = None,
    max_evidence_age_days: int = DEFAULT_MAX_EVIDENCE_AGE_DAYS,
    index_file: Optional[str] = None,
    skip_checksums: bool = False,
) -> bool:
    """Verify release against all trust gates and atomically switch the current pointer."""
    release_dir = release_dir.resolve()
    if not release_dir.is_dir():
        logger.error("Release directory does not exist: %s", release_dir)
        return False

    manifest_path = release_dir / "merge_manifest.json"
    if not manifest_path.is_file():
        logger.error("merge_manifest.json not found in release directory: %s", release_dir)
        return False

    # 1. External trust anchor is mandatory. A co-located checksum is not authenticity.
    try:
        anchor_hash = read_trusted_anchor_hash(trusted_anchor_file)
    except Exception as e:
        logger.error("Trusted anchor could not be read: %s", e)
        return False

    expected_hash = (expected_manifest_sha256 or anchor_hash or "").strip().lower()
    if not expected_hash:
        logger.error(
            "Promotion requires an external trust anchor: pass --expected-manifest-sha256 or "
            "--trusted-anchor-file pointing outside the release directory. Co-located "
            "merge_manifest.sha256 cannot authorize activation."
        )
        return False
    if anchor_hash and expected_manifest_sha256 and anchor_hash != expected_manifest_sha256.strip().lower():
        logger.error(
            "Conflicting trust inputs: --expected-manifest-sha256 (%s) does not match the anchor file (%s).",
            expected_manifest_sha256.strip().lower(),
            anchor_hash,
        )
        return False

    actual_manifest_hash = compute_sha256(manifest_path).lower()
    if actual_manifest_hash != expected_hash:
        logger.error(
            "Manifest authenticity FAILED: trusted anchor expects %s, release manifest hashes to %s.",
            expected_hash,
            actual_manifest_hash,
        )
        return False

    # 2. Full payload verification against the authenticated manifest.
    logger.info("Executing pre-promotion payload verification gate on %s...", release_dir)
    is_valid = verify_export(
        str(release_dir),
        skip_checksums=skip_checksums,
        expected_manifest_sha256=expected_hash,
    )
    if not is_valid:
        logger.error("Promotion gate FAILED on %s. Active serving pointer NOT changed.", release_dir)
        return False

    # 3. Behavioral gate evidence bound to this exact artifact.
    if not attestation_file:
        logger.error(
            "Promotion requires --attestation-file with passing evidence for gates %s bound to manifest %s.",
            list(REQUIRED_GATES),
            actual_manifest_hash,
        )
        return False
    attestation_path = Path(attestation_file).resolve()
    try:
        attestation = validate_attestation(
            attestation_path,
            manifest_sha256=actual_manifest_hash,
            expected_image_digest=expected_image_digest,
            max_age_days=max_evidence_age_days,
        )
    except Exception as e:
        logger.error("Release attestation gate FAILED: %s", e)
        return False

    old_current_target: Optional[Path] = None
    if current_symlink.is_symlink() or current_symlink.exists():
        try:
            old_current_target = current_symlink.resolve()
            logger.info("Current active release before promotion: %s", old_current_target)
        except Exception as e:
            logger.warning("Could not resolve current symlink: %s", e)

    # 4. Update 'previous' BEFORE switching 'current', so no window exists in which
    #    the outgoing release is unreachable for rollback.
    if old_current_target and old_current_target.is_dir() and old_current_target != release_dir:
        logger.info("Updating 'previous' rollback pointer %s -> %s", previous_symlink, old_current_target)
        atomic_symlink_switch(old_current_target, previous_symlink)

    logger.info("Atomically switching 'current' symlink %s -> %s", current_symlink, release_dir)
    atomic_symlink_switch(release_dir, current_symlink)

    index_path = Path(index_file).resolve() if index_file else default_index_path(current_symlink)
    record_activation(
        index_path,
        release_dir,
        actual_manifest_hash,
        summarize_attestation(attestation, attestation_path),
        action="promote",
    )

    logger.info("Release promotion SUCCESSFUL. Active release: %s (index: %s)", release_dir, index_path)
    return True


def rollback_release(
    current_symlink: Path,
    previous_symlink: Path,
    index_file: Optional[str] = None,
    trusted_anchor_file: Optional[str] = None,
    skip_checksums: bool = False,
) -> bool:
    """Atomically swap current and previous pointers after re-verifying the rollback target."""
    index_path = Path(index_file).resolve() if index_file else default_index_path(current_symlink)
    try:
        index = load_release_index(index_path)
    except Exception as e:
        logger.error("%s", e)
        return False

    if not previous_symlink.exists() and not previous_symlink.is_symlink():
        logger.error("No 'previous' release pointer found at %s. Rollback impossible.", previous_symlink)
        return False

    try:
        prev_target = previous_symlink.resolve()
    except Exception as e:
        logger.error("Failed to resolve 'previous' symlink %s: %s", previous_symlink, e)
        return False

    if not prev_target.is_dir():
        logger.error("Previous release target directory does not exist: %s", prev_target)
        return False

    current_target: Optional[Path] = None
    if current_symlink.is_symlink() or current_symlink.exists():
        try:
            current_target = current_symlink.resolve()
        except Exception as e:
            logger.warning("Could not resolve current symlink: %s", e)

    if current_target == prev_target:
        logger.error(
            "'previous' (%s) is the same release as 'current'. Rollback would be a no-op; "
            "inspect %s for the real activation history.",
            prev_target,
            index_path,
        )
        return False

    # Trust anchor for the rollback target: an explicit anchor file wins, otherwise the
    # hash that was trusted when this release was promoted. Rollback is exactly when a
    # tampered "known good" release is most dangerous, so this is not optional.
    try:
        anchor_hash = read_trusted_anchor_hash(trusted_anchor_file)
    except Exception as e:
        logger.error("Trusted anchor could not be read: %s", e)
        return False

    recorded = find_index_entry(index, prev_target)
    expected_hash = anchor_hash or (recorded or {}).get("manifest_sha256")
    if not expected_hash:
        logger.error(
            "No trusted manifest hash is recorded for rollback target %s in %s, and no "
            "--trusted-anchor-file was supplied. Refusing to activate an unauthenticated release.",
            prev_target,
            index_path,
        )
        return False

    logger.info("Verifying rollback target %s against trusted manifest %s...", prev_target, expected_hash)
    is_valid = verify_export(
        str(prev_target),
        skip_checksums=skip_checksums,
        expected_manifest_sha256=expected_hash,
    )
    if not is_valid:
        logger.error("Rollback target failed authenticity/integrity verification: %s. Rollback aborted.", prev_target)
        return False

    # Atomic swap: the release we are leaving becomes the new rollback target, so a
    # second rollback returns to it instead of pinning one stale pointer forever.
    if current_target and current_target.is_dir():
        logger.info("Repointing 'previous' %s -> %s (release being left)", previous_symlink, current_target)
        atomic_symlink_switch(current_target, previous_symlink)

    logger.info("Atomically reverting 'current' %s -> %s", current_symlink, prev_target)
    atomic_symlink_switch(prev_target, current_symlink)

    record_activation(
        index_path,
        prev_target,
        expected_hash,
        (recorded or {}).get("attestation"),
        action="rollback",
    )

    logger.info("Rollback SUCCESSFUL. Active release is now %s", prev_target)
    return True


def get_status(current_symlink: Path, previous_symlink: Path, index_file: Optional[str] = None) -> None:
    """Display active pointer state, targets, and recorded activation history."""
    print("=== TranslateGemma Model Release Pointer Status ===")
    if current_symlink.is_symlink() or current_symlink.exists():
        try:
            curr_target = current_symlink.resolve()
            print(f"  [Active Current]  : {current_symlink} -> {curr_target}")
        except Exception as e:
            print(f"  [Active Current]  : {current_symlink} (Error: {e})")
    else:
        print(f"  [Active Current]  : {current_symlink} (NOT CONFIGURED)")

    if previous_symlink.is_symlink() or previous_symlink.exists():
        try:
            prev_target = previous_symlink.resolve()
            print(f"  [Previous/Backup] : {previous_symlink} -> {prev_target}")
        except Exception as e:
            print(f"  [Previous/Backup] : {previous_symlink} (Error: {e})")
    else:
        print(f"  [Previous/Backup] : {previous_symlink} (NONE)")

    index_path = Path(index_file).resolve() if index_file else default_index_path(current_symlink)
    try:
        index = load_release_index(index_path)
    except Exception as e:
        print(f"  [Release Index]   : {index_path} (Error: {e})")
        return

    history: List[Dict[str, Any]] = index.get("history", [])
    print(f"  [Release Index]   : {index_path} ({len(history)} recorded activations)")
    for entry in history[-5:]:
        print(
            f"    - {entry.get('activated_at')} {entry.get('action')}: "
            f"{entry.get('release_dir')} (manifest {entry.get('manifest_sha256')})"
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Promote subparser
    promote_parser = subparsers.add_parser("promote", help="Verify release and atomically switch active pointer.")
    promote_parser.add_argument(
        "--release-dir",
        type=str,
        required=True,
        help="Path to the immutable release directory to promote.",
    )
    promote_parser.add_argument(
        "--current-symlink",
        type=str,
        default="/opt/models/translategemma/current",
        help="Path to active current symlink (default: /opt/models/translategemma/current).",
    )
    promote_parser.add_argument(
        "--previous-symlink",
        type=str,
        default="/opt/models/translategemma/previous",
        help="Path to previous rollback symlink (default: /opt/models/translategemma/previous).",
    )
    promote_parser.add_argument(
        "--attestation-file",
        type=str,
        required=True,
        help=(
            "Release attestation JSON binding merged-quality, degeneration, vLLM smoke, and "
            "deployment preflight results to this release's manifest SHA256."
        ),
    )
    promote_parser.add_argument(
        "--expected-manifest-sha256",
        type=str,
        default=None,
        help="Expected SHA256 digest of merge_manifest.json from a trusted, out-of-band source.",
    )
    promote_parser.add_argument(
        "--trusted-anchor-file",
        type=str,
        default=None,
        help="Path to an external trusted checksum/signature file (must live outside the release directory).",
    )
    promote_parser.add_argument(
        "--expected-image-digest",
        type=str,
        default=None,
        help="vLLM image digest the attestation must have been produced against.",
    )
    promote_parser.add_argument(
        "--max-evidence-age-days",
        type=int,
        default=DEFAULT_MAX_EVIDENCE_AGE_DAYS,
        help=f"Reject gate evidence older than this many days (default: {DEFAULT_MAX_EVIDENCE_AGE_DAYS}).",
    )
    promote_parser.add_argument(
        "--index-file",
        type=str,
        default=None,
        help="Release index path (default: release_index.json beside the current symlink).",
    )
    promote_parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip full file byte hashing (quick metadata check only; not for production promotion).",
    )

    # Rollback subparser
    rollback_parser = subparsers.add_parser("rollback", help="Revert active pointer to the previous known-good release.")
    rollback_parser.add_argument(
        "--current-symlink",
        type=str,
        default="/opt/models/translategemma/current",
        help="Path to active current symlink.",
    )
    rollback_parser.add_argument(
        "--previous-symlink",
        type=str,
        default="/opt/models/translategemma/previous",
        help="Path to previous rollback symlink.",
    )
    rollback_parser.add_argument(
        "--index-file",
        type=str,
        default=None,
        help="Release index path (default: release_index.json beside the current symlink).",
    )
    rollback_parser.add_argument(
        "--trusted-anchor-file",
        type=str,
        default=None,
        help="External trusted anchor for the rollback target; overrides the hash recorded in the release index.",
    )
    rollback_parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip full file byte hashing.",
    )

    # Status subparser
    status_parser = subparsers.add_parser("status", help="Show current pointer mapping status.")
    status_parser.add_argument(
        "--current-symlink",
        type=str,
        default="/opt/models/translategemma/current",
        help="Path to active current symlink.",
    )
    status_parser.add_argument(
        "--previous-symlink",
        type=str,
        default="/opt/models/translategemma/previous",
        help="Path to previous rollback symlink.",
    )
    status_parser.add_argument(
        "--index-file",
        type=str,
        default=None,
        help="Release index path (default: release_index.json beside the current symlink).",
    )

    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.action == "promote":
        success = promote_release(
            release_dir=Path(args.release_dir),
            current_symlink=Path(args.current_symlink),
            previous_symlink=Path(args.previous_symlink),
            attestation_file=args.attestation_file,
            expected_manifest_sha256=args.expected_manifest_sha256,
            trusted_anchor_file=args.trusted_anchor_file,
            expected_image_digest=args.expected_image_digest,
            max_evidence_age_days=args.max_evidence_age_days,
            index_file=args.index_file,
            skip_checksums=args.skip_checksums,
        )
        return 0 if success else 1

    elif args.action == "rollback":
        success = rollback_release(
            current_symlink=Path(args.current_symlink),
            previous_symlink=Path(args.previous_symlink),
            index_file=args.index_file,
            trusted_anchor_file=args.trusted_anchor_file,
            skip_checksums=args.skip_checksums,
        )
        return 0 if success else 1

    elif args.action == "status":
        get_status(
            current_symlink=Path(args.current_symlink),
            previous_symlink=Path(args.previous_symlink),
            index_file=args.index_file,
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
