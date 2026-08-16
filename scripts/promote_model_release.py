#!/usr/bin/env python
"""Atomic release promotion and rollback manager for TranslateGemma model releases.

Guarantees that active serving pointers (/opt/models/translategemma/current) are
switched ONLY after full export verification, manifest authenticity validation,
and integrity gates pass.

Rollbacks are executed as atomic pointer switches to the recorded previous release.

Usage:
  # Promote a verified release:
  python scripts/promote_model_release.py promote \
      --release-dir /opt/models/translategemma/releases/tg-merged-v1 \
      --current-symlink /opt/models/translategemma/current \
      --previous-symlink /opt/models/translategemma/previous \
      --expected-manifest-sha256 <hex_digest>

  # Roll back to the previous known-good release:
  python scripts/promote_model_release.py rollback \
      --current-symlink /opt/models/translategemma/current \
      --previous-symlink /opt/models/translategemma/previous

  # Check active and previous release status:
  python scripts/promote_model_release.py status \
      --current-symlink /opt/models/translategemma/current \
      --previous-symlink /opt/models/translategemma/previous
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from verify_model_export import verify_export

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("promote_model_release")


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


def promote_release(
    release_dir: Path,
    current_symlink: Path,
    previous_symlink: Path,
    expected_manifest_sha256: Optional[str] = None,
    trusted_anchor_file: Optional[str] = None,
    skip_checksums: bool = False,
) -> bool:
    """Verify release against all gates and atomically switch current pointer."""
    release_dir = release_dir.resolve()
    if not release_dir.is_dir():
        logger.error("Release directory does not exist: %s", release_dir)
        return False

    logger.info("Executing pre-promotion verification gate on %s...", release_dir)
    is_valid = verify_export(
        str(release_dir),
        skip_checksums=skip_checksums,
        expected_manifest_sha256=expected_manifest_sha256,
        trusted_anchor_file=trusted_anchor_file,
    )
    if not is_valid:
        logger.error("Promotion gate FAILED on %s. Active serving pointer NOT changed.", release_dir)
        return False

    old_current_target: Optional[Path] = None
    if current_symlink.is_symlink() or current_symlink.exists():
        try:
            old_current_target = current_symlink.resolve()
            logger.info("Current active release before promotion: %s", old_current_target)
        except Exception as e:
            logger.warning("Could not resolve current symlink: %s", e)

    # 1. Atomically switch 'current' pointer to verified release
    logger.info("Atomically switching 'current' symlink %s -> %s", current_symlink, release_dir)
    atomic_symlink_switch(release_dir, current_symlink)

    # 2. Update 'previous' pointer for instant rollback if old target existed and differed
    if old_current_target and old_current_target.exists() and old_current_target != release_dir:
        logger.info("Updating 'previous' rollback pointer %s -> %s", previous_symlink, old_current_target)
        atomic_symlink_switch(old_current_target, previous_symlink)

    logger.info("Release promotion SUCCESSFUL. Active release: %s", release_dir)
    return True


def rollback_release(
    current_symlink: Path,
    previous_symlink: Path,
    skip_checksums: bool = False,
) -> bool:
    """Atomically roll back current pointer to the last-known-good previous release."""
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

    logger.info("Verifying previous release target before rollback: %s", prev_target)
    is_valid = verify_export(str(prev_target), skip_checksums=skip_checksums)
    if not is_valid:
        logger.error("Previous release failed integrity check: %s. Rollback aborted.", prev_target)
        return False

    # Atomically swap
    logger.info("Atomically reverting 'current' %s -> %s", current_symlink, prev_target)
    atomic_symlink_switch(prev_target, current_symlink)
    logger.info("Rollback SUCCESSFUL. Active release is now %s", prev_target)
    return True


def get_status(current_symlink: Path, previous_symlink: Path) -> None:
    """Display active pointer state and targets."""
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
        "--expected-manifest-sha256",
        type=str,
        default=None,
        help="Expected SHA256 digest of merge_manifest.json from trusted release store.",
    )
    promote_parser.add_argument(
        "--trusted-anchor-file",
        type=str,
        default=None,
        help="Path to external trusted checksum/signature file containing expected manifest hash.",
    )
    promote_parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip full file byte hashing (quick metadata check only).",
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

    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.action == "promote":
        success = promote_release(
            release_dir=Path(args.release_dir),
            current_symlink=Path(args.current_symlink),
            previous_symlink=Path(args.previous_symlink),
            expected_manifest_sha256=args.expected_manifest_sha256,
            trusted_anchor_file=args.trusted_anchor_file,
            skip_checksums=args.skip_checksums,
        )
        return 0 if success else 1

    elif args.action == "rollback":
        success = rollback_release(
            current_symlink=Path(args.current_symlink),
            previous_symlink=Path(args.previous_symlink),
            skip_checksums=args.skip_checksums,
        )
        return 0 if success else 1

    elif args.action == "status":
        get_status(
            current_symlink=Path(args.current_symlink),
            previous_symlink=Path(args.previous_symlink),
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
