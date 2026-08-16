#!/usr/bin/env python
"""Copy the shared generation contract into api/ and gateway/, which must run standalone.

api/ and gateway/ are deployed independently, without the rest of the repository on disk,
so they cannot import prompting.py from the project root. They carry byte-identical copies.

Duplication is machine-checked: tests/test_api_vendored_modules.py fails the moment
a copy differs from its source. Edit the root module, run this script, commit all copies.

    uv run python scripts/sync_api_vendored.py          # copy
    uv run python scripts/sync_api_vendored.py --check  # report drift, exit 1
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_DIR = PROJECT_ROOT / "api"
GATEWAY_DIR = PROJECT_ROOT / "gateway"

# Modules standalone deployments need and must never re-implement.
# Encodes the fixes from docs/2026-08-10_adapter_degeneration_analysis.md.
VENDORED_TARGETS = (
    ("prompting.py", API_DIR / "prompting.py"),
    ("model_loading.py", API_DIR / "model_loading.py"),
    ("prompting.py", GATEWAY_DIR / "prompting.py"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit non-zero if any copy is out of date.",
    )
    args = parser.parse_args()

    stale = []
    for source_name, target in VENDORED_TARGETS:
        source = PROJECT_ROOT / source_name
        source_bytes = source.read_bytes()
        if target.exists() and target.read_bytes() == source_bytes:
            continue
        rel_target = target.relative_to(PROJECT_ROOT)
        stale.append(str(rel_target))
        if not args.check:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source_bytes)
            print(f"updated {rel_target}")

    if args.check and stale:
        print(
            f"Out of date: {', '.join(stale)}. "
            "Run: uv run python scripts/sync_api_vendored.py",
            file=sys.stderr,
        )
        return 1
    if not stale:
        print("Vendored modules in api/ and gateway/ are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
