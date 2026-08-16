#!/usr/bin/env python
"""Copy the shared generation contract into api/, which must run standalone.

api/ is deployed on its own, without the rest of the repository on disk, so it
cannot import prompting.py from the project root. It carries a byte-identical
copy instead.

Duplication is normally the wrong answer, and it is only tolerable here because
the copies are machine-checked: tests/test_api_vendored_modules.py fails the
moment a copy differs from its source. Edit the root module, run this script,
commit both.

    uv run python scripts/sync_api_vendored.py          # copy
    uv run python scripts/sync_api_vendored.py --check  # report drift, exit 1
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_DIR = PROJECT_ROOT / "api"

# Modules api/ needs and must never re-implement. prompting.py encodes the
# fixes from docs/2026-08-10_adapter_degeneration_analysis.md, whose failure
# mode is silent. model_loading.py used to be vendored too; api/ serves through
# vLLM now and loads no weights, so it neither ships nor imports torch.
VENDORED_MODULES = ("prompting.py",)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit non-zero if any copy is out of date.",
    )
    args = parser.parse_args()

    stale = []
    for name in VENDORED_MODULES:
        source, target = PROJECT_ROOT / name, API_DIR / name
        source_bytes = source.read_bytes()
        if target.exists() and target.read_bytes() == source_bytes:
            continue
        stale.append(name)
        if not args.check:
            target.write_bytes(source_bytes)
            print(f"updated api/{name}")

    if args.check and stale:
        print(
            f"Out of date: {', '.join(f'api/{name}' for name in stale)}. "
            "Run: uv run python scripts/sync_api_vendored.py",
            file=sys.stderr,
        )
        return 1
    if not stale:
        print("api/ vendored modules are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
