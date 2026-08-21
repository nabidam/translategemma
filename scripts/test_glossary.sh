#!/usr/bin/env bash
# Run the glossary test suite in an isolated environment.
#
# Deliberately --no-project: the root pyproject pins torch 2.8.0 and CUDA
# wheels for the training stack, and syncing that to run a pure-Python test
# suite is minutes of download for no benefit -- on a machine without a GPU it
# may not resolve at all.
#
# Nothing the glossary imports at module level needs any of it. api/ ships
# standalone, and api/translator.py imports transformers lazily inside
# functions, so the suite runs on the web stack alone.
#
# api/requirements.txt remains the source of truth for what production runs;
# this list only has to be enough to import the modules under test.
#
# Usage:
#   scripts/test_glossary.sh                       # whole glossary suite
#   scripts/test_glossary.sh tests/test_x.py -v    # one file
set -euo pipefail

cd "$(dirname "$0")/.."

exec uv run --no-project \
    --with pytest \
    --with pytest-asyncio \
    --with httpx \
    --with fastapi \
    --with pydantic \
    --with pydantic-settings \
    --with "sqlalchemy[asyncio]" \
    --with aiosqlite \
    pytest "${@:-tests/}"
