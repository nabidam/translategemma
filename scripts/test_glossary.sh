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
# The default target is the whole tests/ directory, which also holds
# test_benchmark_training.py, test_lora_configuration.py, test_split_dataset.py
# and test_translation_benchmark.py -- training-stack tests that need
# yaml/pandas/torch, none of which this runner installs (see above). pytest
# *interrupts* the whole run on a collection error rather than skipping the
# offending file, so without --continue-on-collection-errors a bare
# `scripts/test_glossary.sh` collects zero tests and exits non-zero, which
# reads as "the glossary suite failed" when really nothing ran. The flag makes
# pytest report those four files as collection errors and still run --and
# report on-- everything it could collect, which is the true result of this
# suite. It is passed unconditionally, so it also applies -- harmlessly -- when
# you narrow the run with explicit arguments below.
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
    pytest --continue-on-collection-errors "${@:-tests/}"
