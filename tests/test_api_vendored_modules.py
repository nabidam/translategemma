"""Fail when api/'s vendored copies drift from the root modules they mirror.

api/ ships standalone, so it cannot import prompting.py from the project root
and carries a copy instead. That module encodes the prompt rendering and
stop-token resolution from
docs/2026-08-10_adapter_degeneration_analysis.md, where a mismatch produces
fluent output and an unstopped decoder rather than an error. A copy that quietly
falls behind would reintroduce exactly that failure in production while every
other test still passes.
"""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDORED_MODULES = ("prompting.py",)


@pytest.mark.parametrize("name", VENDORED_MODULES)
def test_api_copy_is_identical_to_root_module(name):
    source = PROJECT_ROOT / name
    target = PROJECT_ROOT / "api" / name
    assert target.is_file(), f"api/{name} is missing; api/ cannot import it from the root."
    assert target.read_bytes() == source.read_bytes(), (
        f"api/{name} differs from {name}. The serving API would render prompts or "
        "resolve stop tokens differently from the evaluation harness, which is "
        "silent at run time. Re-sync with: "
        "uv run python scripts/sync_api_vendored.py"
    )
