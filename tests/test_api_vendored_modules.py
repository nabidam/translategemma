"""Fail when api/ or gateway/ vendored copies drift from the root modules they mirror.

api/ and gateway/ ship standalone, so they cannot import prompting.py from the
project root and carry copies instead. Those modules encode the prompt rendering
and stop-token resolution from docs/2026-08-10_adapter_degeneration_analysis.md,
where a mismatch produces fluent output and an unstopped decoder rather than an
error. A copy that quietly falls behind would reintroduce exactly that failure
in production.
"""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDORED_TARGETS = (
    ("prompting.py", PROJECT_ROOT / "api" / "prompting.py"),
    ("model_loading.py", PROJECT_ROOT / "api" / "model_loading.py"),
    ("prompting.py", PROJECT_ROOT / "gateway" / "prompting.py"),
)


@pytest.mark.parametrize("source_name,target_path", VENDORED_TARGETS)
def test_vendored_copy_is_identical_to_root_module(source_name, target_path):
    source = PROJECT_ROOT / source_name
    rel_target = target_path.relative_to(PROJECT_ROOT)
    assert target_path.is_file(), f"{rel_target} is missing; service cannot import it from root."
    assert target_path.read_bytes() == source.read_bytes(), (
        f"{rel_target} differs from {source_name}. The serving service would render prompts or "
        "resolve stop tokens differently from the evaluation harness, which is "
        "silent at run time. Re-sync with: "
        "uv run python scripts/sync_api_vendored.py"
    )
