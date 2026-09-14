"""Guard the generated-deployable pipeline (scripts/build_deployables.py).

OpenWebUI execs the tool and pipe as standalone modules, so the deployables
(scripts/openwebui_tool.py, scripts/openwebui_pipe.py) are GENERATED from
scripts/translation_core.py + the two *.front files, and the two code blocks
in docs/OPENWEBUI_INTEGRATION.md are re-spliced to stay byte-identical.
These tests fail if any of those copies drift from the canonical sources.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
DOC = PROJECT_ROOT / "docs" / "OPENWEBUI_INTEGRATION.md"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_deployables as bd


def test_generated_tool_matches_build():
    deployable = bd.build(
        (SCRIPTS / "openwebui_tool.front").read_text(),
        (SCRIPTS / "translation_core.py").read_text(),
    )
    assert (SCRIPTS / "openwebui_tool.py").read_text() == deployable


def test_generated_pipe_matches_build():
    deployable = bd.build(
        (SCRIPTS / "openwebui_pipe.front").read_text(),
        (SCRIPTS / "translation_core.py").read_text(),
    )
    assert (SCRIPTS / "openwebui_pipe.py").read_text() == deployable


def _doc_block(path_name: str) -> str:
    text = DOC.read_text()
    anchor = f"and paste (identical to `{path_name}`):\n\n```python\n"
    start = text.index(anchor) + len(anchor)
    end = text.index("\n```\n", start)
    return text[start:end]


def test_doc_tool_block_byte_identical():
    assert _doc_block("scripts/openwebui_tool.py") == (
        SCRIPTS / "openwebui_tool.py"
    ).read_text().rstrip("\n")


def test_doc_pipe_block_byte_identical():
    assert _doc_block("scripts/openwebui_pipe.py") == (
        SCRIPTS / "openwebui_pipe.py"
    ).read_text().rstrip("\n")


def test_deployables_are_self_contained():
    # No deployable may import translation_core: OpenWebUI cannot resolve it.
    for name in ("openwebui_tool.py", "openwebui_pipe.py"):
        source = (SCRIPTS / name).read_text()
        assert "from translation_core import" not in source
        assert "import translation_core" not in source
