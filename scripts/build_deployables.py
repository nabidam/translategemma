"""Generate the self-contained OpenWebUI deployables from the shared core.

OpenWebUI execs Workspace tools and functions as standalone modules
(backend/open_webui/utils/plugin.py), so neither can import
translation_core from this repo. The deployables are therefore GENERATED:
the core source is inlined right after the front end's front-matter
docstring, replacing the single `from translation_core import *` line.

Layout:
  scripts/translation_core.py      canonical core (importable, unit-tested)
  scripts/openwebui_tool.front     tool front end (front matter + class Tools)
  scripts/openwebui_pipe.front     pipe front end (front matter + class Pipe)
  scripts/openwebui_tool.py        generated: paste into Workspace > Tools
  scripts/openwebui_pipe.py        generated: paste into Workspace > Functions

The builder also re-splices the two code blocks in
docs/OPENWEBUI_INTEGRATION.md (section 2 = tool, section 2b = pipe) so they
stay byte-identical to the generated files.

Usage:  python scripts/build_deployables.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DOC = ROOT / "docs" / "OPENWEBUI_INTEGRATION.md"

IMPORT_LINE = "from translation_core import *"

# front end source -> generated deployable -> (doc anchor, doc end marker)
FRONTS = {
    SCRIPTS / "openwebui_tool.front": (
        SCRIPTS / "openwebui_tool.py",
        "and paste (identical to `scripts/openwebui_tool.py`):\n\n```python\n",
        "\n```\n\n---\n\n## 2b.",
    ),
    SCRIPTS / "openwebui_pipe.front": (
        SCRIPTS / "openwebui_pipe.py",
        "and paste (identical to `scripts/openwebui_pipe.py`):\n\n```python\n",
        "\n```\n\n---\n\n## 3.",
    ),
}


def build(front_source: str, core_source: str) -> str:
    """Inline the core into a front end source. Returns the deployable text."""
    lines = [
        line
        for line in front_source.splitlines()
        if line.strip() != IMPORT_LINE
    ]
    front = "\n".join(lines)
    m = re.match(r'^\s*"""[\s\S]*?"""\s*', front)
    if not m:
        raise SystemExit(f"front end must start with the front-matter docstring:\n{front[:80]!r}")
    header = front[: m.end()]
    rest = front[m.end() :].lstrip("\n")
    return header + core_source.rstrip("\n") + "\n\n" + rest


def splice_doc(text: str, anchor: str, end_marker: str, code: str) -> str:
    a = text.index(anchor)
    start = a + len(anchor)
    end = text.index(end_marker, start)
    return text[:start] + code.rstrip("\n") + text[end:]


def main() -> None:
    core_source = (SCRIPTS / "translation_core.py").read_text()

    doc_text = DOC.read_text()
    changed = {}
    for front_path, (out_path, anchor, end_marker) in FRONTS.items():
        deployable = build(front_path.read_text(), core_source)
        out_path.write_text(deployable)
        # Re-splice the doc block for this deployable (unique per anchor).
        doc_text = splice_doc(doc_text, anchor, end_marker, deployable)
        changed[out_path.name] = len(deployable)

    DOC.write_text(doc_text)
    for name, size in changed.items():
        print(f"generated {name} ({size} chars)")
    print(f"re-spliced {DOC.name} (tool + pipe blocks)")


if __name__ == "__main__":
    main()
