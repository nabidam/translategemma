"""Make api/ importable from the test suite.

api/ ships standalone and has no package metadata, so its modules are imported
by putting the directory on sys.path rather than through an installed package.
tests/test_generation_chat_template.py does the same for the repository root.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_ROOT = PROJECT_ROOT / "api"

for path in (PROJECT_ROOT, API_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
