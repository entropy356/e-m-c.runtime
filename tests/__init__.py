"""Test package bootstrap: makes src/ and the repo root importable."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for candidate in (_REPO_ROOT / "src", _REPO_ROOT):
    encoded = str(candidate)
    if encoded not in sys.path:
        sys.path.insert(0, encoded)
