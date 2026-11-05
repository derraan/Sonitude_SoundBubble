"""Redirect ``python -m tools.tech_testbench.measure`` → ``tools.measure``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.measure.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
