#!/usr/bin/env python3
"""Thin CLI entry for C++↔Python offline parity.

From repo root::

    python tools/tech_testbench/scripts/run_parity.py --raw-8ch path/to/raw_8ch.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.measure.parity import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
