#!/usr/bin/env python3
"""Thin CLI entry for offline TrialStore analysis.

From repo root::

    python tools/tech_testbench/scripts/analyze_offline.py --measure-root measure --run-id run_...
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.tech_testbench.app.analysis.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
