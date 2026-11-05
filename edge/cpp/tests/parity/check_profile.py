"""Confirm a C++ profile line satisfies edge/controller/profile_parser.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from edge.controller.profile_parser import parse_profile_line


def main() -> None:
    line = sys.stdin.read().strip()
    parsed = parse_profile_line(line)
    if parsed is None:
        raise SystemExit(f"profile line rejected:\n{line}")
    if "infer_ms" not in parsed:
        raise SystemExit("missing infer_ms")
    print("ok", parsed["infer_ms"])


if __name__ == "__main__":
    main()
