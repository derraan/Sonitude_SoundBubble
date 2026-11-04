"""Offline analysis from taps (+ optional HATS / REW overlays).

Canonical implementation lives in
``tools.tech_testbench.app.analysis``. This module re-exports helpers and
keeps a thin CLI so existing ``python -m tools.measure.analyze`` calls work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Prefer the tech_testbench analysis package.
try:
    from tools.tech_testbench.app.analysis.audio import band_level_db, suppression_db
    from tools.tech_testbench.app.analysis.aggregate import (
        analyze_run,
        iter_trials,
        write_aggregate_plots,
    )
    from tools.tech_testbench.app.analysis.metrics import (
        analyze_trial_dir,
        write_trial_analysis,
    )
except ImportError:  # pragma: no cover — minimal fallback
    import numpy as np

    try:
        from scipy.io.wavfile import read as wavread
    except ImportError:
        wavread = None

    def band_level_db(x, sr, f_lo=300.0, f_hi=3400.0):  # type: ignore[no-redef]
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return -120.0
        n = int(2 ** np.ceil(np.log2(max(256, x.size))))
        w = np.hanning(x.size)
        spec = np.fft.rfft(x * w, n=n)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        power = float(np.mean(np.abs(spec[mask]) ** 2) + 1e-20) if np.any(mask) else float(np.mean(x * x) + 1e-20)
        return 10.0 * np.log10(power)

    def suppression_db(dry, wet, sr, f_lo=300.0, f_hi=3400.0):  # type: ignore[no-redef]
        n = min(dry.size, wet.size)
        dry_l = band_level_db(dry[:n], sr, f_lo, f_hi)
        wet_l = band_level_db(wet[:n], sr, f_lo, f_hi)
        return {"dry_band_db": dry_l, "wet_band_db": wet_l, "suppression_db": dry_l - wet_l, "f_lo": f_lo, "f_hi": f_hi}

    def analyze_trial_dir(trial_dir):  # type: ignore[no-redef]
        raise RuntimeError("tech_testbench analysis package required")

    def write_trial_analysis(*a, **k):  # type: ignore[no-redef]
        raise RuntimeError("tech_testbench analysis package required")

    def analyze_run(*a, **k):  # type: ignore[no-redef]
        raise RuntimeError("tech_testbench analysis package required")

    def iter_trials(*a, **k):  # type: ignore[no-redef]
        return []

    def write_aggregate_plots(*a, **k):  # type: ignore[no-redef]
        return {}


def write_suppression_vs_distance_csv(
    measure_root: Path,
    out_csv: Path,
    *,
    acoustic_profile: str | None = None,
    host: str | None = None,
    runtime: str | None = None,
) -> Path:
    """Compat wrapper — writes aggregate CSV via analyze_run."""
    summary = analyze_run(
        measure_root,
        acoustic_profile=acoustic_profile,
        host=host,
        runtime=runtime,
        kind=None,
        write_per_trial=False,
        write_plots=False,
        out_dir=Path(out_csv).parent,
    )
    # analyze_run always writes suppression_vs_distance.csv into out_dir
    src = Path(summary["artifacts"]["suppression_vs_distance_csv"])
    out_csv = Path(out_csv)
    if src.resolve() != out_csv.resolve():
        out_csv.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return out_csv


def main(argv: list[str] | None = None) -> int:
    # Delegate to the richer CLI when available.
    try:
        from tools.tech_testbench.app.analysis.cli import main as rich_main

        return rich_main(argv)
    except ImportError:
        pass

    p = argparse.ArgumentParser(description="Offline measure analysis from taps / HATS")
    p.add_argument("--measure-root", type=Path, required=True)
    p.add_argument("--trial", type=Path, default=None)
    p.add_argument("--acoustic-profile", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--runtime", default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args(argv)
    if args.trial:
        print(json.dumps(analyze_trial_dir(args.trial), indent=2))
        return 0
    out = args.out_csv or (args.measure_root / "suppression_vs_distance.csv")
    path = write_suppression_vs_distance_csv(
        args.measure_root,
        out,
        acoustic_profile=args.acoustic_profile,
        host=args.host,
        runtime=args.runtime,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
