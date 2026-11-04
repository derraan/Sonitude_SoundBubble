"""CLI for offline TrialStore analysis.

Examples (from repo root)::

    python -m tools.tech_testbench.app.analysis --measure-root measure --run-id run_...
    python -m tools.tech_testbench.app.analysis --trial measure/run_x/trial_y
    python -m tools.tech_testbench.app.analysis --measure-root measure \\
        --acoustic-profile windows_pico --runtime cpp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aggregate import analyze_run
from .metrics import analyze_trial_dir, write_trial_analysis


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tech_testbench.analysis",
        description="Offline suppression / polar / FR analysis from TrialStore taps (+ HATS)",
    )
    p.add_argument(
        "--measure-root",
        type=Path,
        default=None,
        help="Root containing measure/<run_id>/<trial_id>/",
    )
    p.add_argument("--run-id", default=None, help="Limit to one run folder name")
    p.add_argument("--trial", type=Path, default=None, help="Single trial directory")
    p.add_argument(
        "--acoustic-profile",
        default=None,
        help="Filter: windows_pico | pi_hats",
    )
    p.add_argument("--host", default=None, help="Filter: windows_testbench | pi_rig | host_target value")
    p.add_argument("--runtime", default=None, help="Filter: cpp | python")
    p.add_argument(
        "--kind",
        default="algorithm",
        help="Trial kind filter (default algorithm; pass 'all' for every kind)",
    )
    p.add_argument("--f-lo", type=float, default=300.0)
    p.add_argument("--f-hi", type=float, default=3400.0)
    p.add_argument("--no-plots", action="store_true", help="Skip PNG generation")
    p.add_argument(
        "--no-per-trial-write",
        action="store_true",
        help="Do not rewrite each trial metrics.json / plots/",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Aggregate output directory override")
    p.add_argument(
        "--rew-fr-json",
        type=Path,
        default=None,
        help="Optional REW export path noted in single-trial metrics (overlay reminder only)",
    )
    p.add_argument(
        "--mic-delay-only",
        action="store_true",
        help="With --trial: only compute relative mic delay (GCC-PHAT) plots/metrics",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    acoustic_profile = args.acoustic_profile
    runtime = args.runtime
    kind = None if args.kind in ("", "all", "*") else args.kind

    if args.trial:
        trial = Path(args.trial)
        if args.mic_delay_only:
            from .mic_delay import analyze_mic_delay_trial

            result = analyze_mic_delay_trial(trial, write_plots=not args.no_plots) or {
                "error": "no multi-ch WAV found for mic delay"
            }
            print(json.dumps(result, indent=2))
            return 0 if "error" not in result else 1
        if args.no_per_trial_write:
            result = analyze_trial_dir(trial, f_lo=args.f_lo, f_hi=args.f_hi)
        else:
            result = write_trial_analysis(
                trial,
                f_lo=args.f_lo,
                f_hi=args.f_hi,
                write_plots=not args.no_plots,
            )
        if args.rew_fr_json is not None:
            result["rew_fr_overlay"] = str(args.rew_fr_json)
            result["rew_note"] = (
                "Overlay is informational; keep windows_pico hardware FR separate "
                "from algorithm taps / pi_hats HATS."
            )
            (trial / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0

    if args.measure_root is None:
        print("error: provide --measure-root or --trial", file=sys.stderr)
        return 2

    summary = analyze_run(
        args.measure_root,
        run_id=args.run_id,
        acoustic_profile=acoustic_profile,
        host=args.host,
        runtime=runtime,
        kind=kind,
        write_per_trial=not args.no_per_trial_write,
        write_plots=not args.no_plots,
        f_lo=args.f_lo,
        f_hi=args.f_hi,
        out_dir=args.out_dir,
    )
    print(json.dumps({"analysis_dir": summary["analysis_dir"], "artifacts": summary["artifacts"], "n_trials": summary["filters"]["n_trials"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
