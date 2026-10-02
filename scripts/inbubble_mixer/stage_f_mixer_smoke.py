"""Stage F — feature-flagged realtime mixer smoke (offline chunk path, no devices)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from edge.realtime.inbubble_mixer import mix_inbubble
from scripts.inbubble_mixer.common import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="runs/inbubble_mixer_stage_f")
    p.add_argument("--chunk", type=int, default=192)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    rng = np.random.default_rng(0)
    b = rng.standard_normal(args.chunk).astype(np.float32) * 0.1
    s0 = rng.standard_normal(args.chunk).astype(np.float32) * 0.05
    s1 = (b - s0).astype(np.float32)  # consistent split

    y_id = mix_inbubble(b, [s0, s1], gains=(1.0, 1.0)).y
    y_boost = mix_inbubble(b, [s0, s1], gains=(2.0, 1.0)).y
    y_cut = mix_inbubble(b, [s0, s1], gains=(0.0, 1.0)).y

    # Region gain mapping (same logic as inference_pipeline)
    angles = (-60.0, 30.0)
    center = -60.0
    half = 30.0
    gains = []
    for ang in angles:
        delta = abs(((ang - center + 180.0) % 360.0) - 180.0)
        gains.append(2.0 if delta <= half else 1.0)
    y_region = mix_inbubble(b, [s0, s1], gains=gains).y

    identity_err = float(np.max(np.abs(y_id - b)))
    boost_delta = float(np.sqrt(np.mean((y_boost - b) ** 2)))
    summary = {
        "stage": "F",
        "identity_max_abs_err": identity_err,
        "boost_rms_delta": boost_delta,
        "cut_rms_delta": float(np.sqrt(np.mean((y_cut - b) ** 2))),
        "region_gains": gains,
        "region_rms_delta": float(np.sqrt(np.mean((y_region - b) ** 2))),
        "inference_pipeline_flags": [
            "--inbubble-mixer",
            "--speaker-gains 1,1",
            "--region-center-deg",
            "--stream-angles-deg",
        ],
        "gate": {
            "name": "thin_realtime_mixer_default_passthrough",
            "passed": bool(identity_err < 1e-6 and boost_delta > 0.0),
            "note": "Pass if g=1 is exact pass-through and non-unity gains change output. Wired behind --inbubble-mixer.",
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(__import__("json").dumps(summary, indent=2))
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
