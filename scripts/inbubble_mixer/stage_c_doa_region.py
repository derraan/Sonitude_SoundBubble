"""Stage C — vertical-arc DOA / region-gain resolution probe (headphone layout, not UCA)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SSL_ROOT = Path(r"C:\Users\darre\Sound_Bubble\scripts")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if SSL_ROOT.is_dir() and str(SSL_ROOT) not in sys.path:
    sys.path.insert(0, str(SSL_ROOT))

from scripts.inbubble_mixer.common import (
    DEFAULT_DATASET,
    DEFAULT_DIS_THRESHOLD,
    DEFAULT_SR,
    ensure_dir,
    iter_samples,
    write_json,
)


def headphone_mics_cm() -> np.ndarray:
    """Training vertical-arc geometry (cm→m), matching generate_adaptive_dataset.py."""
    mics = np.array(
        [
            [-12.8, -1.5, 0.0],
            [-10.2, 0.0, 11.3],
            [-3.8, 0.0, 16.9],
            [3.8, 0.0, 16.9],
            [10.6, 0.0, 11.7],
            [13.1, -1.5, 0.7],
        ],
        dtype=np.float64,
    )
    return mics / 100.0


def gcc_phat(sig: np.ndarray, refsig: np.ndarray, fs: int, max_tau: float | None = None) -> tuple[float, np.ndarray]:
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n=n)
    max_shift = int(n // 2)
    if max_tau is not None:
        max_shift = min(int(fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc)) - max_shift)
    tau = shift / float(fs)
    return tau, cc


def estimate_azimuth_l_r(mixture: np.ndarray, fs: int, mics: np.ndarray) -> float:
    """Crude L/R azimuth proxy from ear-side pair (mic0 vs mic5) GCC-PHAT."""
    # channels: expect >=6
    left = mixture[0]
    right = mixture[5]
    # max_tau from inter-mic distance
    dist = np.linalg.norm(mics[0] - mics[5])
    max_tau = dist / 343.0 + 5e-4
    tau, _ = gcc_phat(left, right, fs, max_tau=max_tau)
    # map tau to approximate azimuth in degrees (sin approx): tau ≈ (d/c) sin(theta)
    s = np.clip(tau * 343.0 / dist, -1.0, 1.0)
    return float(np.rad2deg(np.arcsin(s)))


def region_gain_from_angle(angle_deg: float, region_center: float, region_half_width: float, g_in: float, g_out: float) -> float:
    if abs(((angle_deg - region_center + 180) % 360) - 180) <= region_half_width:
        return g_in
    return g_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max-samples", type=int, default=80)
    p.add_argument("--dis-threshold", type=float, default=DEFAULT_DIS_THRESHOLD)
    p.add_argument("--sr", type=int, default=DEFAULT_SR)
    p.add_argument("--out-dir", type=str, default="runs/inbubble_mixer_stage_c")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    mics = headphone_mics_cm()

    rows = []
    for sample in iter_samples(
        args.dataset_root,
        split=args.split,
        max_samples=args.max_samples,
        require_k=2,
        sr=args.sr,
        dis_threshold=args.dis_threshold,
        k_max=2,
    ):
        # Per-source free-field proxy: use mic00 stems panned? Better: estimate from mixture is hard.
        # Use GT angles and also estimate dominant DOA from mixture; compare to loudest inside speaker.
        powers = [float(np.mean(s * s)) for s in sample.inside_stems]
        dom = int(np.argmax(powers))
        gt_ang = float(sample.inside_meta[dom]["angle"])
        est_ang = estimate_azimuth_l_r(sample.mixture, args.sr, mics)
        err = abs(((est_ang - gt_ang + 180) % 360) - 180)

        # Region control usefulness: can we assign different gains to the two inside speakers by angle?
        a0 = float(sample.inside_meta[0]["angle"])
        a1 = float(sample.inside_meta[1]["angle"])
        sep = abs(((a0 - a1 + 180) % 360) - 180)
        # If we center a region on speaker0, does speaker1 fall outside a ±30° region?
        g0 = region_gain_from_angle(a0, a0, 30.0, 2.0, 1.0)
        g1 = region_gain_from_angle(a1, a0, 30.0, 2.0, 1.0)
        separable = (g0 != g1)

        # L/R classification (sign of angle)
        lr_ok = (np.sign(est_ang) == np.sign(gt_ang)) or abs(gt_ang) < 10 or abs(est_ang) < 10

        rows.append(
            {
                "sample_id": sample.sample_id,
                "gt_dom_angle": gt_ang,
                "est_lr_angle": est_ang,
                "abs_err_deg": err,
                "inside_angle_sep_deg": sep,
                "region30_separable": int(separable),
                "lr_sign_match": int(bool(lr_ok)),
            }
        )

    if not rows:
        summary = {"stage": "C", "gate": {"passed": False, "note": "no samples"}}
        write_json(out_dir / "summary.json", summary)
        return 2

    mean_err = float(np.mean([r["abs_err_deg"] for r in rows]))
    med_err = float(np.median([r["abs_err_deg"] for r in rows]))
    lr_acc = float(np.mean([r["lr_sign_match"] for r in rows]))
    region_rate = float(np.mean([r["region30_separable"] for r in rows]))
    mean_sep = float(np.mean([r["inside_angle_sep_deg"] for r in rows]))

    summary = {
        "stage": "C",
        "geometry": "vertical_arc_headphone_xz",
        "n_samples": len(rows),
        "doa_abs_err_mean_deg": mean_err,
        "doa_abs_err_median_deg": med_err,
        "lr_sign_accuracy": lr_acc,
        "region30_separable_rate": region_rate,
        "mean_inside_angle_separation_deg": mean_sep,
        "gate": {
            "name": "useful_LR_region_control",
            "passed": bool(lr_acc >= 0.65 and region_rate >= 0.4),
            "note": (
                "Pass if ear-pair GCC-PHAT gets L/R sign >=65% and >=40% of 2-speaker scenes "
                "are separable by a ±30° angular region. Not a full-azimuth UCA claim."
            ),
        },
    }

    with (out_dir / "doa_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_json(out_dir / "summary.json", summary)
    print(__import__("json").dumps(summary, indent=2))
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
