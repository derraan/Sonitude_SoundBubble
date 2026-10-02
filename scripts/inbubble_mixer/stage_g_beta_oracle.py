"""Stage G — oracle β / distance×angle product steered mixes on vertical-arc data.

Uses promoted b_hat only for identity context; supervision targets are built from
stems + angles (never mixture.wav as DSENet labels). Freezes σ, ρ, θ0 grid for Stage H.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from edge.realtime.inbubble_mixer import si_sdr_np
from scripts.inbubble_mixer.common import (
    DEFAULT_DATASET,
    DEFAULT_DIS_THRESHOLD,
    DEFAULT_RUN,
    DEFAULT_SR,
    ensure_dir,
    infer_bubble,
    iter_samples,
    load_promoted_model,
    write_json,
)
from src.datasets.dsenet_beta import (
    DEFAULT_RHO,
    DEFAULT_SIGMA,
    DEFAULT_THETA0_GRID_DEG,
    BetaConfig,
    product_weights,
    steered_target_from_stems,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET)
    p.add_argument("--run-dir", type=str, default=DEFAULT_RUN)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max-samples", type=int, default=40)
    p.add_argument("--dis-threshold", type=float, default=DEFAULT_DIS_THRESHOLD)
    p.add_argument("--sr", type=int, default=DEFAULT_SR)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--rho", type=float, default=DEFAULT_RHO)
    p.add_argument(
        "--theta0-grid-deg",
        type=str,
        default=",".join(str(x) for x in DEFAULT_THETA0_GRID_DEG),
    )
    p.add_argument("--out-dir", type=str, default="runs/inbubble_mixer_stage_g")
    p.add_argument("--write-audio", action="store_true")
    return p.parse_args()


def parse_grid(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    audio_dir = ensure_dir(out_dir / "audio") if args.write_audio else None
    grid = parse_grid(args.theta0_grid_deg)
    cfg = BetaConfig(sigma=args.sigma, rho=args.rho, theta0_grid_deg=grid)

    model, device = load_promoted_model(args.run_dir)

    rows: list[dict] = []
    follow_ok = 0
    n_samples = 0
    outside_checks = 0
    outside_leak_ok = 0

    for sample in iter_samples(
        args.dataset_root,
        split=args.split,
        max_samples=args.max_samples,
        require_k=2,
        sr=args.sr,
        dis_threshold=args.dis_threshold,
        k_max=2,
    ):
        n_samples += 1
        b_hat = infer_bubble(model, sample.mixture, device)
        T = min(
            b_hat.shape[0],
            sample.oracle_bubble.shape[0],
            sample.inside_stems[0].shape[0],
        )
        b_hat = b_hat[:T]
        stems = [s[:T] for s in sample.inside_stems]
        # Include any outside stems for leakage check (should stay zero weight).
        outside = [s[:T] for s in sample.outside_stems]
        angles = [float(m.get("angle", 0.0)) for m in sample.inside_meta]
        dists = [float(m.get("dis", 0.0)) for m in sample.inside_meta]

        # Front look vs left/right extremes: steered energy should follow θ0.
        z_by_theta: dict[float, np.ndarray] = {}
        for theta0 in grid:
            z, w = steered_target_from_stems(
                stems,
                dists,
                angles,
                theta0_deg=theta0,
                dis_threshold=args.dis_threshold,
                sigma=cfg.sigma,
                rho=cfg.rho,
            )
            z_by_theta[theta0] = z
            # Outside-bubble stems must get w=0 (β never resurrects them).
            outside_checks += 1
            if not outside:
                outside_leak_ok += 1
            else:
                out_meta_d = []
                out_meta_a = []
                for om in range(len(outside)):
                    # load_sample does not keep outside meta; use d > R sentinel.
                    out_meta_d.append(args.dis_threshold + 0.5)
                    out_meta_a.append(0.0)
                w_all = product_weights(
                    dists + out_meta_d,
                    angles + out_meta_a,
                    theta0_deg=theta0,
                    dis_threshold=args.dis_threshold,
                    sigma=cfg.sigma,
                    rho=cfg.rho,
                )
                if float(np.max(np.abs(w_all[len(stems) :]))) < 1e-8:
                    outside_leak_ok += 1

            row = {
                "sample_id": sample.sample_id,
                "theta0_deg": theta0,
                "w0": float(w[0]) if len(w) > 0 else 0.0,
                "w1": float(w[1]) if len(w) > 1 else 0.0,
                "angles": angles,
                "distances": dists,
                "steered_vs_bubble_sisdr": si_sdr_np(z, sample.oracle_bubble[:T]),
                "steered_vs_s0_sisdr": si_sdr_np(z, stems[0]),
                "steered_vs_s1_sisdr": si_sdr_np(z, stems[1]),
                "bubble_vs_oracle_sisdr": si_sdr_np(b_hat, sample.oracle_bubble[:T]),
            }
            rows.append(row)

            if audio_dir is not None and theta0 in (grid[0], 0.0, grid[-1]):
                from src.utils import write_audio_file

                tag = f"{sample.sample_id}_th{int(theta0)}"
                write_audio_file(str(audio_dir / f"{tag}_steered.wav"), z[None, :], args.sr)
                write_audio_file(str(audio_dir / f"{tag}_bubble.wav"), sample.oracle_bubble[:T][None, :], args.sr)

        # Gate: looking toward speaker0 should raise s0 SI-SDR vs looking toward speaker1.
        # Pick θ0 closest to each speaker angle.
        a0, a1 = angles[0], angles[1]
        th_near0 = min(grid, key=lambda th: abs(((th - a0 + 180) % 360) - 180))
        th_near1 = min(grid, key=lambda th: abs(((th - a1 + 180) % 360) - 180))
        if th_near0 != th_near1:
            z0 = z_by_theta[th_near0]
            z1 = z_by_theta[th_near1]
            s0_at_th0 = si_sdr_np(z0, stems[0])
            s0_at_th1 = si_sdr_np(z1, stems[0])
            s1_at_th1 = si_sdr_np(z1, stems[1])
            s1_at_th0 = si_sdr_np(z0, stems[1])
            if s0_at_th0 > s0_at_th1 and s1_at_th1 > s1_at_th0:
                follow_ok += 1

    # Aggregate follow rates
    follow_rate = follow_ok / n_samples if n_samples else 0.0
    outside_rate = outside_leak_ok / outside_checks if outside_checks else 1.0

    frozen = {
        "sigma": cfg.sigma,
        "rho": cfg.rho,
        "theta0_grid_deg": list(cfg.theta0_grid_deg),
        "dis_threshold": args.dis_threshold,
        "note": "Frozen Stage G priors for Stage H directional training.",
    }
    write_json(out_dir / "beta_priors_frozen.json", frozen)

    mean_steered_vs_bubble = float(
        np.nanmean([r["steered_vs_bubble_sisdr"] for r in rows])
    ) if rows else float("nan")

    summary = {
        "stage": "G",
        "run_dir_promote": args.run_dir,
        "dataset_root": args.dataset_root,
        "n_samples": n_samples,
        "beta_priors": frozen,
        "follow_theta0_rate": follow_rate,
        "outside_weight_zero_rate": outside_rate,
        "mean_steered_vs_hard_bubble_sisdr": mean_steered_vs_bubble,
        "gate": {
            "name": "oracle_steered_mixes_follow_theta0",
            "follow_rate": follow_rate,
            "outside_suppressed": outside_rate >= 0.99,
            "passed": bool(
                n_samples > 0
                and follow_rate >= 0.6
                and outside_rate >= 0.99
            ),
            "note": (
                "Pass if product-weighted steered mixes audibly follow θ0 "
                "(speaker nearer look direction wins SI-SDR) and outside weights stay 0."
            ),
        },
    }

    csv_path = out_dir / "beta_oracle_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["sample_id"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Froze priors -> {out_dir / 'beta_priors_frozen.json'}")
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
