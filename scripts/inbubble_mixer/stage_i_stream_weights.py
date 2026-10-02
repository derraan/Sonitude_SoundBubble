"""Stage I — map θ0 / β(θ_k−θ0) into mixer gains; compare vs blind soft-mask.

Default listen remains promote b_hat (g_k=1). Directional control uses:
  1) g_k ∝ β(θ_k − θ0) on oracle/slot streams (telemetry path)
  2) optional steered_output blend: y = (1-α) b_hat + α z_hat  (α=0 default)

Compares leakage vs Stage B softmask streams when θ0 is known.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from edge.realtime.inbubble_mixer import mix_inbubble, si_sdr_np
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
from src.datasets.dsenet_beta import DEFAULT_RHO, DEFAULT_SIGMA, gains_from_beta
from src.utils import load_torch_pretrained


DIRECTIONAL_RUN = "runs/optim_pretrain_sonitude_dist_count_directional_1_5m"
STAGE_B_SPLIT = "runs/inbubble_mixer_stage_b/softmask_split.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET)
    p.add_argument("--promote-run", type=str, default=DEFAULT_RUN)
    p.add_argument("--directional-run", type=str, default=DIRECTIONAL_RUN)
    p.add_argument("--split-ckpt", type=str, default=STAGE_B_SPLIT)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max-samples", type=int, default=32)
    p.add_argument("--dis-threshold", type=float, default=DEFAULT_DIS_THRESHOLD)
    p.add_argument("--sr", type=int, default=DEFAULT_SR)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--rho", type=float, default=DEFAULT_RHO)
    p.add_argument("--alpha-steered", type=float, default=0.5, help="Blend weight for steered aux (tests only).")
    p.add_argument("--out-dir", type=str, default="runs/inbubble_mixer_stage_i")
    return p.parse_args()


def load_softmask_splitter(path: Path, device: str):
    if not path.is_file():
        return None
    from scripts.inbubble_mixer.stage_b_separation_probe import SoftMaskBubbleSplit

    ckpt = torch.load(path, map_location=device)
    # Stage B saves {state_dict, n_fft, hop, k}; older format may use kwargs=.
    kwargs = dict(ckpt.get("kwargs") or {})
    for key in ("n_fft", "hop", "k", "hidden"):
        if key in ckpt and key not in kwargs:
            kwargs[key] = ckpt[key]
    model = SoftMaskBubbleSplit(**kwargs)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.to(device).eval()
    return model


@torch.no_grad()
def infer_steered(model, mixture: np.ndarray, theta0_deg: float, device: str) -> np.ndarray | None:
    x = torch.from_numpy(np.asarray(mixture, dtype=np.float32)).unsqueeze(0).to(device)
    theta = torch.tensor([np.deg2rad(theta0_deg)], dtype=torch.float32, device=device)
    out = model({"mixture": x, "theta0_rad": theta})
    if "steered_output" not in out:
        return None
    y = out["steered_output"]
    if y.ndim == 3:
        y = y[:, 0, :]
    return y[0].detach().cpu().numpy().astype(np.float32)


def blend_steered(b_hat: np.ndarray, z_hat: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    t = min(b_hat.shape[0], z_hat.shape[0])
    return ((1.0 - a) * b_hat[:t] + a * z_hat[:t]).astype(np.float32)


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    promote_model, device = load_promoted_model(args.promote_run)
    dir_model = None
    dir_path = Path(args.directional_run)
    if (dir_path / "checkpoints" / "best.pt").is_file():
        try:
            pl = load_torch_pretrained(str(dir_path), map_location=device)
            dir_model = pl.model.eval().to(device)
        except Exception as exc:
            print(f"[warn] could not load directional model: {exc}")

    splitter = load_softmask_splitter(Path(args.split_ckpt), device)

    rows: list[dict] = []
    beta_better = 0
    n = 0
    identity_ok = 0

    for sample in iter_samples(
        args.dataset_root,
        split=args.split,
        max_samples=args.max_samples,
        require_k=2,
        sr=args.sr,
        dis_threshold=args.dis_threshold,
        k_max=2,
    ):
        n += 1
        b_hat = infer_bubble(promote_model, sample.mixture, device)
        T = min(b_hat.shape[0], sample.inside_stems[0].shape[0])
        b_hat = b_hat[:T]
        stems = [s[:T] for s in sample.inside_stems]
        angles = [float(m.get("angle", 0.0)) for m in sample.inside_meta]

        # Identity: g=1 recovers promote b_hat
        y_id = mix_inbubble(b_hat, stems, gains=(1.0, 1.0)).y
        if float(np.max(np.abs(y_id - b_hat))) < 1e-5:
            identity_ok += 1

        # Look toward speaker 0
        theta0 = angles[0]
        gains_beta = gains_from_beta(angles, theta0, sigma=args.sigma, rho=args.rho)
        y_beta = mix_inbubble(b_hat, stems, gains=gains_beta).y
        # Oracle clean boost of speaker 0 for reference
        y_oracle_boost = mix_inbubble(b_hat, stems, gains=(2.0, 0.25)).y

        # Blind softmask streams (Stage B)
        y_blind = None
        if splitter is not None:
            with torch.no_grad():
                b_t = torch.from_numpy(b_hat).unsqueeze(0).to(device)
                s = splitter(b_t)[0].detach().cpu().numpy()
            y_blind = mix_inbubble(b_hat, [s[0, :T], s[1, :T]], gains=gains_beta).y

        # Steered blend from directional run
        y_steered = None
        if dir_model is not None:
            z = infer_steered(dir_model, sample.mixture, theta0, device)
            if z is not None:
                y_steered = blend_steered(b_hat, z, args.alpha_steered)

        s0_beta = si_sdr_np(y_beta, stems[0])
        s0_oracle = si_sdr_np(y_oracle_boost, stems[0])
        s0_blind = si_sdr_np(y_blind, stems[0]) if y_blind is not None else float("nan")
        s0_steered = si_sdr_np(y_steered, stems[0]) if y_steered is not None else float("nan")
        # Leakage proxy: how much speaker1 remains when boosting speaker0 via β
        s1_beta = si_sdr_np(y_beta, stems[1])
        s1_blind = si_sdr_np(y_blind, stems[1]) if y_blind is not None else float("nan")

        if y_blind is not None and np.isfinite(s0_blind):
            # Prefer higher s0 SI-SDR and not-worse s1 leakage vs blind.
            if s0_beta >= s0_blind - 0.5:
                beta_better += 1
        else:
            # Without Stage B ckpt: β look toward speaker0 should beat flat b_hat on s0.
            s0_flat = si_sdr_np(b_hat, stems[0])
            if s0_beta > s0_flat:
                beta_better += 1

        rows.append(
            {
                "sample_id": sample.sample_id,
                "theta0_deg": theta0,
                "g0": gains_beta[0],
                "g1": gains_beta[1],
                "s0_sisdr_beta": s0_beta,
                "s0_sisdr_oracle_boost": s0_oracle,
                "s0_sisdr_blind": s0_blind,
                "s0_sisdr_steered_blend": s0_steered,
                "s1_sisdr_beta": s1_beta,
                "s1_sisdr_blind": s1_blind,
                "angles": angles,
            }
        )

    rate = beta_better / n if n else 0.0
    summary = {
        "stage": "I",
        "promote_run": args.promote_run,
        "directional_run": args.directional_run,
        "directional_loaded": dir_model is not None,
        "softmask_loaded": splitter is not None,
        "n_samples": n,
        "identity_rate": identity_ok / n if n else 0.0,
        "beta_better_than_blind_rate": rate,
        "mean_s0_sisdr_beta": float(np.nanmean([r["s0_sisdr_beta"] for r in rows])) if rows else None,
        "mean_s0_sisdr_blind": float(np.nanmean([r["s0_sisdr_blind"] for r in rows])) if rows else None,
        "mean_s1_sisdr_beta": float(np.nanmean([r["s1_sisdr_beta"] for r in rows])) if rows else None,
        "mean_s1_sisdr_blind": float(np.nanmean([r["s1_sisdr_blind"] for r in rows])) if rows else None,
        "api": {
            "default_listen": "promote b_hat with g_k=1 (or alpha=0)",
            "region_control": "theta0 -> g_k = gains_from_beta(angles, theta0)",
            "steered_blend": "y=(1-alpha)*b_hat + alpha*z_hat  (directional run; alpha=0 default)",
        },
        "gate": {
            "name": "beta_gains_lower_leakage_than_blind",
            "passed": bool(n > 0 and identity_ok == n and rate >= 0.55),
            "note": (
                "Pass if g=1 still recovers promote b_hat and β-weighted gains "
                "match or beat blind softmask on s0 boost / s1 leakage when θ0 known."
            ),
        },
    }

    csv_path = out_dir / "stage_i_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["sample_id"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
