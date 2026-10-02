"""Stage A — oracle mix-rule + leakage sensitivity on promoted b_hat."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from edge.realtime.inbubble_mixer import apply_leakage, mix_inbubble, si_sdr_np
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET)
    p.add_argument("--run-dir", type=str, default=DEFAULT_RUN)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max-samples", type=int, default=40)
    p.add_argument("--dis-threshold", type=float, default=DEFAULT_DIS_THRESHOLD)
    p.add_argument("--sr", type=int, default=DEFAULT_SR)
    p.add_argument("--gains", type=str, default="1,1;0.25,1;2.0,1;1,0.25;1,2.0;0,1;1,0")
    p.add_argument("--leakage", type=str, default="0,0.1,0.25,0.5,1.0")
    p.add_argument("--out-dir", type=str, default="runs/inbubble_mixer_stage_a")
    p.add_argument("--write-audio", action="store_true")
    return p.parse_args()


def parse_gain_sets(s: str) -> list[tuple[float, ...]]:
    sets = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        sets.append(tuple(float(x) for x in part.split(",")))
    return sets


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    audio_dir = ensure_dir(out_dir / "audio") if args.write_audio else None

    model, device = load_promoted_model(args.run_dir)
    gain_sets = parse_gain_sets(args.gains)
    leaks = [float(x) for x in args.leakage.split(",") if x.strip()]

    rows: list[dict] = []
    identity_ok = 0
    n_samples = 0

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
        T = min(b_hat.shape[0], sample.oracle_bubble.shape[0], sample.inside_stems[0].shape[0])
        b_hat = b_hat[:T]
        oracle_b = sample.oracle_bubble[:T]
        s_oracle = [s[:T] for s in sample.inside_stems]

        # Identity: g=1 recovers b_hat
        y_id = mix_inbubble(b_hat, s_oracle, gains=(1.0, 1.0)).y
        id_err = float(np.max(np.abs(y_id - b_hat)))
        if id_err < 1e-5:
            identity_ok += 1

        bubble_sisdr = si_sdr_np(b_hat, oracle_b)

        for leak in leaks:
            s_leak = apply_leakage(s_oracle, leak)
            for gains in gain_sets:
                if len(gains) != 2:
                    continue
                # Oracle-base control (isolates mix-rule + leakage; not limited by model bubble SI-SDR)
                y_oracle_clean = mix_inbubble(oracle_b, s_oracle, gains=gains).y
                y_oracle_leaky = mix_inbubble(oracle_b, s_leak, gains=gains).y
                # Model-base with oracle streams (what Stage B must eventually approach)
                y_model_clean = mix_inbubble(b_hat, s_oracle, gains=gains).y
                y_model_leaky = mix_inbubble(b_hat, s_leak, gains=gains).y

                row = {
                    "sample_id": sample.sample_id,
                    "leak": leak,
                    "g0": gains[0],
                    "g1": gains[1],
                    "identity_max_abs_err": id_err,
                    "bubble_sisdr_vs_oracle_b": bubble_sisdr,
                    "oracle_ctrl_sisdr_leaky_vs_clean": si_sdr_np(y_oracle_leaky, y_oracle_clean),
                    "model_ctrl_sisdr_leaky_vs_clean": si_sdr_np(y_model_leaky, y_model_clean),
                    "model_vs_oracle_ctrl_sisdr": si_sdr_np(y_model_clean, y_oracle_clean),
                    "s0_sisdr_at_gains": si_sdr_np(y_oracle_clean, s_oracle[0]),
                    "s1_sisdr_at_gains": si_sdr_np(y_oracle_clean, s_oracle[1]),
                    "angles": [float(m.get("angle", float("nan"))) for m in sample.inside_meta],
                    "distances": [float(m.get("dis", float("nan"))) for m in sample.inside_meta],
                }
                rows.append(row)

                if audio_dir is not None and leak == 0.0 and gains in ((1.0, 1.0), (2.0, 1.0), (0.25, 1.0)):
                    from src.utils import write_audio_file

                    tag = f"{sample.sample_id}_g{gains[0]}_{gains[1]}"
                    write_audio_file(str(audio_dir / f"{tag}_bhat.wav"), b_hat[None, :], args.sr)
                    write_audio_file(str(audio_dir / f"{tag}_y.wav"), y_model_clean[None, :], args.sr)

    # Aggregate leakage curves (exclude identity gains)
    by_leak: dict[float, list[float]] = {L: [] for L in leaks}
    boost_s0 = []
    cut_s0 = []
    bubble_scores = []
    for r in rows:
        if r["leak"] == 0.0 and (r["g0"], r["g1"]) == (1.0, 1.0):
            bubble_scores.append(r["bubble_sisdr_vs_oracle_b"])
        if (r["g0"], r["g1"]) == (1.0, 1.0):
            continue
        by_leak[r["leak"]].append(r["oracle_ctrl_sisdr_leaky_vs_clean"])
        if r["leak"] == 0.0 and (r["g0"], r["g1"]) == (2.0, 1.0):
            boost_s0.append(r["s0_sisdr_at_gains"])
        if r["leak"] == 0.0 and (r["g0"], r["g1"]) == (0.25, 1.0):
            cut_s0.append(r["s0_sisdr_at_gains"])

    leak0 = float(np.nanmean(by_leak[0.0])) if by_leak[0.0] else float("nan")
    leak_hi = float(np.nanmean(by_leak[max(leaks)])) if by_leak[max(leaks)] else float("nan")
    mean_boost_s0 = float(np.nanmean(boost_s0)) if boost_s0 else float("nan")
    mean_cut_s0 = float(np.nanmean(cut_s0)) if cut_s0 else float("nan")

    summary = {
        "stage": "A",
        "run_dir": args.run_dir,
        "dataset_root": args.dataset_root,
        "n_samples": n_samples,
        "identity_recoveries": identity_ok,
        "identity_rate": (identity_ok / n_samples) if n_samples else 0.0,
        "mean_bubble_sisdr": float(np.nanmean(bubble_scores)) if bubble_scores else None,
        "oracle_ctrl_sisdr_vs_leak": {
            str(L): {
                "mean": float(np.nanmean(by_leak[L])) if by_leak[L] else None,
                "p25": float(np.nanpercentile(by_leak[L], 25)) if by_leak[L] else None,
                "p75": float(np.nanpercentile(by_leak[L], 75)) if by_leak[L] else None,
                "n": len(by_leak[L]),
            }
            for L in leaks
        },
        "mean_s0_sisdr_boost_g2": mean_boost_s0,
        "mean_s0_sisdr_cut_g0p25": mean_cut_s0,
        "gate": {
            "name": "oracle_streams_enable_useful_control",
            "identity_ok": identity_ok == n_samples and n_samples > 0,
            "leak0_ctrl_sisdr": leak0,
            "leak_high_ctrl_sisdr": leak_hi,
            "boost_beats_cut_on_s0": bool(mean_boost_s0 > mean_cut_s0),
            "passed": bool(
                n_samples > 0
                and identity_ok == n_samples
                and (leak0 > 40.0 or not np.isfinite(leak0))  # leak=0 ≈ perfect
                and mean_boost_s0 > mean_cut_s0
                and leak_hi < leak0 - 5.0
            ),
            "note": (
                "Pass if g=1 recovers b_hat, oracle leak=0 control matches clean mix, "
                "boosting speaker0 raises s0 SI-SDR vs cutting it, and leakage degrades control."
            ),
        },
    }

    csv_path = out_dir / "oracle_mix_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["sample_id"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    write_json(out_dir / "summary.json", summary)
    print(json_dumps(summary))
    print(f"Wrote {csv_path}")
    return 0 if summary["gate"]["passed"] else 2


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
