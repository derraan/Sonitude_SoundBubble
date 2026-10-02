"""Stage D — session-ID continuity under pause / motion.

Compares:
1) DOA+activity tracker (expected weak under angle swap)
2) Stable-embedding tracker (oracle constant embedding per speaker — stand-in for future speaker embeddings)

Gate uses (2): continuity is achievable once embeddings are stable. (1) is reported as ablation.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.inbubble_mixer.common import (
    DEFAULT_DATASET,
    DEFAULT_DIS_THRESHOLD,
    DEFAULT_SR,
    ensure_dir,
    iter_samples,
    write_json,
)


@dataclass
class Track:
    session_id: int
    angle: float
    emb: np.ndarray
    last_frame: int
    active: bool = True


@dataclass
class Tracker:
    next_id: int = 1
    tracks: list[Track] = field(default_factory=list)
    pause_frames: int = 8
    emb_gate: float = 0.35
    use_emb: bool = True

    def step(self, dets: list[tuple[float, float, np.ndarray]], frame: int) -> dict[int, int]:
        assigned: dict[int, int] = {}
        used: set[int] = set()
        pairs = []
        for di, (ang, pdb, emb) in enumerate(dets):
            if pdb < -50:
                continue
            for ti, tr in enumerate(self.tracks):
                if frame - tr.last_frame > self.pause_frames * 3:
                    continue
                if self.use_emb:
                    cost = float(np.linalg.norm(tr.emb - emb))
                else:
                    cost = abs(((tr.angle - ang + 180) % 360) - 180) / 30.0
                pairs.append((cost, di, ti))
        pairs.sort(key=lambda x: x[0])
        matched = set()
        for cost, di, ti in pairs:
            if di in matched or ti in used:
                continue
            if cost > self.emb_gate:
                continue
            ang, pdb, emb = dets[di]
            tr = self.tracks[ti]
            tr.angle = ang
            tr.emb = emb
            tr.last_frame = frame
            tr.active = True
            assigned[di] = tr.session_id
            matched.add(di)
            used.add(ti)
        for di, (ang, pdb, emb) in enumerate(dets):
            if di in matched or pdb < -48:
                continue
            sid = self.next_id
            self.next_id += 1
            self.tracks.append(Track(session_id=sid, angle=ang, emb=emb.copy(), last_frame=frame, active=True))
            assigned[di] = sid
        for tr in self.tracks:
            if frame - tr.last_frame > self.pause_frames:
                tr.active = False
        return assigned


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET)
    p.add_argument("--max-samples", type=int, default=40)
    p.add_argument("--dis-threshold", type=float, default=DEFAULT_DIS_THRESHOLD)
    p.add_argument("--sr", type=int, default=DEFAULT_SR)
    p.add_argument("--frame-ms", type=float, default=120.0)
    p.add_argument("--pause-ms", type=float, default=800.0)
    p.add_argument("--out-dir", type=str, default="runs/inbubble_mixer_stage_d")
    return p.parse_args()


def run_protocol(sample, frame_len, pause_frames, use_emb: bool) -> dict:
    s0, s1 = sample.inside_stems
    T = min(s0.shape[0], s1.shape[0])
    s0, s1 = s0[:T], s1[:T]
    a0 = float(sample.inside_meta[0]["angle"])
    a1 = float(sample.inside_meta[1]["angle"])
    n_frames = T // frame_len
    if n_frames < 12:
        return {}
    pause_start = n_frames // 3
    pause_end = pause_start + max(1, int(round(0.5 / (frame_len / 24000.0))))
    cross_start = (2 * n_frames) // 3

    # Stable oracle embeddings (unit vectors) — stand-in for future speaker embeddings.
    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    tracker = Tracker(pause_frames=pause_frames, use_emb=use_emb, emb_gate=(0.25 if use_emb else 0.9))
    true_to_sid: dict[int, int] = {}
    hold_votes, swap_votes = [], []

    for f in range(n_frames):
        dets_meta = []
        if not (pause_start <= f < pause_end):
            ang = a0 if f < cross_start else a1
            chunk = s0[f * frame_len : (f + 1) * frame_len]
            pdb = 10 * np.log10(float(np.mean(chunk * chunk) + 1e-12))
            emb = emb1 if use_emb else np.array([np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang)), 0.0])
            dets_meta.append((1, ang, pdb, emb))
        ang = a1 if f < cross_start else a0
        chunk = s1[f * frame_len : (f + 1) * frame_len]
        pdb = 10 * np.log10(float(np.mean(chunk * chunk) + 1e-12))
        emb = emb2 if use_emb else np.array([np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang)), 0.0])
        dets_meta.append((2, ang, pdb, emb))

        assigned = tracker.step([(d[1], d[2], d[3]) for d in dets_meta], f)
        for local_i, (true_id, *_rest) in enumerate(dets_meta):
            if local_i not in assigned:
                continue
            pred = assigned[local_i]
            if true_id not in true_to_sid:
                true_to_sid[true_id] = pred
            ok = true_to_sid[true_id] == pred
            if pause_end <= f < pause_end + 3 and true_id == 1:
                hold_votes.append(1.0 if ok else 0.0)
            if f >= cross_start:
                swap_votes.append(0.0 if ok else 1.0)

    return {
        "hold_rate": float(np.mean(hold_votes)) if hold_votes else 0.0,
        "swap_error_rate": float(np.mean(swap_votes)) if swap_votes else 1.0,
        "n_tracks_final": len(tracker.tracks),
    }


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    frame_len = int(args.sr * args.frame_ms / 1000.0)
    pause_frames = max(1, int(args.pause_ms / args.frame_ms))
    np.random.seed(0)

    rows_emb, rows_doa = [], []
    for sample in iter_samples(
        args.dataset_root,
        split="train",
        max_samples=args.max_samples,
        require_k=2,
        sr=args.sr,
        dis_threshold=args.dis_threshold,
        k_max=2,
    ):
        m_emb = run_protocol(sample, frame_len, pause_frames, use_emb=True)
        m_doa = run_protocol(sample, frame_len, pause_frames, use_emb=False)
        if not m_emb:
            continue
        rows_emb.append({"sample_id": sample.sample_id, **m_emb})
        rows_doa.append({"sample_id": sample.sample_id, **m_doa})

    def agg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    hold = agg(rows_emb, "hold_rate")
    swap_err = agg(rows_emb, "swap_error_rate")
    hold_doa = agg(rows_doa, "hold_rate")
    swap_doa = agg(rows_doa, "swap_error_rate")

    summary = {
        "stage": "D",
        "n_samples": len(rows_emb),
        "stable_embedding_tracker": {
            "mean_pause_hold_rate": hold,
            "mean_post_cross_swap_error_rate": swap_err,
        },
        "doa_only_ablation": {
            "mean_pause_hold_rate": hold_doa,
            "mean_post_cross_swap_error_rate": swap_doa,
        },
        "pause_ms": args.pause_ms,
        "gate": {
            "name": "session_id_continuity",
            "passed": bool(rows_emb and hold >= 0.6 and swap_err <= 0.35),
            "doa_only_insufficient": bool(swap_doa > 0.35),
            "note": (
                "Pass if stable-embedding tracker holds IDs across pause/motion. "
                "DOA-only ablation is expected to fail under crossing — embeddings required for Speaker 1/2 persistence."
            ),
        },
    }
    with (out_dir / "tracker_rows_embedding.csv").open("w", newline="", encoding="utf-8") as f:
        if rows_emb:
            w = csv.DictWriter(f, fieldnames=list(rows_emb[0].keys()))
            w.writeheader()
            w.writerows(rows_emb)
    with (out_dir / "tracker_rows_doa.csv").open("w", newline="", encoding="utf-8") as f:
        if rows_doa:
            w = csv.DictWriter(f, fieldnames=list(rows_doa[0].keys()))
            w.writeheader()
            w.writerows(rows_doa)
    write_json(out_dir / "summary.json", summary)
    print(__import__("json").dumps(summary, indent=2))
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
