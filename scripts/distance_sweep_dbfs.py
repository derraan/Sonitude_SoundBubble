"""
LibriSpeech distance sweep with RT60 overlay for a PyTorch run directory.
"""
from __future__ import annotations

import argparse
import math
import os

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import torch

from src.utils import load_torch_pretrained

SR = 24000
DUR_S = 2.0
TARGET_IN_DBFS = -25.0
DISTANCES = np.round(np.arange(3.0, 0.49, -0.1), 2)
RT60_LIST = [0.2, 0.8, 1.0, 1.6]
SPEECH_PATH = r"C:\Users\darre\Sound_Bubble\datasets\_speech_cache\5694-64038-0000.flac"

MIC_OFFSETS_M = np.array(
    [
        [-0.128, -0.015, 0.0],
        [-0.102, 0.0, 0.113],
        [-0.038, 0.0, 0.169],
        [0.038, 0.0, 0.169],
        [0.106, 0.0, 0.117],
        [0.131, -0.015, 0.007],
    ],
    dtype=np.float64,
)

COLORS = {
    0.2: "#F58518",
    0.8: "#54A24B",
    1.0: "#B279A2",
    1.6: "#E45756",
}


def db_fs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    return 20.0 * math.log10(max(rms, 1e-12))


def load_libri_segment(path: str, sr: int, dur_s: float) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    n = int(sr * dur_s)
    if len(y) < n:
        reps = int(np.ceil(n / max(len(y), 1)))
        y = np.tile(y, reps)
    y = y[:n].astype(np.float32)
    peak = float(np.max(np.abs(y)) + 1e-12)
    return (y / peak * 0.3).astype(np.float32)


def simulate_at_distance(distance_m: float, src: np.ndarray, sr: int, rt60_s: float) -> np.ndarray:
    room_dim = [8.0, 8.0, 3.0]
    mic_center = np.array([4.0, 4.0, 1.5])
    mics = MIC_OFFSETS_M + mic_center
    src_pos = mic_center + np.array([0.0, distance_m, 0.0])
    src_pos = np.clip(src_pos, [0.3, 0.3, 0.3], [room_dim[0] - 0.3, room_dim[1] - 0.3, room_dim[2] - 0.3])

    e_absorption, max_order = pra.inverse_sabine(rt60_s, room_dim)
    max_order = int(min(max_order, 80))
    room = pra.ShoeBox(
        room_dim,
        fs=sr,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )
    room.add_microphone_array(mics.T)
    room.add_source(src_pos, signal=src.astype(np.float64))
    room.simulate()
    mix = np.asarray(room.mic_array.signals, dtype=np.float32)
    n = len(src)
    if mix.shape[1] < n:
        mix = np.concatenate([mix, np.zeros((6, n - mix.shape[1]), dtype=np.float32)], axis=1)
    else:
        mix = mix[:, :n]
    return mix


def normalize_mic0_level(mix: np.ndarray, target_dbfs: float) -> np.ndarray:
    cur = db_fs(mix[0])
    gain = 10.0 ** ((target_dbfs - cur) / 20.0)
    return (mix * gain).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--bubble-m", type=float, default=1.5)
    parser.add_argument("--output-prefix", type=str, default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    bubble_m = float(args.bubble_m)
    out_prefix = args.output_prefix or os.path.join(run_dir, "distance_sweep_dbfs_libri_rt60_overlay")
    out_png = out_prefix + ".png"
    out_csv = out_prefix + ".csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl = load_torch_pretrained(run_dir, map_location=device)
    model = pl.model.to(device).eval()
    src = load_libri_segment(SPEECH_PATH, SR, DUR_S)
    print(f"Loaded {run_dir} epoch={pl.epoch} device={device}")
    print(f"Speech={SPEECH_PATH} RT60s={RT60_LIST} target_in={TARGET_IN_DBFS} dBFS")

    results: dict[float, list[float]] = {}
    in_check: dict[float, list[float]] = {}

    for rt60 in RT60_LIST:
        outs = []
        ins = []
        print(f"\n=== RT60={rt60}s ===")
        for d in DISTANCES:
            mix = simulate_at_distance(float(d), src, SR, float(rt60))
            mix = normalize_mic0_level(mix, TARGET_IN_DBFS)
            mix_t = torch.from_numpy(mix).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model({"mixture": mix_t})["output"].squeeze(0).detach().cpu().numpy()
            if out.ndim > 1:
                out = out[0]
            ins.append(db_fs(mix[0]))
            outs.append(db_fs(out))
            print(f"d={d:4.2f}m  in={ins[-1]:7.2f}  out={outs[-1]:7.2f}")
        results[rt60] = outs
        in_check[rt60] = ins

    # Passband ripple on RT60=0.2: peak−peak inside bubble among active passband points.
    rt02 = results.get(0.2)
    if rt02 is not None:
        interior = [lvl for d, lvl in zip(DISTANCES, rt02) if float(d) < bubble_m and lvl > -60.0]
        ripple = (max(interior) - min(interior)) if len(interior) >= 2 else float("nan")
        print(f"\npassband_ripple_0_2_db={ripple:.2f}  (max−min out dBFS for d<{bubble_m}m & out>-60, RT60=0.2s)")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        header = "distance_m," + ",".join([f"out_rt60_{rt60}" for rt60 in RT60_LIST]) + ",input_mic0_dbfs\n"
        f.write(header)
        for i, d in enumerate(DISTANCES):
            row = [f"{d}"] + [f"{results[rt60][i]:.4f}" for rt60 in RT60_LIST] + [f"{in_check[RT60_LIST[0]][i]:.4f}"]
            f.write(",".join(row) + "\n")

    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=140)
    ax.plot(
        DISTANCES,
        in_check[RT60_LIST[0]],
        "o-",
        color="#4C78A8",
        label="Input mic0 (level-matched)",
        markersize=3,
        linewidth=1.5,
    )
    for rt60 in RT60_LIST:
        ax.plot(
            DISTANCES,
            results[rt60],
            "s-",
            color=COLORS[rt60],
            label=f"Model Output RT60: {rt60}s",
            markersize=3.5,
            linewidth=1.8,
        )
    ax.axvline(
        bubble_m,
        color="#333333",
        linestyle="--",
        linewidth=2,
        label=f"Bubble radius = {bubble_m:.2f} m",
    )
    ax.set_xlabel("Source distance (m)")
    ax.set_ylabel("Signal level (dBFS)")
    ax.set_title(f"LibriSpeech distance sweep -- RT60 overlay\n{run_dir} (epoch {pl.epoch}, bubble={bubble_m:.2f} m)")
    ax.set_xlim(float(max(DISTANCES)), float(min(DISTANCES)))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
