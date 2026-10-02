"""Export epoch-progress CSV/JSON + PNG plots for promoted Sonitude model."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROM = Path(r"C:\Users\darre\Sound_Bubble_optim_distance\runs\optim_pretrain_sonitude_dist_count_1_5m")
SUP_SWEEP = Path(
    r"C:\Users\darre\Sound_Bubble_optim_distance\runs\count_distance_head_optim_1_5m_suppress_v1\val_distance_sweep"
)


def avg(mv, epoch: int, name: str) -> float | None:
    if name not in mv[epoch]:
        return None
    e = mv[epoch][name]
    return float(e["epoch"] / max(e["num_elements"], 1))


def series(mv, epochs, name: str) -> list[float]:
    out = []
    for e in epochs:
        v = avg(mv, e, name)
        if v is not None:
            out.append(v)
    return out


def series_aligned(mv, epochs, name: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for e in epochs:
        v = avg(mv, e, name)
        if v is not None:
            xs.append(e)
            ys.append(v)
    return xs, ys


def main() -> None:
    out = PROM / "epoch_progress"
    sweep_dst = PROM / "val_distance_sweep"
    out.mkdir(parents=True, exist_ok=True)
    sweep_dst.mkdir(parents=True, exist_ok=True)
    if SUP_SWEEP.exists():
        for p in SUP_SWEEP.glob("*"):
            (sweep_dst / p.name).write_bytes(p.read_bytes())

    ckpt = torch.load(PROM / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    best_ep = int(
        torch.load(PROM / "checkpoints" / "best.pt", map_location="cpu", weights_only=False).get(
            "current_epoch"
        )
    )
    mv = ckpt["metric_values"]
    epochs = sorted(int(k) for k in mv.keys())
    last_ep = int(ckpt.get("current_epoch"))

    metric_names = [
        "train/loss",
        "val/loss",
        "train/audio_loss",
        "val/audio_loss",
        "train/count_loss",
        "val/count_loss",
        "train/count_acc",
        "val/count_acc",
        "train/spk_distance_loss",
        "val/spk_distance_loss",
        "train/spk_distance_mae",
        "val/spk_distance_mae",
        "train/spk_activity_loss",
        "val/spk_activity_loss",
        "train/si_sdr_i",
        "val/si_sdr_i",
        "train/snr_i",
        "val/snr_i",
        "val/sweep_out_dbfs_0_5m",
        "val/sweep_out_dbfs_0_8m",
        "val/sweep_out_dbfs_1_0m",
        "val/sweep_out_dbfs_1_2m",
        "val/sweep_out_dbfs_1_5m",
        "val/sweep_out_dbfs_2_0m",
        "val/sweep_out_dbfs_3_0m",
        "val/sweep_passband_minus_far_db",
        "val/sweep_edge_drop_1_5_to_2_0_db",
        "val/sweep_passband_ripple_0_2_db",
    ]
    rows = []
    for ep in epochs:
        row = {"epoch": ep}
        for n in metric_names:
            if n in mv[ep]:
                row[n.replace("/", "_")] = avg(mv, ep, n)
        rows.append(row)

    fieldnames = ["epoch"] + [n.replace("/", "_") for n in metric_names]
    with (out / "metrics_by_epoch.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    payload = {"best_epoch": best_ep, "last_epoch": last_ep, "rows": rows}
    (out / "metrics_by_epoch.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def savefig(name: str) -> None:
        plt.tight_layout()
        plt.savefig(out / name, dpi=140)
        plt.close()
        print("wrote", name)

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, [avg(mv, e, "train/loss") for e in epochs], label="train/loss")
    plt.plot(epochs, [avg(mv, e, "val/loss") for e in epochs], label="val/loss")
    plt.axvline(best_ep, color="gray", ls="--", lw=1, label=f"best ep {best_ep}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("optim_pretrain_sonitude_dist_count_1_5m — train/val loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("loss_train_val.png")

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, [avg(mv, e, "train/si_sdr_i") for e in epochs], label="train SI-SDRi")
    plt.plot(epochs, [avg(mv, e, "val/si_sdr_i") for e in epochs], label="val SI-SDRi")
    plt.plot(epochs, [avg(mv, e, "val/snr_i") for e in epochs], label="val SNRi")
    plt.axvline(best_ep, color="gray", ls="--", lw=1, label=f"best ep {best_ep}")
    plt.xlabel("Epoch")
    plt.ylabel("dB")
    plt.title("SI-SDRi / SNRi by epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("sisdri_snri.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ("train/count_loss", "val/count_loss", "Count CE", "loss"),
        ("train/spk_distance_mae", "val/spk_distance_mae", "Distance MAE", "m"),
        ("train/count_acc", "val/count_acc", "Count accuracy", "acc"),
    ]
    for ax, (tr, va, title, ylab) in zip(axes, panels):
        ax.plot(epochs, [avg(mv, e, tr) for e in epochs], label="train")
        ax.plot(epochs, [avg(mv, e, va) for e in epochs], label="val")
        ax.axvline(best_ep, color="gray", ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Count / distance head metrics")
    savefig("head_metrics.png")

    plt.figure(figsize=(9, 5))
    dist_keys = [
        ("val/sweep_out_dbfs_0_5m", "0.5 m"),
        ("val/sweep_out_dbfs_0_8m", "0.8 m"),
        ("val/sweep_out_dbfs_1_0m", "1.0 m"),
        ("val/sweep_out_dbfs_1_2m", "1.2 m"),
        ("val/sweep_out_dbfs_1_5m", "1.5 m"),
        ("val/sweep_out_dbfs_2_0m", "2.0 m"),
        ("val/sweep_out_dbfs_3_0m", "3.0 m"),
    ]
    for key, lab in dist_keys:
        xs, ys = series_aligned(mv, epochs, key)
        plt.plot(xs, ys, marker="o", ms=3, label=lab)
    plt.axvline(best_ep, color="gray", ls="--", lw=1, label=f"best ep {best_ep}")
    plt.axhline(-25, color="black", ls=":", lw=1, alpha=0.5, label="target in -25 dBFS")
    plt.xlabel("Epoch")
    plt.ylabel("Output level (dBFS)")
    plt.title("Val distance-dBFS sweep (suppress curve vs epoch)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    savefig("distance_dbfs_by_epoch.png")

    plt.figure(figsize=(8, 4.5))
    x_pf, y_pf = series_aligned(mv, epochs, "val/sweep_passband_minus_far_db")
    x_ed, y_ed = series_aligned(mv, epochs, "val/sweep_edge_drop_1_5_to_2_0_db")
    x_rip, y_rip = series_aligned(mv, epochs, "val/sweep_passband_ripple_0_2_db")
    plt.plot(x_pf, y_pf, marker="o", label="pass-far (dB)")
    plt.plot(x_ed, y_ed, marker="o", label="edge 1.5->2.0 (dB)")
    if x_rip:
        plt.plot(x_rip, y_rip, marker="o", label="passband ripple 0.2s (dB)")
    plt.axvline(best_ep, color="gray", ls="--", lw=1, label=f"best ep {best_ep}")
    plt.xlabel("Epoch")
    plt.ylabel("dB")
    plt.title("Bubble edge metrics vs epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("edge_passfar_by_epoch.png")

    latest = pd.read_csv(sweep_dst / "latest.csv")
    rt_cols = [c for c in latest.columns if c.startswith("out_rt60_")]
    latest["out_mean"] = latest[rt_cols].mean(axis=1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(latest["distance_m"], latest["out_mean"], marker="o", ms=3, label="mean over RT60")
    for c in rt_cols:
        plt.plot(latest["distance_m"], latest[c], alpha=0.35, lw=1, label=c.replace("out_", ""))
    plt.axvline(1.5, color="red", ls="--", lw=1, label="bubble 1.5 m")
    plt.axhline(-25, color="black", ls=":", lw=1, alpha=0.5)
    plt.xlabel("Distance (m)")
    plt.ylabel("Output (dBFS)")
    plt.title(f"Latest distance-dBFS profile (epoch {last_ep})")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    savefig("distance_dbfs_profile_latest.png")

    e1 = sweep_dst / "epoch_001.csv"
    e13 = sweep_dst / "epoch_013.csv"
    if e1.exists() and e13.exists():
        d1 = pd.read_csv(e1)
        d13 = pd.read_csv(e13)
        d1["m"] = d1[rt_cols].mean(axis=1)
        d13["m"] = d13[rt_cols].mean(axis=1)
        plt.figure(figsize=(8, 4.5))
        plt.plot(d1["distance_m"], d1["m"], label="epoch 1")
        eb = sweep_dst / f"epoch_{best_ep:03d}.csv"
        if eb.exists():
            db = pd.read_csv(eb)
            db["m"] = db[rt_cols].mean(axis=1)
            plt.plot(db["distance_m"], db["m"], label=f"best epoch {best_ep}")
        plt.plot(d13["distance_m"], d13["m"], label="epoch 13")
        plt.axvline(1.5, color="red", ls="--", lw=1, label="bubble 1.5 m")
        plt.axhline(-25, color="black", ls=":", lw=1, alpha=0.5)
        plt.xlabel("Distance (m)")
        plt.ylabel("Output mean (dBFS)")
        plt.title("Distance-dBFS profile evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()
        savefig("distance_dbfs_profile_evolution.png")

    print("best_epoch", best_ep, "last", last_ep, "n_epochs", len(epochs))
    for label, ep in [("ep0", 0), ("best", best_ep), ("last", epochs[-1])]:
        vals = {
            "1.0": avg(mv, ep, "val/sweep_out_dbfs_1_0m"),
            "1.5": avg(mv, ep, "val/sweep_out_dbfs_1_5m"),
            "2.0": avg(mv, ep, "val/sweep_out_dbfs_2_0m"),
            "edge": avg(mv, ep, "val/sweep_edge_drop_1_5_to_2_0_db"),
            "passfar": avg(mv, ep, "val/sweep_passband_minus_far_db"),
            "sisdri": avg(mv, ep, "val/si_sdr_i"),
        }
        fmt = {k: (None if v is None else round(v, 2 if k == "sisdri" else 1)) for k, v in vals.items()}
        print(label, fmt)


if __name__ == "__main__":
    main()
