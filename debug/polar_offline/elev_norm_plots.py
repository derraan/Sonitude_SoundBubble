"""Figures for the per-pose normalized elevation evaluation (reads results.json; writes NEW pngs)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "measure" / "run_20261104_040645_1abaf4" / "analysis" / "offline_elev_norm_20261104"
DOCIMG = REPO / "docs" / "img" / "offline_elev_norm_20261104"
(OUT / "figures").mkdir(exist_ok=True); DOCIMG.mkdir(parents=True, exist_ok=True)
R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))["results"]
PROT = ["tail5s", "silence5s", "tail330_5s", "tail10s", "fresh"]
BANDS = ["100-300", "300-1k", "1-2k", "2-4k", "4-6k", "6-8k", "8-11k"]
COL = {"norm": "#1d4ed8", "bench": "#c2410c", "live": "#111827"}


def per_elev(gname):
    """elevation -> list of per-take protocol medians, and all protocol values pooled."""
    out = {}
    for r in R:
        g, p = r["condition"].split("|")
        if g != gname:
            continue
        out.setdefault(r["elevation_deg"], {}).setdefault(r["trial"], []).append(r["suppression_db"])
    return out


def save(fig, name):
    for d in (OUT / "figures", DOCIMG):
        fig.savefig(d / name, dpi=140)
    plt.close(fig)


# 1) suppression vs elevation: median across protocols (per take, then mean of takes), min-max band
fig, ax = plt.subplots(figsize=(6.6, 4.0))
for g, lab in (("bench", "bench mic gains (dev ch4 13 dB hot off-axis)"), ("norm", "per-pose normalized mics (all at −19 dBFS)")):
    pe = per_elev(g); els = sorted(pe)
    med = [np.mean([np.median(v) for v in pe[e].values()]) for e in els]
    lo = [min(min(v) for v in pe[e].values()) for e in els]; hi = [max(max(v) for v in pe[e].values()) for e in els]
    ax.fill_between(els, lo, hi, color=COL[g], alpha=0.15)
    ax.plot(els, med, "-o", color=COL[g], lw=2, label=f"offline, {lab}\n(line = median over 5 state protocols; band = min–max)")
live = {}
for r in R:
    live.setdefault(r["elevation_deg"], {})[r["trial"]] = r["live_suppression_db"]
for e, d in live.items():
    ax.scatter([e] * len(d), list(d.values()), marker="x", s=50, color=COL["live"], zorder=4)
ax.scatter([], [], marker="x", color=COL["live"], label="live bench takes")
ax.set_xlabel("Elevation (°), azimuth 0°, 1 m (inside 1.5 m bubble)"); ax.set_ylabel("Suppression (dB) — 0 = kept")
ax.set_xticks([-60, -30, 0, 30, 60]); ax.set_ylim(0, 90); ax.grid(alpha=0.3)
ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
ax.set_title("EP35 elevation response: effect of per-pose mic normalization", fontsize=9)
fig.tight_layout(); save(fig, "01_elev_suppression_norm_vs_bench.png")

# 2) per-protocol detail (normalized)
fig, ax = plt.subplots(figsize=(6.6, 3.8))
mk = dict(zip(PROT, ["o", "s", "^", "D", "v"]))
for p in PROT:
    pts = [(r["elevation_deg"], r["suppression_db"]) for r in R if r["condition"] == f"norm|{p}"]
    jitter = {"tail5s": -3, "silence5s": -1.5, "tail330_5s": 0, "tail10s": 1.5, "fresh": 3}[p]
    ax.scatter([e + jitter for e, _ in pts], [s for _, s in pts], marker=mk[p], s=28, label=p)
ax.set_xlabel("Elevation (°)"); ax.set_ylabel("Suppression (dB)"); ax.set_xticks([-60, -30, 0, 30, 60]); ax.set_ylim(0, 90)
ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.17))
ax.set_title("Normalized mics: every take × state protocol (state still swings results up to ~39 dB)", fontsize=9)
fig.tight_layout(); save(fig, "02_elev_norm_by_state_protocol.png")

# 3) band keep heatmaps, protocol tail5s
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
for ax, g in zip(axes, ("bench", "norm")):
    rows = sorted([r for r in R if r["condition"] == f"{g}|tail5s"], key=lambda r: (r["elevation_deg"], r["trial"]))
    M = np.array([r["band_keep_db"] for r in rows])
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=-80, vmax=0)
    ax.set_xticks(range(len(BANDS))); ax.set_xticklabels(BANDS, rotation=45, fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r['elevation_deg']:+.0f}° …{r['trial'][-4:]}" for r in rows], fontsize=7)
    ax.set_title(f"{g}: output − input per band (dB), state tail5s", fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6, color="w" if M[i, j] < -40 else "k")
fig.colorbar(im, ax=axes, shrink=0.8, label="keep (dB)")
save(fig, "03_elev_band_keep_heatmap.png")
print("figures:", sorted(p.name for p in (OUT / "figures").glob("*.png")))
