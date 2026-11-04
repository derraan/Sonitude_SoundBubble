"""Figures for docs/polar_offline_ep35_20261104.md (reads debug/polar_offline/*.json + live trials)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parent
RUN = REPO / "measure" / "run_20261104_040645_1abaf4"
IMG = REPO / "docs" / "img" / "polar_offline_ep35_20261104"
IMG.mkdir(parents=True, exist_ok=True)
CANON, MIRROR = "5,4,3,2,1,0", "0,1,2,3,4,5"
C = {"live": "#c2410c", "canon": "#1d4ed8", "mirror": "#059669", "best": "#7c3aed", "grey": "#9ca3af"}

live = {}
for tj in RUN.glob("polar_*/trial.json"):
    d = json.loads(tj.read_text(encoding="utf-8"))
    if d.get("valid"):
        live.setdefault(float(d["azimuth_deg"]), []).append(d["metrics_summary"]["suppression_db"])
live = {a: float(np.mean(v)) for a, v in sorted(live.items())}
sweeps = json.loads((D / "sweeps_full.json").read_text())
screen = json.loads((D / "screen_0_30_330.json").read_text()) if (D / "screen_0_30_330.json").exists() else {}


def polar(ax, series: dict, label, color, ls="-"):
    a = sorted(float(k) for k in series)
    v = [series[str(k)] if str(k) in series else series[k] for k in a]
    th = np.deg2rad(a + [a[0]]); r = v + [v[0]]
    ax.plot(th, r, ls, color=color, lw=2, marker="o", ms=4, label=label)


def setup(ax, rmax=80):
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_rlim(0, rmax); ax.set_rlabel_position(100)
    ax.set_title("Suppression (dB) of a 1 m source, inside 1.5 m bubble\n(0 dB = kept perfectly; higher = more deleted)", fontsize=9)


# 1) live vs offline-fresh (canonical map)
fig = plt.figure(figsize=(6.4, 6.4)); ax = fig.add_subplot(projection="polar"); setup(ax)
polar(ax, {a: v for a, v in live.items()}, "Live bench (n=13 takes)", C["live"])
polar(ax, sweeps[CANON], f"Offline, fresh state, map {CANON}", C["canon"], "--")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(IMG / "01_polar_live_vs_offline.png", dpi=140); plt.close(fig)

# 2) history dependence
hc = json.loads((D / "history_curve.json").read_text())
fig, ax = plt.subplots(figsize=(6.4, 3.4))
xs = [r["history_s"] for r in hc["rows"]]; ys = [r["suppression_db"] for r in hc["rows"]]
ax.plot(xs, ys, "-o", color=C["canon"], lw=2, label="Offline: same 0° sweep after N s of real audio")
ax.axhline(hc["live_db"], color=C["live"], ls="--", lw=1.5, label=f"Live take ({hc['live_db']:.1f} dB)")
ax.set_xlabel("Audio streamed into the model before the sweep (s)"); ax.set_ylabel("Suppression at 0° (dB)")
ax.set_ylim(0, 80); ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.set_title("Same input, different history → 37–73 dB: output depends on recurrent state", fontsize=9)
fig.tight_layout(); fig.savefig(IMG / "02_history_dependence.png", dpi=140); plt.close(fig)

if len(screen) >= 720:  # figures 3-4 need the full channel-order screen
    # 3) permutation screen
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    pts = {m: (np.mean([v["0.0"], v["30.0"], v["330.0"]]), abs(v["30.0"] - v["330.0"])) for m, v in screen.items()}
    xy = np.array(list(pts.values()))
    ax.scatter(xy[:, 0], xy[:, 1], s=8, color=C["grey"], alpha=0.6, label="all 720 channel orders")
    best = sorted(pts, key=lambda m: pts[m][0] + pts[m][1])[:5]
    for m, col, lab in [(CANON, C["canon"], f"canon {CANON}"), (MIRROR, C["mirror"], f"mirror {MIRROR}")] + [(b, C["best"], None) for b in best]:
        ax.scatter(*pts[m], s=60, color=col, edgecolor="k", zorder=3, label=lab)
        ax.annotate(m, pts[m], fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean suppression at 0°/30°/330° (dB)  — lower = keeps in-bubble source"); ax.set_ylabel("|30° − 330°| (dB) — lower = symmetric")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("Channel-order screen (offline, fresh state). Purple = 5 best by mean+asymmetry", fontsize=9)
    fig.tight_layout(); fig.savefig(IMG / "03_channel_order_screen.png", dpi=140); plt.close(fig)

    # 4) full offline polars for candidate maps
    fig = plt.figure(figsize=(6.4, 6.4)); ax = fig.add_subplot(projection="polar"); setup(ax, 90)
    cols = [C["canon"], C["mirror"], C["best"], "#db2777", "#0891b2", "#ca8a04", "#4b5563"]
    for i, (m, s) in enumerate(sweeps.items()):
        polar(ax, s, f"map {m}" + (" (canon)" if m == CANON else " (mirror)" if m == MIRROR else ""), cols[i % len(cols)], "-" if i < 2 else "--")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(IMG / "04_polar_candidate_maps.png", dpi=140); plt.close(fig)
print("wrote", sorted(p.name for p in IMG.glob("*.png")))
