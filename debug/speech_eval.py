"""Evaluate the in_context_speech algorithm takes of 2026-11-04 (EP35).

Writes NEW files only:
  measure/run_20261104_073046_3447eb/analysis/speech_eval_20261104/{results.json,results.csv,figures/*}
  docs/img/speech_ep35_20261104/*.png
"""
import csv, json, re
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
import sys
RUN = REPO / "measure" / (sys.argv[1] if len(sys.argv) > 1 else "run_20261104_073046_3447eb")
OUT = RUN / "analysis" / (sys.argv[2] if len(sys.argv) > 2 else "speech_eval_20261104"); (OUT / "figures").mkdir(parents=True, exist_ok=True)
IMG = REPO / "docs" / "img" / (sys.argv[3] if len(sys.argv) > 3 else "speech_ep35_20261104"); IMG.mkdir(parents=True, exist_ok=True)
BANDS = [(100, 300), (300, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 11000)]
BL = ["100-300", "300-1k", "1-2k", "2-4k", "4-6k", "6-8k", "8-11k"]


def gcc_ms(x, ref, sr):
    X, Y = np.fft.rfft(x), np.fft.rfft(ref); G = X * np.conj(Y)
    g = np.fft.irfft(G / (np.abs(G) + 1e-12)); g = np.concatenate([g[-48:], g[:49]])
    return (int(np.argmax(g)) - 48) / sr * 1000


def bands(x, sr):
    f, P = welch(x, sr, nperseg=4096)
    return np.array([10 * np.log10(P[(f >= lo) & (f < hi)].mean() + 1e-30) for lo, hi in BANDS])


rows = []
for t in sorted(RUN.glob("algorithm_20261104_*")):
    tj = t / "trial.json"
    if not tj.is_file() or not (t / "wet_out.wav").is_file():
        continue
    d = json.loads(tj.read_text(encoding="utf-8"))
    if not d.get("stimulus_speech"):
        continue
    m = d["metrics_summary"]
    raw, sr = sf.read(t / "raw_8ch.wav", always_2d=True)
    dry, _ = sf.read(t / "dry_ref.wav"); wet, _ = sf.read(t / "wet_out.wav")
    n = min(len(dry), len(wet)); dry, wet = dry[:n], wet[:n]
    w = sr // 20
    e = np.array([np.mean(dry[k:k + w] ** 2) for k in range(0, n - w, w)]); act = e > np.percentile(e, 35)
    segs, s0 = [], None
    for k, a in enumerate(act):
        if a and s0 is None: s0 = k
        if (not a or k == len(act) - 1) and s0 is not None:
            if k - s0 > 12: segs.append((s0 * w, k * w))
            s0 = None
    per_utt = [float(10 * np.log10(np.mean(wet[a:b] ** 2) / np.mean(dry[a:b] ** 2))) for a, b in segs]
    seg_t = [a / sr for a, _ in segs]
    er = np.array([np.mean(raw[k:k + w, 0] ** 2) for k in range(0, len(raw) - w, w)]); ar = er > np.percentile(er, 35)
    am = np.repeat(ar, w)[: len(raw)]; am = np.concatenate([am, np.zeros(len(raw) - len(am), bool)])
    mm = raw[am][:, :6]; dl = [gcc_ms(mm[:, c], mm[:, 0], sr) for c in range(6)]
    log = (t / "pipeline.log").read_text(encoding="utf-8", errors="ignore")
    ur = [int(v) for v in re.findall(r"underrun_frames=(\d+)", log)]
    keep = bands(wet, sr) - bands(dry, sr)
    top_delay = (dl[2] + dl[3]) / 2
    el = float(d["elevation_deg"])
    exp = {0.0: 0.02, 30.0: -0.21, 60.0: -0.34, -30.0: 0.125, -60.0: 0.333} if float(d["distance_m"]) == 1.0 else {}
    pose_check = "ok" if (el not in exp or abs(top_delay - exp[el]) < 0.06) else "CHECK"
    rows.append({
        "trial": t.name, "time_utc": d["created_utc"][11:19], "elevation_deg": el, "azimuth_deg": d["azimuth_deg"], "distance_m": d["distance_m"],
        "input_gain_db_actual": d["gains"]["input_gain_db"], "gain_label_corrected": bool(d.get("settings_mismatch")),
        "dry_dbfs": m["dry_fullband_db"], "wet_dbfs": m["wet_fullband_db"], "suppression_db": m["suppression_db"],
        "keep_db_per_band": [round(float(v), 1) for v in keep], "keep_per_utterance_db": [round(v, 1) for v in per_utt],
        "utterance_onsets_s": [round(v, 2) for v in seg_t], "trend_db": round(per_utt[-1] - per_utt[0], 1) if len(per_utt) > 1 else 0.0,
        "top_mic_delay_ms": round(top_delay, 3), "pose_check": pose_check, "underruns": (ur[-1] - ur[0]) if ur else None,
        "output_gain_db": d["gains"].get("output_gain_db"), "dev4_minus_dev0_db": round(float(20 * np.log10(np.std(raw[:, 4]) / np.std(raw[:, 0]))), 1),
        "utterances": [u["file"] for u in d["stimulus_speech"]["utterances"]], "notes": d.get("notes", ""),
    })

# Pose check by ORDERING (no fixed per-angle reference): each take must sit within
# 0.06 ms of the median top-mic delay of its (distance, elevation) group, and group
# medians must decrease strictly with elevation (source higher -> top mics earlier).
if "--order-pose-check" in sys.argv:
    from collections import defaultdict
    grp = defaultdict(list)
    for r in rows:
        grp[(r["distance_m"], r["elevation_deg"])].append(r["top_mic_delay_ms"])
    med = {k: float(np.median(v)) for k, v in grp.items()}
    order_ok = {}
    for dist in {k[0] for k in med}:
        els = sorted(e for (dd, e) in med if dd == dist)
        ms = [med[(dist, e)] for e in els]
        order_ok[dist] = all(a > b for a, b in zip(ms, ms[1:]))
    for r in rows:
        k = (r["distance_m"], r["elevation_deg"])
        r["pose_check"] = "ok" if order_ok[r["distance_m"]] and abs(r["top_mic_delay_ms"] - med[k]) < 0.06 else "CHECK"
    print("pose ordering ok per distance:", order_ok, "group median delays:", {f"{k[0]}m {k[1]:+.0f}": round(v, 3) for k, v in sorted(med.items())})

(OUT / "results.json").write_text(json.dumps({"bands_hz": BANDS, "takes": rows}, indent=1), encoding="utf-8")
with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.writer(fh)
    wr.writerow(["trial", "time_utc", "elevation_deg", "input_gain_db_actual", "dry_dbfs", "suppression_db", "trend_db", "top_mic_delay_ms", "pose_check"] + [f"keep_{b}" for b in BL])
    for r in rows:
        wr.writerow([r["trial"], r["time_utc"], r["elevation_deg"], r["input_gain_db_actual"], round(r["dry_dbfs"], 1), round(r["suppression_db"], 1),
                     r["trend_db"], r["top_mic_delay_ms"], r["pose_check"]] + r["keep_db_per_band"])


def save(fig, name):
    for d in (OUT / "figures", IMG):
        fig.savefig(d / name, dpi=140)
    plt.close(fig)


if "--by-distance" in sys.argv:
    col = lambda r: {1.0: "#1d4ed8", 0.5: "#c2410c"}.get(float(r["distance_m"]), "#6b7280")
    lab = lambda r: f"{r['distance_m']:.1f} m"
else:
    col = lambda r: {(15.0, 0.0): "#1d4ed8", (25.0, 0.0): "#c2410c", (25.0, 30.0): "#7c3aed", (25.0, 60.0): "#059669"}.get((r["input_gain_db_actual"], r["elevation_deg"]), "#6b7280")
    lab = lambda r: f"in {r['input_gain_db_actual']:.0f} dB, el {r['elevation_deg']:+.0f}°"

# 1) suppression per take in time order
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for i, r in enumerate(rows):
    ax.bar(i, r["suppression_db"], color=col(r), hatch="//" if r["pose_check"] != "ok" else None, edgecolor="k" if r["pose_check"] != "ok" else None)
    ax.text(i, r["suppression_db"] + 1.5, f"{r['suppression_db']:.0f}", ha="center", fontsize=7)
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["time_utc"][:5] + "\n…" + r["trial"][-4:] for r in rows], fontsize=6)
handles = {}
for r in rows: handles[lab(r)] = plt.Rectangle((0, 0), 1, 1, color=col(r))
handles["pose CHECK (delay ≠ label)"] = plt.Rectangle((0, 0), 1, 1, facecolor="white", hatch="//", edgecolor="k")
ax.legend(handles.values(), handles.keys(), fontsize=7, frameon=False, loc="upper left", ncol=2)
ax.set_ylabel("Suppression (dB) — 0 = kept"); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
ax.set_title("Speech takes (talker 1 m, inside bubble): same talker level, 20–78 dB suppression", fontsize=9)
fig.tight_layout(); save(fig, "01_suppression_per_take.png")

# 2) per-utterance keep trajectories
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for r in rows:
    ax.plot(r["utterance_onsets_s"], r["keep_per_utterance_db"], "-o", ms=3, color=col(r), alpha=0.85, lw=1.4)
for k, h in handles.items():
    if "CHECK" not in k: ax.plot([], [], "-o", color=h.get_facecolor(), label=k)
ax.set_xlabel("Time into take (s)"); ax.set_ylabel("Keep per utterance segment (dB)"); ax.set_ylim(-90, 0); ax.grid(alpha=0.3)
ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
ax.set_title("Within a take the model either stays closed (−70…−80) or starts open and closes over 15–20 s", fontsize=9)
fig.tight_layout(); save(fig, "02_keep_per_utterance.png")

# 3) band keep heatmap
M = np.array([r["keep_db_per_band"] for r in rows])
fig, ax = plt.subplots(figsize=(6.8, 5.2))
im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=-80, vmax=0)
ax.set_xticks(range(len(BL))); ax.set_xticklabels(BL, rotation=45, fontsize=7)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r['time_utc'][:5]} …{r['trial'][-4:]}  {lab(r)}{(' el %+.0f°' % r['elevation_deg']) if '--by-distance' in sys.argv else ''}{'  [CHECK]' if r['pose_check']!='ok' else ''}" for r in rows], fontsize=6.5)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6, color="w" if M[i, j] < -40 else "k")
fig.colorbar(im, ax=ax, shrink=0.8, label="keep (dB)")
ax.set_title("Output − input per band (dB): when open, highs pass, lows are cut (bright residue)", fontsize=8)
fig.tight_layout(); save(fig, "03_band_keep_heatmap.png")

for r in rows:
    print(f"{r['time_utc']} {r['trial'][-9:]} el {r['elevation_deg']:+3.0f} in {r['input_gain_db_actual']:.0f} dry {r['dry_dbfs']:5.1f} supp {r['suppression_db']:5.1f} trend {r['trend_db']:+6.1f} pose {r['pose_check']} keep {r['keep_db_per_band']} utt {r['keep_per_utterance_db']}")

# 4) suppression vs elevation per distance (only when the run spans elevations)
if "--by-distance" in sys.argv:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for dist, c in ((1.0, "#1d4ed8"), (0.5, "#c2410c")):
        rr = sorted([r for r in rows if float(r["distance_m"]) == dist], key=lambda r: r["elevation_deg"])
        if not rr: continue
        els = sorted({r["elevation_deg"] for r in rr})
        med = [np.mean([r["suppression_db"] for r in rr if r["elevation_deg"] == e]) for e in els]
        ax.plot(els, med, "-o", color=c, lw=2, label=f"{dist:.1f} m")
        for r in rr:
            ax.scatter(r["elevation_deg"], r["suppression_db"], color=c, s=14, alpha=0.6)
            ax.annotate(r["time_utc"][:5], (r["elevation_deg"], r["suppression_db"]), xytext=(3, -9), textcoords="offset points", fontsize=5.5, color=c)
    ax.set_xticks([-60, -30, 0, 30, 60]); ax.set_ylim(0, 90); ax.grid(alpha=0.3)
    ax.set_xlabel("Elevation (°), azimuth 0°"); ax.set_ylabel("Suppression (dB) — 0 = kept")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("Speech, talker inside bubble: closed (69–77 dB) at every elevation and distance", fontsize=9)
    fig.tight_layout(); save(fig, "04_suppression_vs_elevation.png")
