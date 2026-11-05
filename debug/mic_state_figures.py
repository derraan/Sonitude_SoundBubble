"""Figures for docs/mic_low_state_investigation_20261104.md (reads debug/mic_state_check_* data)."""
import glob, json
from pathlib import Path
import numpy as np, soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

REPO = Path(__file__).resolve().parents[1]
D = REPO / "debug"
IMG = REPO / "docs" / "img" / "mic_low_state_20261104"; IMG.mkdir(parents=True, exist_ok=True)
NOREP = ["065519", "065539", "065558", "065617", "065636"]
REP = ["070657", "070731", "070758", "070836", "070906"]
load = lambda ids: [json.load(open(D / f"mic_state_check_20261104_{i}.json")) for i in ids]
A, B = load(NOREP), load(REP)
C = {"norm": "#6b7280", "low": "#c2410c", "a": "#1d4ed8", "b": "#059669"}

# 1) per-mic deficit, both series
fig, ax = plt.subplots(figsize=(6.6, 3.6))
x = np.arange(6); w = 0.38
for k, (S, lab, col) in enumerate(((A, "5 runs, no replug", C["a"]), (B, "5 runs, USB power cycle before each", C["b"]))):
    v = np.array([[r["mics"][c]["mean_deficit_db"] for c in range(6)] for r in S])
    ax.bar(x + (k - 0.5) * w, v.mean(0), w, yerr=v.std(0), color=col, label=lab, capsize=3)
    for c in range(6):
        ax.text(x[c] + (k - 0.5) * w, v.mean(0)[c] + (0.4 if v.mean(0)[c] >= 0 else -1.1), f"{v.mean(0)[c]:+.1f}", ha="center", fontsize=7)
ax.axhline(-6, color="#9ca3af", ls="--", lw=0.8); ax.text(5.6, -5.6, "−6 dB (low threshold)", fontsize=7, ha="right", color="#6b7280")
ax.set_xticks(x); ax.set_xticklabels([f"dev{c}" for c in range(6)]); ax.set_ylim(-15, 3)
ax.set_ylabel("Level vs 3 loudest mics (dB)"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_title("Per-mic level offset (ESS, 300 Hz–8 kHz bands averaged): identical with and without power cycles", fontsize=8)
fig.tight_layout(); fig.savefig(IMG / "01_per_mic_offset_replug_vs_not.png", dpi=140); plt.close(fig)

# 2) per-band deficit spectra (mean of all 10 runs)
bands = ["250-500", "500-1k", "1-2k", "2-4k", "4-8k"]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
for c in range(6):
    v = np.array([r["mics"][c]["deficit_db_per_band"] for r in A + B]).mean(0)
    hf = np.mean([r["mics"][c]["hf_8_11k_deficit_db"] for r in A + B])
    low = c in (2, 5)
    ax.plot(range(6), list(v) + [hf], "-o" if low else ":o", ms=3, color=({2: C["low"], 5: "#7c2d12"}[c] if low else None), lw=2 if low else 1, label=f"dev{c}" + (" (low)" if low else ""))
ax.set_xticks(range(6)); ax.set_xticklabels(bands + ["8-11k"], fontsize=8); ax.set_ylim(-16, 3)
ax.set_ylabel("Deficit vs 3 loudest (dB)"); ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False, ncol=3, loc="center right")
ax.set_title("Low-state offset is wideband: flat within ~2 dB to 8 kHz (dev5 ≈1 dB extra above 8 kHz)", fontsize=8)
fig.tight_layout(); fig.savefig(IMG / "02_band_deficit_spectra.png", dpi=140); plt.close(fig)

# 3) silence noise floor: low vs normal, and expected curve for a pure digital /4
x, sr = sf.read(D / "mic_state_check_20261104_silence10s.wav", always_2d=True)
f, P = welch(x[:, :6].T, sr, nperseg=4096)
ref = P[[0, 1, 3]].mean(0)
sm = lambda y: np.convolve(y, np.ones(9) / 9, mode="same")
fig, ax = plt.subplots(figsize=(6.6, 3.4))
k = (f > 80) & (f < 11500)
for c, col in ((2, C["low"]), (5, "#7c2d12")):
    ax.semilogx(f[k], sm(10 * np.log10(P[c] / ref))[k], color=col, lw=1.8, label=f"dev{c} noise floor vs normal mics (silence)")
ax.axhline(-12.75, color=C["a"], ls="--", lw=1.2, label="expected if digital ÷4 after ADC (all noise scaled −12.6…−12.9 dB)")
ax.axhline(0, color="#9ca3af", lw=0.8)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Relative noise floor (dB)"); ax.set_ylim(-16, 6); ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.set_title("Silence: low-mic noise floor −11 dB at LF but only −5 dB at HF\n(not a uniform digital scale → loss before the mic's own noise)", fontsize=8)
fig.tight_layout(); fig.savefig(IMG / "03_silence_noise_floor.png", dpi=140); plt.close(fig)

# 4) dev ch4 timeline across the day (raw level relative to dev0/1/3 at front poses)
tl = [("00:04", 0.0, "level sweeps"), ("00:20", 0.0, ""), ("02:12", -12.0, "polar takes"), ("04:13", -12.2, "az 0 ref"),
      ("05:25", -12.0, "az 0 repeat"), ("05:37", 0.3, "elevation takes (rig reconfigured)"), ("05:53", 0.4, ""),
      ("06:52", 0.2, ""), ("06:55", 0.2, "direct checks\n(1 + 5 runs + 5 power cycles)"), ("07:06", 0.2, "")]
hh = [int(t[:2]) + int(t[3:]) / 60 for t, _, _ in tl]
fig, ax = plt.subplots(figsize=(6.6, 3.0))
ax.step(hh, [v for _, v, _ in tl], where="post", color=C["low"], lw=2)
ax.scatter(hh, [v for _, v, _ in tl], color=C["low"], zorder=3, s=18)
for h, (t, v, lab) in zip(hh, tl):
    ax.annotate(lab, (h, v), xytext=((-10, 7) if v <= -6 or h < 6.5 else (-30, -24)) if v > -6 else (-10, -13), textcoords="offset points", fontsize=6.5)
ax.axvspan(2.2, 5.42, color="#fde68a", alpha=0.35, label="azimuth sweep + 0° refs (ch4 low, gains matched)")
ax.set_xlabel("Time of day (UTC, 2026-11-04)"); ax.set_ylabel("dev4 vs dev0/1/3 (dB)"); ax.set_ylim(-16, 5); ax.grid(alpha=0.3)
ax.legend(fontsize=7, frameon=False, loc="center right")
ax.set_title("dev ch4 toggles low ↔ normal with rig handling, not with power cycles", fontsize=8)
fig.tight_layout(); fig.savefig(IMG / "04_dev4_timeline.png", dpi=140); plt.close(fig)
print(sorted(p.name for p in IMG.glob("*.png")))
