"""Pi-side mic hardware-state check (no pipeline): record the 6 Pico mics directly and apply
the wideband low-state test from docs/mic_low_state_investigation_20261104.md.

Play a broadband stimulus at the headset while it runs (bench ESS from the testbench laptop,
pink noise, or steady speech). Same rule as debug/mic_state_check.py: a mic is "wideband low"
when its deficit vs the 3 loudest mics is >= 6 dB and flat within 3 dB over 250 Hz-8 kHz.

  python pi_mic_check.py [--device 0] [--seconds 12] [--out /mnt/sonitude-drive/sonitude-recordings/mic_checks]
Writes a NEW <stamp>.wav (raw 8 ch, device order) + <stamp>.json; stop the service first.
"""
import argparse, json, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, sounddevice as sd, soundfile as sf
from scipy.signal import butter, sosfilt

BANDS = [(250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
FLAT_DB, MIN_DB = 3.0, 6.0
EXPECTED = {2: -12.6, 5: -12.9}  # bench 2026-11-04; dev 4 toggles (low ~ -12.1)
SR = 24000
DRIVE = "/mnt/sonitude-drive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--out", default=f"{DRIVE}/sonitude-recordings/mic_checks")
    a = ap.parse_args()
    if a.out.startswith(DRIVE + "/") and not os.path.ismount(DRIVE):
        sys.exit(f"{DRIVE} is not mounted; plug in the Samsung T5 and retry")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"recording {a.seconds:.0f} s from device {a.device}; play the stimulus now")
    x = sd.rec(int(a.seconds * SR), samplerate=SR, channels=8, dtype="float32", device=a.device)
    sd.wait(); time.sleep(0.1)
    sf.write(out / f"{stamp}.wav", x, SR, subtype="FLOAT")

    m = x[:, :6].astype(np.float64); w = SR // 10
    e = np.array([np.mean(m[i:i + w] ** 2, 0) for i in range(0, len(m) - w, w)]).sum(1)
    pad = lambda b: np.concatenate([np.repeat(b, w), np.zeros(len(m) - len(b) * w, bool)])
    act, noise = pad(e > np.percentile(e, 30)), pad(e < np.percentile(e, 5))
    L = np.array([10 * np.log10(np.mean(sosfilt(butter(4, b, "band", fs=SR, output="sos"), m, axis=0)[act] ** 2, 0) + 1e-20) for b in BANDS])
    broad = 10 * np.log10(np.mean(m[act] ** 2, 0) + 1e-20)
    snr = broad - 10 * np.log10(np.mean(m[noise] ** 2, 0) + 1e-20)
    top3 = np.argsort(L.mean(0))[-3:]; deficit = L - L[:, top3].mean(1)[:, None]
    rows = []
    print(f"{'dev':>4} {'dBFS':>7} {'SNR':>5}  deficit/band 250..8k        flat   mean  state")
    for c in range(6):
        d = deficit[:, c]; flat, mean = float(np.ptp(d)), float(d.mean())
        wide = flat <= FLAT_DB and -mean >= MIN_DB
        state = f"LOW wideband {mean:+.1f} dB" if wide else ("normal" if mean > -MIN_DB else "low, NOT wideband")
        rows.append({"dev_ch": c, "active_dbfs": round(float(broad[c]), 2), "snr_db": round(float(snr[c]), 1),
                     "deficit_db_per_band": [round(float(v), 1) for v in d], "flatness_db": round(flat, 1),
                     "mean_deficit_db": round(mean, 1), "wideband_low": bool(wide), "state": state})
        print(f"{c:>4} {broad[c]:7.1f} {snr[c]:5.0f}  {np.round(d, 1)}  {flat:4.1f} {mean:6.1f}  {state}")
    low = {r["dev_ch"]: r["mean_deficit_db"] for r in rows if r["wideband_low"]}
    gains = [round(10 ** (-low[c] / 20), 3) if c in low else 1.0 for c in range(6)]
    match = set(low) == set(EXPECTED)
    print(f"\nwideband-low: {low or 'none'}  ({'matches' if match else 'DIFFERS from'} bench state {EXPECTED})")
    print(f"relative trims (multiply into MIC_GAINS): {','.join(f'{g:g}' for g in gains)}")
    if min(snr) < 15:
        print("WARNING: SNR < 15 dB on some channel; stimulus too quiet, result unreliable")
    json.dump({"stamp": stamp, "device": a.device, "seconds": a.seconds, "bands_hz": BANDS,
               "reference_dev": sorted(int(i) for i in top3), "channels": rows, "wideband_low": low,
               "relative_trims": gains, "matches_bench_state": match},
              open(out / f"{stamp}.json", "w"), indent=1)
    print(f"wrote {out / stamp}.wav/.json")


if __name__ == "__main__":
    main()
