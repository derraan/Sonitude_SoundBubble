"""Live mic hardware-state check: play the bench ESS on the stimulus device, record the
6 Pico mics directly (no pipeline), and apply the wideband deficit test.

Usage: python debug/mic_state_check.py [--in 37] [--out 41] [--stim PATH]
Writes debug/mic_state_check_<utc>.wav and .json (never into measure/).
"""
import argparse, json, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, sounddevice as sd, soundfile as sf
from scipy.signal import butter, sosfilt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.measure.stimulus import StimulusPlayer  # noqa: E402

STIM = r"C:\Users\darre\Documents\Obsidian Notes\CDE3301\3301-ANCHRTF\AY 2627 SEM1\Software and Firmware\6ChMic-DataCollection\256kMeasSweep_0_to_20000_-20_dBFS_44k_Float_LR_refL.wav"
BANDS = [(250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
FLAT_DB, MIN_DB = 3.0, 6.0
EXPECTED = {2: -12.6, 5: -12.9}  # stable wideband offsets; dev4 toggles (low ≈ -12.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="dev_in", type=int, default=37)
    ap.add_argument("--out", dest="dev_out", type=int, default=41)
    ap.add_argument("--stim", default=STIM)
    a = ap.parse_args()
    sr = 24000
    chunks = []
    stop = threading.Event()

    def cb(indata, frames, t, status):
        chunks.append(indata.copy())

    stream = sd.InputStream(device=a.dev_in, channels=8, samplerate=sr, dtype="int32", callback=cb)
    stream.start(); time.sleep(0.5)
    player = StimulusPlayer(a.stim, device=a.dev_out, repeat=1, play_channels="left")
    info = player.play_blocking()
    time.sleep(0.5); stream.stop(); stream.close()
    xi = np.concatenate(chunks)                       # raw int32 from the USB device
    x = (xi.astype(np.float64) / 2**31).astype(np.float32)
    # Bit structure: trailing zero bits per channel (mode over non-zero samples).
    tz = []
    for c in range(6):
        v = xi[:, c].astype(np.int64); v = v[v != 0]
        t = np.zeros(len(v), int); vv = np.abs(v)
        for b in range(32):
            m = (vv >> b) & 1; t = np.where((t == b) & (m == 0), b + 1, t)
        tz.append(int(np.bincount(t).argmax()) if len(t) else -1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    wav = REPO / "debug" / f"mic_state_check_{stamp}.wav"
    sf.write(wav, x, sr, subtype="FLOAT")

    m = x[:, :6]
    w = sr // 10
    e = np.array([np.mean(m[i:i + w] ** 2, 0) for i in range(0, len(m) - w, w)]).sum(1)
    act = np.repeat(e > np.percentile(e, 30), w)[: len(m)]
    act = np.concatenate([act, np.zeros(len(m) - len(act), bool)])
    noise = np.repeat(e < np.percentile(e, 5), w)[: len(m)]
    noise = np.concatenate([noise, np.zeros(len(m) - len(noise), bool)])
    L = np.array([10 * np.log10(np.mean(sosfilt(butter(4, b, "band", fs=sr, output="sos"), m, axis=0)[act] ** 2, 0)) for b in BANDS])
    hf = 10 * np.log10(np.mean(sosfilt(butter(4, (8000, 11000), "band", fs=sr, output="sos"), m, axis=0)[act] ** 2, 0))
    nf = 10 * np.log10(np.mean(m[noise] ** 2, 0) + 1e-20)
    broad = 10 * np.log10(np.mean(m[act] ** 2, 0)); snr = broad - 10 * np.log10(np.mean(m[noise] ** 2, 0) + 1e-20)
    top3 = np.argsort(L.mean(0))[-3:]; deficit = L - L[:, top3].mean(1)[:, None]
    rows = []
    print(f"\nrecorded {len(x)/sr:.1f} s  ->  {wav.name}   reference (3 loudest) = dev {sorted(int(i) for i in top3)}")
    print(f"{'dev':>4} {'raw dBFS':>9} {'SNR':>5}   deficit per band (dB) 250-500 .. 4k-8k     flat   mean   state")
    for c in range(6):
        d = deficit[:, c]; flat = float(np.ptp(d)); mean = float(d.mean())
        wide = flat <= FLAT_DB and -mean >= MIN_DB
        state = f"LOW (wideband {mean:+.1f} dB)" if wide else ("normal" if mean > -MIN_DB else "low but NOT wideband")
        rows.append({"dev_ch": c, "raw_active_dbfs": round(float(broad[c]), 2), "snr_db": round(float(snr[c]), 1),
                     "noise_floor_dbfs": round(float(nf[c]), 2), "hf_8_11k_deficit_db": round(float(hf[c] - hf[top3].mean()), 1),
                     "trailing_zero_bits": tz[c],
                     "deficit_db_per_band": [round(float(v), 1) for v in d], "flatness_db": round(flat, 1),
                     "mean_deficit_db": round(mean, 1), "wideband_low": bool(wide), "state": state})
        print(f"{c:>4} {broad[c]:9.1f} {snr[c]:5.0f}   {np.round(d, 1)}   {flat:4.1f} {mean:6.1f}   {state}")
    low = [r["dev_ch"] for r in rows if r["wideband_low"]]
    print(f"\nwideband-low channels: {low}")
    for c, exp in EXPECTED.items():
        got = rows[c]["mean_deficit_db"]
        print(f"  dev{c}: expected {exp:+.1f} dB, measured {got:+.1f} dB -> {'OK' if abs(got - exp) <= 1.5 and rows[c]['wideband_low'] else 'CHANGED'}")
    print(f"  dev4: {'LOW state (fader ~x6.1)' if rows[4]['wideband_low'] else 'NORMAL state (default x1.514 OK)'}")
    json.dump({"utc": stamp, "devices": {"in": a.dev_in, "out": a.dev_out}, "stimulus": a.stim, "play_channels": "left",
               "bands_hz": BANDS, "reference_dev_ch": sorted(int(i) for i in top3), "mics": rows, "wideband_low": low},
              open(wav.with_suffix(".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
