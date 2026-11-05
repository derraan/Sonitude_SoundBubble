"""Technical QA of a polar azimuth sweep run (read-only)."""
import json, re, sys, numpy as np, soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt
run = Path(sys.argv[1]); ref_id = sys.argv[2]; SKIP = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else set()
ONLY = sys.argv[4] if len(sys.argv) > 4 else ""
ref = json.loads((run / ref_id / "trial.json").read_text(encoding="utf-8"))
sos = butter(4, (300, 4000), 'band', fs=24000, output='sos')
def snr(x):
    y = sosfilt(sos, x, axis=0); w = 2400
    e = np.array([np.mean(y[i:i+w]**2, 0) for i in range(0, len(y)-w, w)]).sum(1)
    return 10*np.log10(np.percentile(e, 90) / max(np.percentile(e, 5), 1e-30))
rows = []
for tj in sorted(run.glob("polar_*/trial.json")):
    if ONLY and tj.parent.name < ONLY: continue
    t = tj.parent; d = json.loads(tj.read_text(encoding="utf-8")); m = d.get("metrics_summary", {}); fails = []
    for k in [k for k in ("model", "channel_map", "distance_m", "elevation_deg", "stimulus_sha256_prefix", "runtime") if k not in SKIP]:
        if d.get(k) != ref.get(k): fails.append(f"{k}={d.get(k)}")
    g, rg = d.get("gains", {}), ref.get("gains", {})
    for k in ("input_gain_db", "output_gain_db", "mix_dry"):
        if g.get(k) != rg.get(k): fails.append(f"{k}={g.get(k)}")
    if [round(v, 3) for v in g.get("mic_gains", [])] != [round(v, 3) for v in rg.get("mic_gains", [])]: fails.append("mic_gains differ")
    if d.get("runtime_exit_code") != 0: fails.append(f"exit={d.get('runtime_exit_code')}")
    if d.get("taps_missing") or d.get("tap_truncated"): fails.append("taps missing/truncated")
    if list(t.glob("*.part*.wav")): fails.append("part files")
    tap_mt = max((t / f).stat().st_mtime for f in ("raw_8ch.wav", "mapped_6ch.wav", "dry_ref.wav", "wet_out.wav"))
    if tap_mt > (t / "hats_l.wav").stat().st_mtime + 30: fails.append("taps rewritten after take")
    raw, sr = sf.read(t / "raw_8ch.wav", always_2d=True); dur = len(raw) / sr
    if dur < 13.0: fails.append(f"raw {dur:.1f}s")
    s_mic = snr(raw[:, :6]); h, hsr = sf.read(t / "hats_l.wav", always_2d=True)
    s_hats = snr(sosfilt(butter(4, (300, 4000), 'band', fs=hsr, output='sos'), h[:, :1], axis=0)) if True else 0
    if s_mic < 25: fails.append(f"mic SNR {s_mic:.0f}dB")
    if s_hats < 15: fails.append(f"HATS SNR {s_hats:.0f}dB")
    log = (t / "pipeline.log").read_text(encoding="utf-8", errors="ignore")
    ur = [int(x) for x in re.findall(r"underrun_frames=(\d+)", log)]
    if ur and ur[-1] - ur[0] > 0: fails.append(f"underruns +{ur[-1]-ur[0]}")
    if m.get("dry_fullband_db") is None or abs(m["dry_fullband_db"] - ref["metrics_summary"]["dry_fullband_db"]) > 6: fails.append(f"dry level {m.get('dry_fullband_db')}")
    rows.append((d["azimuth_deg"], t.name, m.get("suppression_db"), m.get("product_hats_band_db"), m.get("dry_fullband_db"), s_mic, s_hats, (ur[-1]-ur[0]) if ur else None, fails))
azs = [r[0] for r in rows]
print(f"{'az':>5} {'trial':28s} {'supp':>6} {'hats':>6} {'dry':>6} {'micSNR':>6} {'hatsSNR':>7} {'xrun':>4}  result")
for r in sorted(rows):
    print(f"{r[0]:5.0f} {r[1]:28s} {r[2]:6.1f} {r[3]:6.1f} {r[4]:6.1f} {r[5]:6.0f} {r[6]:7.0f} {str(r[7]):>4}  {'PASS' if not r[8] else 'FAIL: ' + '; '.join(r[8])}")
expected = list(range(0, 360, 30)); dup = sorted({a for a in azs if azs.count(a) > 1})
print("missing angles:", [a for a in expected if a not in azs], " duplicates:", dup)
pass  # json.dump([{"az": r[0], "trial": r[1], "pass": not r[8], "fails": r[8]} for r in rows], open(run / ".." / ".." / "debug" / "polar_sweep_qa.json", "w"), indent=1)
