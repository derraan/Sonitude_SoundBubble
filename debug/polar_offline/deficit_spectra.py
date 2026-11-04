"""Per-band level deficit of each raw mic vs the 3 loudest mics (same take, active sweep).
Wideband test: deficit flat within FLAT_DB across 250 Hz-8 kHz AND mean deficit >= MIN_DB."""
import json, sys, numpy as np, soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt
RUN = Path("measure/run_20261104_040645_1abaf4")
BANDS = [(250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
FLAT_DB, MIN_DB = 3.0, 6.0
def band_levels(x, sr):
    w = sr // 10; tot = np.array([np.mean(x[i:i+w]**2) for i in range(0, len(x)-w, w)]) if x.ndim == 1 else None
    e = np.array([np.mean(x[i:i+w]**2, 0) for i in range(0, len(x)-w, w)]).sum(1)
    act = np.repeat(e > np.percentile(e, 30), w)[: len(x)]; act = np.concatenate([act, np.zeros(len(x) - len(act), bool)])
    out = []
    for lo, hi in BANDS:
        y = sosfilt(butter(4, (lo, hi), 'band', fs=sr, output='sos'), x, axis=0)
        out.append(10 * np.log10(np.mean(y[act]**2, 0)))
    return np.array(out)  # [band, mic]
res = {}
for tj in sorted(RUN.glob("polar_*/trial.json")):
    d = json.loads(tj.read_text(encoding="utf-8"))
    if float(d["azimuth_deg"]) != 0.0: continue
    x, sr = sf.read(tj.parent / "raw_8ch.wav", always_2d=True); L = band_levels(x[:, :6], sr)
    broad = L.mean(0); top3 = np.argsort(broad)[-3:]; ref = L[:, top3].mean(1)
    deficit = L - ref[:, None]
    rows = []
    for m in range(6):
        dfc = deficit[:, m]; flat = float(np.ptp(dfc)); mean = float(dfc.mean())
        wide = bool(flat <= FLAT_DB and -mean >= MIN_DB)
        rows.append({"dev_ch": m, "deficit_db_per_band": [round(float(v), 1) for v in dfc], "flatness_db": round(flat, 1),
                     "mean_deficit_db": round(mean, 1), "wideband_low": wide, "correction_db": round(-mean, 2) if wide else 0.0})
    res[tj.parent.name] = {"elevation_deg": d["elevation_deg"], "reference_dev_ch": sorted(int(i) for i in top3), "mics": rows}
    print(f"\n{tj.parent.name} el {d['elevation_deg']:+.0f}  ref(top3)={sorted(int(i) for i in top3)}   bands {[f'{a}-{b}' for a,b in BANDS]}")
    for r in rows:
        print(f"   dev{r['dev_ch']}: deficit {r['deficit_db_per_band']}  flat {r['flatness_db']:4.1f}  mean {r['mean_deficit_db']:6.1f}  -> {'WIDEBAND, +%.1f dB' % r['correction_db'] if r['wideband_low'] else 'keep as recorded'}")
json.dump({"bands_hz": BANDS, "flat_db": FLAT_DB, "min_db": MIN_DB, "takes": res}, open(sys.argv[1], "w"), indent=1)
