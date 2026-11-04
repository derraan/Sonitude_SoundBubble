"""Offline elevation evaluation with per-pose mic-level normalization (EP35).

For every elevation take in run_20261104_040645_1abaf4:
  * measure each raw mic's active-sweep full-band RMS (pre-gain),
  * per-pose gains -> every mic at TARGET_DBFS (then +15 dB input gain, as on the bench),
  * replay through soundbubble-rt offline from a defined model state:
      - "fresh+preroll": reset + identical 5 s room-noise preroll (primary)
      - "fresh":         reset, no preroll (cross-check)
  * control: same protocol with the bench mic gains (dev ch4 hot in off-axis takes).
Suppression uses the testbench analyzer (300-3400 Hz) on the sweep portion only.
Writes NEW files only (results.json/.csv, figures) to OUT.
"""
import csv, json, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import welch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "tech_testbench"))
from app.analysis.audio import suppression_db  # noqa: E402

RUN = REPO / "measure" / "run_20261104_040645_1abaf4"
OUT = RUN / "analysis" / "offline_elev_norm_20261104"
EXE = REPO / "edge" / "cpp" / "build-local" / "soundbubble-rt.exe"
ZOO = REPO / "edge" / "zoo"
TARGET_DBFS = -19.0
INPUT_GAIN_DB = 15.0
PREROLL_S = 5.0
BANDS = [(100, 300), (300, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 11000)]


def elevation_takes():
    out = []
    for tj in sorted(RUN.glob("polar_*/trial.json")):
        d = json.loads(tj.read_text(encoding="utf-8"))
        if float(d["azimuth_deg"]) == 0.0 and d.get("valid") is True:
            out.append((float(d["elevation_deg"]), tj.parent, d))
    return out


def active_levels(x, sr):
    w = sr // 10
    e = np.array([np.mean(x[i:i + w] ** 2, 0) for i in range(0, len(x) - w, w)])
    act = e.sum(1) > np.percentile(e.sum(1), 30)  # sweep span (drops pre/post silence)
    return 10 * np.log10(e[act].mean(0)), act


def band_keep(dry, wet, sr):
    f, Pd = welch(dry, sr, nperseg=4096); _, Pw = welch(wet, sr, nperseg=4096)
    return [float(10 * np.log10(Pw[(f >= lo) & (f < hi)].mean() / Pd[(f >= lo) & (f < hi)].mean())) for lo, hi in BANDS]


def run(raw, sr, gains, preroll):
    tmp = Path(tempfile.mkdtemp(prefix="sbel_"))
    try:
        inp = np.concatenate([preroll, raw]) if len(preroll) else raw
        sf.write(tmp / "in.wav", inp, sr, subtype="FLOAT")
        env = dict(os.environ, PATH=os.pathsep.join([str(EXE.parent), "C:/Strawberry/c/bin", os.environ["PATH"]]))
        subprocess.run([str(EXE), "--model", str(ZOO / "optim_sonitude_dist_count_1_5m_ep35.onnx"),
                        "--contract-path", str(ZOO / "optim_sonitude_dist_count_1_5m_ep35.runtime.json"),
                        "--bubble-radius", "1.5", "--input-sr", "24000", "--output-sr", "24000", "--input-channels", "8",
                        "--output-channels", "1", "--channel-map", "5,4,3,2,1,0", "--input-gain-db", str(INPUT_GAIN_DB),
                        "--silence-dbfs", "-80", "--mic-gains", ",".join(f"{g:.6g}" for g in gains), "--hops-per-io", "1",
                        "--output-gain-db", "0", "--mix-dry", "0", "--no-output-limiter", "--wet-gate-dbfs", "off",
                        "--tap-dir", str(tmp), "--tap-select", "dry,wet,mapped",
                        "--input-wav", str(tmp / "in.wav"), "--output-wav", str(tmp / "o.wav")],
                       cwd=str(EXE.parent), env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        n0 = len(preroll)
        dry, _ = sf.read(tmp / "dry_ref.wav"); wet, _ = sf.read(tmp / "wet_out.wav")
        mapped, _ = sf.read(tmp / "mapped_6ch.wav", always_2d=True)
        dry, wet, mapped = dry[n0:], wet[n0:], mapped[n0:]
        n = min(len(dry), len(wet))
        s = suppression_db(dry[:n], wet[:n], sr)
        lv, _ = active_levels(mapped * 10 ** (INPUT_GAIN_DB / 20), sr)  # mapped tap is pre input-gain
        return {"suppression_db": float(s["suppression_db"]), "band_keep_db": band_keep(dry[:n], wet[:n], sr),
                "model_in_mic_dbfs": [round(float(v), 2) for v in lv], "model_in_spread_db": float(np.ptp(lv)),
                "wet": wet[:n].astype(np.float32)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    takes = elevation_takes()
    ref = json.loads((RUN / "polar_20261104_041315_fa49" / "trial.json").read_text(encoding="utf-8"))
    bench_gains = [float(v) for v in ref["gains"]["mic_gains"]]
    x0, sr = sf.read(RUN / "polar_20261104_041315_fa49" / "raw_8ch.wav", always_2d=True, dtype="float32")
    x330, _ = sf.read(RUN / "polar_20261104_044933_7fb5" / "raw_8ch.wav", always_2d=True, dtype="float32")
    loop = lambda seg, secs: np.tile(seg, (int(secs / (len(seg) / sr)) + 1, 1))[: int(secs * sr)]
    tail0, tail330 = x0[-int(0.4 * sr):], x330[-int(0.4 * sr):]
    protocols = {
        "tail5s": loop(tail0, 5.0),
        "silence5s": np.zeros((int(5 * sr), 8), np.float32),
        "tail330_5s": loop(tail330, 5.0),
        "tail10s": loop(tail0, 10.0),
        "fresh": np.zeros((0, 8), np.float32),
    }
    jobs = []
    for el, tdir, d in takes:
        raw, _ = sf.read(tdir / "raw_8ch.wav", always_2d=True, dtype="float32")
        lv, _ = active_levels(raw[:, :6], sr)
        norm_gains = [float(10 ** ((TARGET_DBFS - v) / 20)) for v in lv]
        meta = {"elevation_deg": el, "trial": tdir.name, "status": d.get("validity_status"),
                "live_suppression_db": d["metrics_summary"]["suppression_db"], "raw_active_dbfs": [round(float(v), 2) for v in lv],
                "per_pose_gains": [round(g, 4) for g in norm_gains], "bench_gains": bench_gains}
        for gname, gains in (("norm", norm_gains), ("bench", bench_gains)):
            for pname, pre in protocols.items():
                jobs.append((meta, f"{gname}|{pname}", raw, gains, pre))
    with ThreadPoolExecutor(max_workers=int(os.environ.get("SB_WORKERS", "3"))) as ex:
        futs = [(m, c, ex.submit(run, r, sr, g, p)) for m, c, r, g, p in jobs]
        results = []
        for m, c, f in futs:
            r = f.result(); wet = r.pop("wet")
            if c == "norm|tail5s":
                (OUT / "wavs").mkdir(exist_ok=True)
                sf.write(OUT / "wavs" / f"{m['trial']}_el{int(m['elevation_deg']):+d}_{c.replace('|', '_')}_wet.wav", wet, sr, subtype="FLOAT")
            results.append({**m, "condition": c, **r})
    results.sort(key=lambda r: (r["condition"], r["elevation_deg"], r["trial"]))
    (OUT / "results.json").write_text(json.dumps({"target_dbfs": TARGET_DBFS, "input_gain_db": INPUT_GAIN_DB,
                                                  "state_protocols": {"tail5s": "reset + 5 s looped 0.4 s room tail of polar_20261104_041315_fa49", "silence5s": "reset + 5 s digital silence", "tail330_5s": "reset + 5 s looped room tail of polar_20261104_044933_7fb5", "tail10s": "reset + 10 s looped room tail (fa49)", "fresh": "reset (12 ORT warm-up runs, state zeroed), no preroll"},
                                                  "model": "optim_sonitude_dist_count_1_5m_ep35", "channel_map": "5,4,3,2,1,0",
                                                  "results": results}, indent=1), encoding="utf-8")
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "elevation_deg", "trial", "status", "suppression_db", "live_suppression_db", "model_in_spread_db"]
                   + [f"keep_{lo}_{hi}" for lo, hi in BANDS])
        for r in results:
            w.writerow([r["condition"], r["elevation_deg"], r["trial"], r["status"], round(r["suppression_db"], 2),
                        round(r["live_suppression_db"], 2), round(r["model_in_spread_db"], 2)] + [round(v, 2) for v in r["band_keep_db"]])
    for r in results[:0]:
        print(f"{r['condition']:20s} el {r['elevation_deg']:+4.0f} {r['trial'][-9:]} supp {r['suppression_db']:6.1f} (live {r['live_suppression_db']:5.1f}) spread {r['model_in_spread_db']:4.1f}")


if __name__ == "__main__":
    main()
