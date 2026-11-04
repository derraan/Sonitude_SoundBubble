"""Offline fresh-state replay of the speech takes in a run (same gains/map/model as live).
Usage: python debug/speech_offline_replay.py <run_name> <out_subdir>
Writes NEW files: measure/<run>/analysis/<out_subdir>/offline_replay.json
"""
import json, os, subprocess, sys, tempfile, shutil
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import welch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "tech_testbench"))
from app.analysis.audio import suppression_db  # noqa: E402

RUN = REPO / "measure" / sys.argv[1]
OUT = RUN / "analysis" / sys.argv[2]; OUT.mkdir(parents=True, exist_ok=True)
EXE = REPO / "edge" / "cpp" / "build-local" / "soundbubble-rt.exe"
BANDS = [(100, 300), (300, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 11000)]


def bands(x, sr):
    f, P = welch(x, sr, nperseg=4096)
    return np.array([10 * np.log10(P[(f >= lo) & (f < hi)].mean() + 1e-30) for lo, hi in BANDS])


res = []
for t in sorted(RUN.glob("algorithm_*")):
    d = json.loads((t / "trial.json").read_text(encoding="utf-8"))
    if not d.get("stimulus_speech") or not (t / "raw_8ch.wav").is_file():
        continue
    g = d["gains"]; tmp = Path(tempfile.mkdtemp(prefix="sbsp_"))
    try:
        env = dict(os.environ, PATH=os.pathsep.join([str(EXE.parent), "C:/Strawberry/c/bin", os.environ["PATH"]]))
        subprocess.run([str(EXE), "--model", d["model"], "--contract-path", d["contract"], "--bubble-radius", str(d.get("bubble_radius_m", 1.5)),
                        "--input-sr", "24000", "--output-sr", "24000", "--input-channels", "8", "--output-channels", "1",
                        "--channel-map", d["channel_map"], "--input-gain-db", str(g["input_gain_db"]), "--silence-dbfs", "-80",
                        "--mic-gains", ",".join(f"{v:g}" for v in g["mic_gains"]), "--hops-per-io", "1",
                        "--output-gain-db", "0", "--mix-dry", "0", "--no-output-limiter", "--wet-gate-dbfs", "off",
                        "--tap-dir", str(tmp), "--tap-select", "dry,wet", "--input-wav", str(t / "raw_8ch.wav"), "--output-wav", str(tmp / "o.wav")],
                       cwd=str(EXE.parent), env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dry, sr = sf.read(tmp / "dry_ref.wav"); wet, _ = sf.read(tmp / "wet_out.wav"); n = min(len(dry), len(wet))
        s = float(suppression_db(dry[:n], wet[:n], sr)["suppression_db"])
        keep = (bands(wet[:n], sr) - bands(dry[:n], sr)).round(1).tolist()
        res.append({"trial": t.name, "distance_m": d["distance_m"], "elevation_deg": d["elevation_deg"],
                    "live_suppression_db": d["metrics_summary"]["suppression_db"], "offline_fresh_suppression_db": round(s, 2), "offline_keep_db_per_band": keep})
        print(f"{t.name[-9:]} {d['distance_m']:.1f} m el {d['elevation_deg']:+4.0f}  live {d['metrics_summary']['suppression_db']:5.1f}  offline-fresh {s:5.1f}  keep {keep}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
(OUT / "offline_replay.json").write_text(json.dumps({"protocol": "fresh state (reset + 12 ORT warm-up runs), same gains/map/model as live, limiter/gate off", "bands_hz": BANDS, "takes": res}, indent=1), encoding="utf-8")
