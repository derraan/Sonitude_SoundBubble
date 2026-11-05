"""Replay bench takes (raw_8ch) through two ONNX models offline from a fresh state.

Same C++ runtime, same recorded gains/mic gains/channel map per take; only the model differs.
Usage: python -I debug/eval_models_on_bench.py <out_dir>
Writes NEW file <out_dir>/bench_replay.json.
"""
import json, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import welch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "tech_testbench"))
from app.analysis.audio import suppression_db  # noqa: E402

OUT = Path(sys.argv[1]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
EXE = REPO / "edge" / "cpp" / "build-local" / "soundbubble-rt.exe"
MODELS = {
    "v2_ep25": (OUT / "v2_ep25.onnx", OUT / "v2_ep25.runtime.json"),
    "ep35": (REPO / "edge/zoo/optim_sonitude_dist_count_1_5m_ep35.onnx", REPO / "edge/zoo/optim_sonitude_dist_count_1_5m_ep35.runtime.json"),
}
SETS = {
    "ess_azimuth": ("run_20261104_040645_1abaf4", lambda d: d["kind"] == "polar" and float(d["elevation_deg"]) == 0.0 and d.get("valid") is True),
    "ess_elevation": ("run_20261104_040645_1abaf4", lambda d: d["kind"] == "polar" and float(d["azimuth_deg"]) == 0.0 and d.get("valid") is True and d["created_utc"] > "2026-11-04T05:2"),
    "speech_run1": ("run_20261104_073046_3447eb", lambda d: bool(d.get("stimulus_speech"))),
    "speech_run2": ("run_20261104_075009_b48fe6", lambda d: bool(d.get("stimulus_speech"))),
}
BANDS = [(100, 300), (300, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 11000)]


def bands(x, sr):
    f, P = welch(x, sr, nperseg=4096)
    return np.array([10 * np.log10(P[(f >= lo) & (f < hi)].mean() + 1e-30) for lo, hi in BANDS])


def replay(trial: Path, d: dict, model: str) -> dict:
    onnx, contract = MODELS[model]; g = d["gains"]; tmp = Path(tempfile.mkdtemp(prefix="sbev_"))
    try:
        env = dict(os.environ, PATH=os.pathsep.join([str(EXE.parent), "C:/Strawberry/c/bin", os.environ["PATH"]]))
        argv = [str(EXE), "--model", str(onnx), "--contract-path", str(contract), "--bubble-radius", "1.5",
                "--input-sr", "24000", "--output-sr", "24000", "--input-channels", "8", "--output-channels", "1",
                "--channel-map", d["channel_map"], "--input-gain-db", str(g["input_gain_db"]), "--silence-dbfs", "-80",
                "--hops-per-io", "1", "--output-gain-db", "0", "--mix-dry", "0", "--no-output-limiter", "--wet-gate-dbfs", "off",
                "--tap-dir", str(tmp), "--tap-select", "dry,wet", "--input-wav", str(trial / "raw_8ch.wav"), "--output-wav", str(tmp / "o.wav")]
        if g.get("mic_gains"):
            argv[argv.index("--hops-per-io"):argv.index("--hops-per-io")] = ["--mic-gains", ",".join(f"{v:g}" for v in g["mic_gains"])]
        subprocess.run(argv, cwd=str(EXE.parent), env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dry, sr = sf.read(tmp / "dry_ref.wav"); wet, _ = sf.read(tmp / "wet_out.wav"); n = min(len(dry), len(wet))
        return {"suppression_db": round(float(suppression_db(dry[:n], wet[:n], sr)["suppression_db"]), 2),
                "keep_db_per_band": (bands(wet[:n], sr) - bands(dry[:n], sr)).round(1).tolist()}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


jobs = []
for set_name, (run, sel) in SETS.items():
    for tj in sorted((REPO / "measure" / run).glob("*/trial.json")):
        d = json.loads(tj.read_text(encoding="utf-8"))
        if (tj.parent / "raw_8ch.wav").is_file() and sel(d):
            for m in MODELS:
                jobs.append((set_name, tj.parent, d, m))
rows = {}
with ThreadPoolExecutor(max_workers=int(os.environ.get("SB_WORKERS", "4"))) as ex:
    futs = [(s, t, d, m, ex.submit(replay, t, d, m)) for s, t, d, m in jobs]
    for s, t, d, m, f in futs:
        key = (s, t.name)
        rows.setdefault(key, {"set": s, "trial": t.name, "distance_m": d["distance_m"], "azimuth_deg": d["azimuth_deg"],
                              "elevation_deg": d["elevation_deg"], "status": d.get("validity_status"),
                              "live_suppression_db": d.get("metrics_summary", {}).get("suppression_db")})
        rows[key][m] = f.result()
out = sorted(rows.values(), key=lambda r: (r["set"], r["distance_m"], r["elevation_deg"], r["azimuth_deg"], r["trial"]))
(OUT / "bench_replay.json").write_text(json.dumps({"protocol": "fresh state, C++ offline, recorded gains/mic gains/map, limiter+gate off",
                                                   "models": {k: str(v[0]) for k, v in MODELS.items()}, "bands_hz": BANDS, "takes": out}, indent=1), encoding="utf-8")
for r in out:
    print(f"{r['set']:14s} {r['trial'][-9:]} d{r['distance_m']:.1f} az{r['azimuth_deg']:+4.0f} el{r['elevation_deg']:+4.0f}  v2 {r['v2_ep25']['suppression_db']:5.1f}  ep35 {r['ep35']['suppression_db']:5.1f}  diff {r['v2_ep25']['suppression_db']-r['ep35']['suppression_db']:+6.1f}")
