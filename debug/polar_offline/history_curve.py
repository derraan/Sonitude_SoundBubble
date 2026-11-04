"""az0 suppression vs length of real-audio history fed before the sweep (EP35, bench gains)."""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "tech_testbench"))
from app.analysis.audio import suppression_db  # noqa: E402

RUN = REPO / "measure" / "run_20261104_040645_1abaf4"
EXE = REPO / "edge" / "cpp" / "build-local" / "soundbubble-rt.exe"
ZOO = REPO / "edge" / "zoo"
AZ0 = RUN / "polar_20261104_041315_fa49"
OUT = Path(__file__).resolve().parent
g = json.loads((AZ0 / "trial.json").read_text(encoding="utf-8"))["gains"]
live = json.loads((AZ0 / "trial.json").read_text(encoding="utf-8"))["metrics_summary"]["suppression_db"]
x, sr = sf.read(AZ0 / "raw_8ch.wav", always_2d=True, dtype="float32")
hist = np.concatenate([sf.read(t / "raw_8ch.wav", always_2d=True, dtype="float32")[0]
                       for t in sorted(RUN.glob("polar_*")) if t.name != AZ0.name])


def run(inp, pre_n, keep=None):
    tmp = Path(tempfile.mkdtemp(prefix="sbhc_"))
    try:
        sf.write(tmp / "in.wav", inp, sr, subtype="FLOAT")
        env = dict(os.environ, PATH=os.pathsep.join([str(EXE.parent), "C:/Strawberry/c/bin", os.environ["PATH"]]))
        subprocess.run([str(EXE), "--model", str(ZOO / "optim_sonitude_dist_count_1_5m_ep35.onnx"),
                        "--contract-path", str(ZOO / "optim_sonitude_dist_count_1_5m_ep35.runtime.json"),
                        "--bubble-radius", "1.5", "--input-sr", "24000", "--output-sr", "24000", "--input-channels", "8",
                        "--output-channels", "1", "--channel-map", "5,4,3,2,1,0", "--input-gain-db", str(g["input_gain_db"]),
                        "--silence-dbfs", "-80", "--mic-gains", ",".join(f"{v:g}" for v in g["mic_gains"]), "--hops-per-io", "1",
                        "--output-gain-db", "0", "--mix-dry", "0", "--no-output-limiter", "--wet-gate-dbfs", "off",
                        "--tap-dir", str(tmp), "--tap-select", "dry,wet", "--tap-max-seconds", "900", "--tap-max-mb", "4096",
                        "--input-wav", str(tmp / "in.wav"), "--output-wav", str(tmp / "o.wav")],
                       cwd=str(EXE.parent), env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dry, _ = sf.read(tmp / "dry_ref.wav"); wet, _ = sf.read(tmp / "wet_out.wav")
        dry, wet = dry[pre_n:], wet[pre_n:]; n = min(len(dry), len(wet))
        if keep:
            sf.write(OUT / keep, wet[:n], sr, subtype="FLOAT")
        return float(suppression_db(dry[:n], wet[:n], sr)["suppression_db"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


rows = []
for secs in (0, 15, 30, 45, 60, 90, 120, 150, 167):
    n = int(min(secs, len(hist) / sr) * sr)
    s = run(np.concatenate([hist[:n], x]), n, keep="az0_wet_hist167.wav" if secs == 167 else None)
    rows.append({"history_s": secs, "suppression_db": s}); print(secs, round(s, 1), flush=True)
json.dump({"live_db": live, "rows": rows}, open(OUT / "history_curve.json", "w"), indent=1)
