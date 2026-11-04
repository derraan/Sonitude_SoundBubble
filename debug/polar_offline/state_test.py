"""Does EP35's output on the az0 sweep depend on recurrent-state history / hops_per_io?"""
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
g = json.loads((AZ0 / "trial.json").read_text(encoding="utf-8"))["gains"]
live = json.loads((AZ0 / "trial.json").read_text(encoding="utf-8"))["metrics_summary"]["suppression_db"]

x, sr = sf.read(AZ0 / "raw_8ch.wav", always_2d=True, dtype="float32")
tail = x[-int(0.4 * sr):]
prev, _ = sf.read(RUN / "polar_20261104_044933_7fb5" / "raw_8ch.wav", always_2d=True, dtype="float32")
histories = {
    "fresh": np.zeros((0, 8), np.float32),
    "silence30s": np.zeros((30 * sr, 8), np.float32),
    "tailnoise60s": np.tile(tail, (int(60 / 0.4) + 1, 1))[: 60 * sr],
    "prev_take330": prev,
}


def run(inp: np.ndarray, pre_n: int, hops: int) -> float:
    tmp = Path(tempfile.mkdtemp(prefix="sbst_"))
    try:
        sf.write(tmp / "in.wav", inp, sr, subtype="FLOAT")
        env = dict(os.environ, PATH=os.pathsep.join([str(EXE.parent), "C:/Strawberry/c/bin", os.environ["PATH"]]))
        subprocess.run([str(EXE), "--model", str(ZOO / "optim_sonitude_dist_count_1_5m_ep35.onnx"),
                        "--contract-path", str(ZOO / "optim_sonitude_dist_count_1_5m_ep35.runtime.json"),
                        "--bubble-radius", "1.5", "--input-sr", "24000", "--output-sr", "24000", "--input-channels", "8",
                        "--output-channels", "1", "--channel-map", "5,4,3,2,1,0", "--input-gain-db", str(g["input_gain_db"]),
                        "--silence-dbfs", "-80", "--mic-gains", ",".join(f"{v:g}" for v in g["mic_gains"]),
                        "--hops-per-io", str(hops), "--output-gain-db", "0", "--mix-dry", "0", "--no-output-limiter",
                        "--wet-gate-dbfs", "off", "--tap-dir", str(tmp), "--tap-select", "dry,wet", "--tap-max-seconds", "900", "--tap-max-mb", "4096",
                        "--input-wav", str(tmp / "in.wav"), "--output-wav", str(tmp / "o.wav")],
                       cwd=str(EXE.parent), env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dry, _ = sf.read(tmp / "dry_ref.wav"); wet, _ = sf.read(tmp / "wet_out.wav")
        dry, wet = dry[pre_n:], wet[pre_n:]
        n = min(len(dry), len(wet))
        return float(suppression_db(dry[:n], wet[:n], sr)["suppression_db"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if len(sys.argv) == 1: print(f"live az0 suppression: {live:.1f} dB")
for hops in ((1, 2) if len(sys.argv) == 1 else ()):
    for name, h in histories.items():
        s = run(np.concatenate([h, x]), len(h), hops)
        print(f"hops_per_io={hops}  history={name:14s} suppression={s:6.1f} dB  (offline - live {s - live:+.1f})", flush=True)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "long":
    others = [t for t in sorted(RUN.glob("polar_*")) if t.name != AZ0.name]
    clips = [sf.read(t / "raw_8ch.wav", always_2d=True, dtype="float32")[0] for t in others]
    hist = np.concatenate(clips)
    for mins in (1, 2, 3):
        h = hist[: int(mins * 60 * sr)]
        s = run(np.concatenate([h, x]), len(h), 1)
        print(f"history = {len(h)/sr:5.0f} s of real takes -> az0 suppression {s:6.1f} dB (live {live:.1f})", flush=True)
