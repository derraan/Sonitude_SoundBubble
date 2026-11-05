"""Offline polar replay: raw_8ch takes -> soundbubble-rt (offline) -> suppression.

Suppression uses the testbench analyzer's own suppression_db() on dry_ref/wet_out
taps, so numbers are comparable to live trial metrics. Read-only on measure/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "tech_testbench"))
from app.analysis.audio import suppression_db  # noqa: E402

RUN = REPO / "measure" / "run_20261104_040645_1abaf4"
EXE = REPO / "edge" / "cpp" / "build-local" / "soundbubble-rt.exe"
MODEL = REPO / "edge" / "zoo" / "optim_sonitude_dist_count_1_5m_ep35.onnx"
CONTRACT = REPO / "edge" / "zoo" / "optim_sonitude_dist_count_1_5m_ep35.runtime.json"


def takes() -> dict[float, list[Path]]:
    out: dict[float, list[Path]] = {}
    for tj in sorted(RUN.glob("polar_*/trial.json")):
        d = json.loads(tj.read_text(encoding="utf-8"))
        if d.get("valid") is True:
            out.setdefault(float(d["azimuth_deg"]), []).append(tj.parent)
    return out


def ref_gains(trial: Path) -> dict:
    return json.loads((trial / "trial.json").read_text(encoding="utf-8"))["gains"]


def run_one(trial: Path, cmap: str, keep_dir: Path | None = None) -> float:
    g = ref_gains(trial)
    tmp = Path(tempfile.mkdtemp(prefix="sbpol_", dir=os.environ.get("SB_TMP")))
    try:
        argv = [
            str(EXE), "--model", str(MODEL), "--contract-path", str(CONTRACT), "--bubble-radius", "1.5",
            "--input-sr", "24000", "--output-sr", "24000", "--input-channels", "8", "--output-channels", "1",
            "--channel-map", cmap, "--input-gain-db", str(g["input_gain_db"]), "--silence-dbfs", "-80",
            "--mic-gains", ",".join(f"{v:g}" for v in g["mic_gains"]),
            "--output-gain-db", "0", "--mix-dry", "0", "--no-output-limiter", "--wet-gate-dbfs", "off",
            "--tap-dir", str(tmp), "--tap-select", "dry,wet",
            "--input-wav", str(trial / "raw_8ch.wav"), "--output-wav", str(tmp / "out.wav"),
        ]
        env = dict(os.environ)
        # The exe is built with Strawberry MinGW; Git-bash puts an older libstdc++ first
        # on PATH (STATUS_ENTRYPOINT_NOT_FOUND). Pin the toolchain + ORT dirs first.
        env["PATH"] = os.pathsep.join([str(EXE.parent), "C:/Strawberry/c/bin", env.get("PATH", "")])
        subprocess.run(argv, cwd=str(EXE.parent), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        dry, sr = sf.read(tmp / "dry_ref.wav")
        wet, _ = sf.read(tmp / "wet_out.wav")
        n = min(len(dry), len(wet))
        s = float(suppression_db(dry[:n], wet[:n], int(sr))["suppression_db"])
        if keep_dir is not None:
            keep_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(tmp / "wet_out.wav", keep_dir / "wet_out.wav")
            shutil.copy(tmp / "dry_ref.wav", keep_dir / "dry_ref.wav")
        return s
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sweep(cmap: str, angles: list[float] | None = None, keep_root: Path | None = None) -> dict[float, float]:
    tk = takes()
    jobs = [(a, t) for a in sorted(tk) if angles is None or a in angles for t in tk[a]]
    res: dict[float, list[float]] = {}
    with ThreadPoolExecutor(max_workers=int(os.environ.get("SB_WORKERS", "6"))) as ex:
        futs = {ex.submit(run_one, t, cmap, (keep_root / cmap.replace(",", "") / t.name) if keep_root else None): a for a, t in jobs}
        for f, a in futs.items():
            res.setdefault(a, []).append(f.result())
    return {a: float(np.mean(v)) for a, v in sorted(res.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep", "screen"])
    ap.add_argument("--maps", nargs="*", default=["5,4,3,2,1,0"])
    ap.add_argument("--angles", nargs="*", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-wavs", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.mode == "screen":
        maps = [",".join(map(str, p)) for p in itertools.permutations(range(6))]
    else:
        maps = a.maps
    results = {}
    if out.exists():
        results = json.loads(out.read_text())
    for i, m in enumerate(maps):
        if m in results:
            continue
        results[m] = {str(k): v for k, v in sweep(m, a.angles, out.parent / "wavs" if a.keep_wavs else None).items()}
        if i % 10 == 0 or a.mode == "sweep":
            out.write_text(json.dumps(results, indent=1))
            print(f"[{i + 1}/{len(maps)}] {m} {results[m]}", flush=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
