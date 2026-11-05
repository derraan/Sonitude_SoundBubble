"""Deterministic offline C++ vs Python wet-output cross-check.

Feed one ``raw_8ch.wav`` through both runtimes with identical gains; diff wet;
save report into a TrialStore trial (``kind=parity``).

CLI::

    python -m tools.measure parity --raw-8ch path/to/raw_8ch.wav --measure-root measure
    python -m tools.measure.parity --raw-8ch path/to/raw_8ch.wav --out-dir measure/_parity/take1
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tools.measure.runtime import Gains, RuntimeLaunch, run_offline_wav

try:
    from scipy.io.wavfile import read as wavread
    from scipy.io.wavfile import write as wavwrite
except ImportError:  # pragma: no cover
    wavread = None
    wavwrite = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ParityResult:
    trial_dir: Path
    run_id: str
    trial_id: str
    metrics: dict[str, Any]
    wet_cpp: Path | None
    wet_python: Path | None


def _load_wav(path: Path) -> tuple[int, np.ndarray]:
    if sf is not None:
        audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
        return int(sr), np.asarray(audio, dtype=np.float64)
    if wavread is None:
        raise RuntimeError("scipy or soundfile required")
    sr, audio = wavread(str(path))
    x = np.asarray(audio)
    if x.ndim == 1:
        x = x[:, None]
    if np.issubdtype(x.dtype, np.integer):
        denom = float(max(abs(int(np.iinfo(x.dtype).min)), int(np.iinfo(x.dtype).max)))
        x = x.astype(np.float64) / denom
    else:
        x = x.astype(np.float64, copy=False)
    return int(sr), x


def _write_wav(path: Path, sr: int, audio: np.ndarray) -> None:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    x = np.clip(x, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sf is not None:
        sf.write(str(path), x, int(sr), subtype="FLOAT")
        return
    if wavwrite is None:
        raise RuntimeError("scipy or soundfile required to write WAV")
    wavwrite(str(path), int(sr), x)


def diff_wavs(a: Path, b: Path) -> dict[str, Any]:
    sr_a, xa = _load_wav(a)
    sr_b, xb = _load_wav(b)
    n = min(xa.shape[0], xb.shape[0])
    c = min(xa.shape[1], xb.shape[1])
    da = xa[:n, :c]
    db = xb[:n, :c]
    err = da - db
    rms = float(np.sqrt(np.mean(err * err) + 1e-30))
    peak = float(np.max(np.abs(err))) if err.size else 0.0
    corr = 0.0
    flat_a = da.reshape(-1)
    flat_b = db.reshape(-1)
    if flat_a.size > 1 and float(np.std(flat_a)) > 0 and float(np.std(flat_b)) > 0:
        corr = float(np.corrcoef(flat_a, flat_b)[0, 1])
    return {
        "sr_a": sr_a,
        "sr_b": sr_b,
        "frames_compared": n,
        "channels_compared": c,
        "rms_error": rms,
        "peak_error": peak,
        "max_abs_diff": peak,
        "rms_diff": rms,
        "corr": corr,
        "near_identical": peak < 1e-4 and rms < 1e-5,
    }


def _write_parity_series_csv(path: Path, series: dict[str, Any]) -> Path:
    """Flatten wet/tap diff scalars into a two-column series CSV."""
    rows: list[dict[str, Any]] = []
    for scope in ("wet_diff", "tap_wet_diff"):
        block = series.get(scope)
        if not isinstance(block, dict):
            continue
        for key, val in block.items():
            if isinstance(val, (int, float, bool)) or val is None:
                rows.append({"scope": scope, "metric": key, "value": val})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "metric", "value"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def _ensure_trial_dir(
    *,
    measure_root: Path | None,
    run_id: str | None,
    out_dir: Path | None,
    trial_fields: dict[str, Any] | None,
) -> tuple[Path, str, str, Any]:
    """Return (trial_dir, run_id, trial_id, store_or_None)."""
    if out_dir is not None and measure_root is None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "plots").mkdir(exist_ok=True)
        return out_dir, out_dir.parent.name, out_dir.name, None

    measure_root = Path(measure_root or (REPO_ROOT / "measure"))
    try:
        from tools.tech_testbench.app.storage.trial_store import TrialStore
    except ImportError:  # pragma: no cover
        try:
            from app.storage.trial_store import TrialStore  # type: ignore
        except ImportError:
            TrialStore = None  # type: ignore

    if TrialStore is None:
        run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        run_root = measure_root / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        trial_id = f"parity_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        trial_dir = run_root / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "plots").mkdir(exist_ok=True)
        return trial_dir, run_id, trial_id, None

    store = TrialStore(measure_root)
    if not run_id:
        run_root = store.new_run(meta={"kind": "parity", **(trial_fields or {})})
        run_id = run_root.name
    paths = store.new_trial(
        run_id=run_id,
        kind="parity",
        trial_fields={
            "runtime_targets": ["cpp", "python"],
            **(trial_fields or {}),
        },
    )
    return paths.root, paths.run_id, paths.trial_id, (store, paths)


def run_parity(
    raw_8ch: str | Path,
    *,
    out_dir: str | Path | None = None,
    measure_root: str | Path | None = None,
    run_id: str | None = None,
    gains: Gains | None = None,
    zoo_dir: str = "",
    bubble_radius_m: float = 1.5,
    model_path: str = "",
    contract_path: str = "",
    soundbubble_rt_path: str = "",
    input_sr: int = 24000,
    output_sr: int = 24000,
    channel_map: str = "0,1,2,3,4,5",
    input_channels: int = 8,
    output_channels: int = 2,
    trial_fields: dict[str, Any] | None = None,
    skip_cpp: bool = False,
    skip_python: bool = False,
) -> ParityResult:
    """Replay ``raw_8ch`` through cpp + python offline; write TrialStore parity trial."""
    raw_8ch = Path(raw_8ch)
    if not raw_8ch.is_file():
        raise FileNotFoundError(raw_8ch)

    fields = dict(trial_fields or {})
    fields.setdefault("parity_raw_8ch", str(raw_8ch))
    trial_dir, run_id_out, trial_id, store_pack = _ensure_trial_dir(
        measure_root=Path(measure_root) if measure_root else None,
        run_id=run_id,
        out_dir=Path(out_dir) if out_dir else None,
        trial_fields=fields,
    )

    dest_raw = trial_dir / "raw_8ch.wav"
    if raw_8ch.resolve() != dest_raw.resolve():
        shutil.copy2(raw_8ch, dest_raw)

    gains = gains or Gains()
    base_kwargs = dict(
        pipeline_in_device=None,
        pipeline_out_device=None,
        input_sr=input_sr,
        output_sr=output_sr,
        input_channels=input_channels,
        output_channels=output_channels,
        channel_map=channel_map,
        gains=gains,
        model_path=model_path,
        contract_path=contract_path,
        zoo_dir=zoo_dir,
        bubble_radius_m=bubble_radius_m,
        use_zoo=bool(zoo_dir) and not bool(model_path),
        soundbubble_rt_path=soundbubble_rt_path,
        tap_dir="",
        profile=False,
        monitor=False,
    )

    results: dict[str, Any] = {
        "pending": False,
        "raw_8ch": str(raw_8ch),
        "gains": gains.to_dict(),
        "expected_wavs": ["raw_8ch.wav", "wet_cpp.wav", "wet_python.wav", "wet_diff.wav"],
        "wets": {},
    }

    runtimes: list[str] = []
    if not skip_cpp:
        runtimes.append("cpp")
    if not skip_python:
        runtimes.append("python")

    for runtime in runtimes:
        launch = RuntimeLaunch(runtime=runtime, **base_kwargs)
        # Optional taps beside trial for extra wet_out compare
        tap_dir = trial_dir / f"taps_{runtime}"
        tap_dir.mkdir(exist_ok=True)
        launch.tap_dir = str(tap_dir)
        launch.tap_select = "dry,wet"
        wet_path = trial_dir / f"wet_{runtime}.wav"
        print(f"[parity] running {runtime} → {wet_path.name}")
        cp = run_offline_wav(launch, input_wav=raw_8ch, output_wav=wet_path)
        (trial_dir / f"parity_{runtime}.log").write_text(
            (cp.stdout or "") + "\n" + (cp.stderr or ""), encoding="utf-8"
        )
        results["wets"][runtime] = {
            "path": str(wet_path),
            "returncode": cp.returncode,
            "stdout_tail": (cp.stdout or "")[-2000:],
            "stderr_tail": (cp.stderr or "")[-2000:],
        }

    wet_cpp = trial_dir / "wet_cpp.wav"
    wet_py = trial_dir / "wet_python.wav"
    wet_cpp_p = wet_cpp if wet_cpp.is_file() else None
    wet_py_p = wet_py if wet_py.is_file() else None

    if wet_cpp_p and wet_py_p:
        diff = diff_wavs(wet_cpp_p, wet_py_p)
        results["diff_cpp_vs_python"] = diff
        results["wet_diff"] = diff
        sr_a, xa = _load_wav(wet_cpp_p)
        _, xb = _load_wav(wet_py_p)
        n = min(xa.shape[0], xb.shape[0])
        c = min(xa.shape[1], xb.shape[1])
        _write_wav(trial_dir / "wet_diff.wav", sr_a, xa[:n, :c] - xb[:n, :c])

        tw_cpp = trial_dir / "taps_cpp" / "wet_out.wav"
        tw_py = trial_dir / "taps_python" / "wet_out.wav"
        if tw_cpp.is_file() and tw_py.is_file():
            results["tap_wet_diff"] = diff_wavs(tw_cpp, tw_py)

    series = {
        "type": "parity_wet_diff",
        "wet_diff": results.get("wet_diff"),
        "tap_wet_diff": results.get("tap_wet_diff"),
        "gains": gains.to_dict(),
    }
    plots_dir = trial_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    csv_path = plots_dir / "parity_diff.csv"
    _write_parity_series_csv(csv_path, series)
    (plots_dir / "parity_diff.json").write_text(json.dumps(series, indent=2), encoding="utf-8")

    trial_doc: dict[str, Any] = {
        "run_id": run_id_out,
        "trial_id": trial_id,
        "kind": "parity",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_targets": ["cpp", "python"],
        "parity_raw_8ch": str(raw_8ch),
        "gains": gains.to_dict(),
        "model": model_path,
        "contract": contract_path,
        "zoo_dir": zoo_dir if base_kwargs["use_zoo"] else "",
        "bubble_radius_m": bubble_radius_m,
        "channel_map": channel_map,
        "sr": {"input": input_sr, "output": output_sr},
        "artifacts": {
            "wavs": sorted(p.name for p in trial_dir.glob("*.wav")),
            "plots": ["plots/parity_diff.csv", "plots/parity_diff.json"],
            "files": sorted(p.name for p in trial_dir.glob("parity_*.log")),
        },
        "metrics_summary": results.get("wet_diff") or {},
        **fields,
    }
    trial_doc["kind"] = "parity"
    (trial_dir / "trial.json").write_text(json.dumps(trial_doc, indent=2), encoding="utf-8")
    (trial_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Compat for older callers expecting parity.json
    (trial_dir / "parity.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    if store_pack is not None:
        store, paths = store_pack
        for name in ("raw_8ch.wav", "wet_cpp.wav", "wet_python.wav", "wet_diff.wav"):
            if (trial_dir / name).is_file():
                store.register_wav(paths, name)
        store.register_plot(
            paths, stem="parity_diff", series_csv=csv_path, series_json=series
        )
        store.save_metrics(paths, results)
        store.save_trial_json(paths, {**store.load_trial_json(paths), **trial_doc})

    # Compact telemetry (never WAVs)
    try:
        from tools.measure.core import append_telemetry

        append_telemetry(
            trial_dir.parent,
            {
                "kind": "parity",
                "trial_path": str(trial_dir),
                "wet_max_abs_diff": (results.get("wet_diff") or {}).get("max_abs_diff"),
                "wet_rms_diff": (results.get("wet_diff") or {}).get("rms_diff"),
                "wet_corr": (results.get("wet_diff") or {}).get("corr"),
                "gains": gains.to_dict(),
            },
        )
    except Exception:  # noqa: BLE001
        pass

    # Optional: if dry_ref + wet_out landed (e.g. copied taps), emit suppression FR.
    try:
        from tools.measure.core import _auto_suppression_fr

        fr_plots = _auto_suppression_fr(trial_dir)
        if fr_plots:
            trial_doc.setdefault("artifacts", {}).setdefault("plots", [])
            for p in fr_plots:
                if p not in trial_doc["artifacts"]["plots"]:
                    trial_doc["artifacts"]["plots"].append(p)
            (trial_dir / "trial.json").write_text(
                json.dumps(trial_doc, indent=2), encoding="utf-8"
            )
    except Exception:  # noqa: BLE001
        pass

    return ParityResult(
        trial_dir=trial_dir,
        run_id=run_id_out,
        trial_id=trial_id,
        metrics=results,
        wet_cpp=wet_cpp_p,
        wet_python=wet_py_p,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline C++/Python wet parity cross-check")
    p.add_argument("--raw-8ch", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None, help="Legacy: write directly into this folder")
    p.add_argument("--measure-root", type=Path, default=None, help="TrialStore root (preferred)")
    p.add_argument("--run-id", default=None)
    p.add_argument("--zoo-dir", type=str, default=str(REPO_ROOT / "edge" / "zoo"))
    p.add_argument("--bubble-radius", type=float, default=1.5)
    p.add_argument("--model", type=str, default="")
    p.add_argument("--contract-path", type=str, default="")
    p.add_argument("--soundbubble-rt", type=str, default="")
    p.add_argument("--channel-map", type=str, default="0,1,2,3,4,5")
    p.add_argument("--input-sr", type=int, default=24000)
    p.add_argument("--output-sr", type=int, default=24000)
    p.add_argument("--input-gain-db", type=float, default=0.0)
    p.add_argument("--output-gain-db", type=float, default=0.0)
    p.add_argument("--mix-dry", type=float, default=0.0)
    p.add_argument("--no-output-limiter", action="store_true")
    p.add_argument("--limiter-release-ms", type=float, default=25.0)
    p.add_argument("--skip-cpp", action="store_true")
    p.add_argument("--skip-python", action="store_true")
    p.add_argument("--acoustic-profile", default="windows_pico")
    p.add_argument("--host", default="windows_testbench")
    args = p.parse_args(argv)

    if args.out_dir is None and args.measure_root is None:
        args.measure_root = REPO_ROOT / "measure"

    gains = Gains(
        input_gain_db=args.input_gain_db,
        output_gain_db=args.output_gain_db,
        mix_dry=args.mix_dry,
        output_limiter=not args.no_output_limiter,
        limiter_release_ms=args.limiter_release_ms,
    )
    result = run_parity(
        args.raw_8ch,
        out_dir=args.out_dir,
        measure_root=args.measure_root,
        run_id=args.run_id,
        gains=gains,
        zoo_dir=args.zoo_dir if not args.model else "",
        bubble_radius_m=args.bubble_radius,
        model_path=args.model,
        contract_path=args.contract_path,
        soundbubble_rt_path=args.soundbubble_rt,
        channel_map=args.channel_map,
        input_sr=args.input_sr,
        output_sr=args.output_sr,
        trial_fields={
            "acoustic_profile": args.acoustic_profile,
            "host": args.host,
        },
        skip_cpp=args.skip_cpp,
        skip_python=args.skip_python,
    )
    summary = {
        "trial_dir": str(result.trial_dir),
        "run_id": result.run_id,
        "trial_id": result.trial_id,
        "wet_diff": result.metrics.get("wet_diff"),
        "cpp_rc": (result.metrics.get("wets") or {}).get("cpp", {}).get("returncode"),
        "python_rc": (result.metrics.get("wets") or {}).get("python", {}).get("returncode"),
    }
    print(json.dumps(summary, indent=2))
    if result.wet_cpp is None or result.wet_python is None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
