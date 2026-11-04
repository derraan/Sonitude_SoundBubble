"""Mic array calibration / bit-alignment probe (INMP441 6ch over Pico UAC2).

Origin: debug session 5cff74 (2026-10-28). Captures raw audio from the Pico
USB mic array and reports, per USB channel:
  - I2S bit alignment (data LSB position; detects half/double-level bit slips)
  - signal levels (p50 background, p99/p99.9 transients, peak, RMS)
  - clipping / wrap events
  - suggested per-channel calibration gain to equalise module sensitivity

Writes NDJSON to the log path (one record per channel + summary), prints a
human-readable table, and emits sibling ``calibration.json`` for firmware
handoff (``k_mic_cal_gain[6]``).

Modes:
  - Clap (default): record only; operator makes noise.
  - Stimulus sweep: ``--stimulus`` + ``--out-device`` + ``--in-device`` plays
    the Setup ESS while capturing (same WAV as Raw polar / Algorithm).

Usage:
  python tools/mic_calibration_probe.py [run_id] [--log PATH] [--secs N] [--rate HZ]
  python tools/mic_calibration_probe.py [run_id] --stimulus WAV --out-device N --in-device N \\
      [--wav-out PATH] [--calibration-json PATH]

Notes:
  - Close Audacity / other users first (WDM-KS/WASAPI-exclusive needs the device).
  - Best raw capture is the native-rate 32-bit debug firmware; the 16-bit
    production build also works for level matching (alignment fields N/A).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Allow ``python tools/mic_calibration_probe.py`` (script argv[0]) to import ``tools.*``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import sounddevice as sd

DEFAULT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "debug-5cff74.log")
NAME_HINTS = ("inmp441", "6 active", "pico", "mic array")


def make_logger(log_path, run_id):
    def log(hyp, msg, data):
        rec = {
            "runId": run_id,
            "hypothesisId": hyp,
            "location": "mic_calibration_probe.py",
            "message": msg,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    return log


def find_device(channels):
    apis = sd.query_hostapis()
    cands = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] >= channels and any(k in d["name"].lower() for k in NAME_HINTS):
            api = apis[d["hostapi"]]["name"]
            rank = 0 if "wdm-ks" in api.lower() else 1 if "wasapi" in api.lower() else 2
            cands.append((rank, i, d["name"], api))
    cands.sort()
    return cands


def record(dev, api, rate, channels, secs, log):
    attempts = []
    if "wdm-ks" in api.lower():
        attempts.append(("wdmks_int32", "int32", None))
    if "wasapi" in api.lower():
        attempts.append(("wasapi_exclusive_int32", "int32", sd.WasapiSettings(exclusive=True)))
    attempts += [("shared_int32", "int32", None), ("shared_float32", "float32", None)]
    for mode, dtype, extra in attempts:
        try:
            x = sd.rec(
                int(rate * secs),
                samplerate=rate,
                channels=channels,
                dtype=dtype,
                device=dev,
                extra_settings=extra,
                blocking=True,
            )
            if dtype == "float32":
                x = np.round(x.astype(np.float64) * 2147483648.0).clip(-(2**31), 2**31 - 1).astype(np.int64)
            return mode, x.astype(np.int64)
        except Exception as e:
            log("E", "record_attempt_failed", {"mode": mode, "err": str(e)})
    return None, None


def _file_sha256_prefix(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def record_with_stimulus(
    *,
    stimulus_path: Path,
    out_device: int,
    in_device: int,
    rate: int,
    channels: int,
    post_ms: float,
    log,
    play_channels: str = "both",
) -> tuple[str, np.ndarray, dict]:
    """Play Setup ESS on out_device while capturing pipeline_in → int64 [T, C]."""
    from tools.measure.runtime import sleep_ms
    from tools.measure.stimulus import StimulusPlayer

    player = StimulusPlayer(stimulus_path, device=out_device, play_channels=play_channels)
    chunks: list[np.ndarray] = []

    def _cb(indata, _frames, _time_info, status) -> None:  # noqa: ANN001
        _ = status
        chunks.append(np.asarray(indata, dtype=np.float32).copy())

    stream = sd.InputStream(
        samplerate=rate,
        channels=channels,
        device=in_device,
        dtype="float32",
        callback=_cb,
    )
    try:
        stream.start()
        sleep_ms(50.0)
        player.play_blocking()
        sleep_ms(post_ms)
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    if not chunks:
        raise RuntimeError("no input frames captured during stimulus playback")
    audio_f = np.concatenate(chunks, axis=0)
    # Scale float ≈[-1,1] to int32 domain for shared analyse() path.
    x = np.round(audio_f.astype(np.float64) * 2147483648.0).clip(-(2**31), 2**31 - 1).astype(np.int64)
    meta = {
        **player.as_dict(),
        "in_device": in_device,
        "capture_rate": rate,
        "capture_channels": channels,
        "capture_frames": int(x.shape[0]),
        "stimulus_sha256_prefix": _file_sha256_prefix(Path(stimulus_path)),
    }
    log("E", "stimulus_capture", meta)
    return "stimulus_playrec_float32", x, meta


def _levels_from_abs(a_cols: list[np.ndarray], active_ch: int, method: str) -> np.ndarray:
    """Per-channel level scalars used for gain matching."""
    if method == "rms":
        return np.array(
            [float(np.sqrt(np.mean(a.astype(np.float64) ** 2) + 1e-20)) for a in a_cols[:active_ch]]
        )
    return np.array([float(np.percentile(a.astype(np.float64), 99)) for a in a_cols[:active_ch]])


def analyse(x, channels, active_ch, log, *, level_method: str = "p99"):
    u = (x & 0xFFFFFFFF).astype(np.uint64)
    peaks = np.abs(x).max(axis=0)
    abs_cols = [np.abs(x[:, c]) for c in range(channels)]
    for c in range(channels):
        col = x[:, c]
        a = abs_cols[c]
        adj = [
            round(float((((u[:, c] >> b) & 1) == ((u[:, c] >> (b + 1)) & 1)).mean()), 4)
            for b in range(14)
        ]
        hold_lsb = next(
            (
                b + 1
                for b in range(13, -1, -1)
                if adj[b] > 0.98 and all(a2 > 0.98 for a2 in adj[: b + 1])
            ),
            None,
        )
        log(
            "A",
            "channel_fingerprint",
            {
                "usb_ch": c,
                "slot": ("L" if c % 2 == 0 else "R") if c < active_ch else "pad",
                "sd_lane": c // 2 if c < active_ch else None,
                "inferred_data_lsb_bit": hold_lsb,
                "adj_agree_b_b1_0_13": adj,
                "p50": float(np.percentile(a, 50)),
                "p99": float(np.percentile(a, 99)),
                "p999": float(np.percentile(a, 99.9)),
                "peak": int(peaks[c]),
                "rms": float(np.sqrt((col.astype(np.float64) ** 2).mean())),
                "wrap_events": int((np.abs(np.diff(col)) > 2**31).sum()),
                "frac_near_fullscale": float((a > 2**30).mean()),
            },
        )
    # Calibration: equalise active channels to the quietest (headroom-safe, gain <= 1).
    levels = _levels_from_abs(abs_cols, active_ch, level_method)
    target = float(levels.min()) if levels.size else 1.0
    gains = [round(float(target / p) if p > 0 else 1.0, 4) for p in levels]
    quietest = int(np.argmin(levels)) if levels.size else 0
    level_db_rel = [
        round(float(20.0 * np.log10(max(p, 1e-20) / max(target, 1e-20))), 2) for p in levels
    ]
    spread = (
        round(float(20 * np.log10(levels.max() / max(levels.min(), 1e-20))), 2)
        if levels.size and levels.min() > 0
        else 0.0
    )
    summary = {
        "level_method": level_method,
        "target_level": target,
        "level_per_ch": [round(float(p), 1) for p in levels],
        "p99_per_ch": [round(float(np.percentile(abs_cols[c], 99)), 1) for c in range(active_ch)],
        "k_mic_cal_gain": gains,
        "spread_db_before": spread,
        "level_db_rel": level_db_rel,
        "ref_ch": quietest,
    }
    # Keep legacy key name for older log consumers.
    summary["target_p99"] = summary["target_level"] if level_method == "p99" else float(
        np.percentile(abs_cols[quietest], 99)
    )
    log("D", "suggested_calibration", summary)
    return gains, summary


def write_calibration_json(
    path: Path,
    *,
    method: str,
    summary: dict,
    rate_hz: int,
    active_channels: int,
    stimulus_path: str | None = None,
    stimulus_sha256_prefix: str | None = None,
    run_id: str | None = None,
) -> None:
    doc = {
        "method": method,
        "run_id": run_id,
        "k_mic_cal_gain": summary["k_mic_cal_gain"],
        "spread_db_before": summary["spread_db_before"],
        "level_db_rel": summary["level_db_rel"],
        "ref_ch": summary["ref_ch"],
        "level_method": summary["level_method"],
        "level_per_ch": summary["level_per_ch"],
        "stimulus_path": stimulus_path,
        "stimulus_sha256_prefix": stimulus_sha256_prefix,
        "rate_hz": rate_hz,
        "active_channels": active_channels,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _maybe_write_wav(path: Path | None, x: np.ndarray, rate: int) -> None:
    if path is None:
        return
    try:
        from scipy.io.wavfile import write as wavwrite
    except ImportError:
        print("scipy missing — skip WAV write:", path)
        return
    # Store as float32 ≈[-1,1] for Library / offline tools.
    audio = (x.astype(np.float64) / 2147483648.0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavwrite(str(path), rate, audio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?", default="run1")
    ap.add_argument("--log", default=os.path.abspath(DEFAULT_LOG))
    ap.add_argument("--secs", type=float, default=12.0)
    ap.add_argument("--rate", type=int, default=24000)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--active", type=int, default=6)
    ap.add_argument("--stimulus", type=str, default="", help="Setup ESS WAV (enables play+record)")
    ap.add_argument("--out-device", type=int, default=None, help="Stimulus out PortAudio device")
    ap.add_argument("--in-device", type=int, default=None, help="Pico mic PortAudio device")
    ap.add_argument("--post-ms", type=float, default=200.0, help="Post-stimulus capture tail (ms)")
    ap.add_argument("--wav-out", type=str, default="", help="Optional levels_8ch.wav path")
    ap.add_argument(
        "--stimulus-play-channels",
        type=str,
        default="both",
        choices=("both", "left", "right"),
        help="Which stereo DAC legs to drive (mute the other). ESS *_refL → left.",
    )
    ap.add_argument(
        "--calibration-json",
        type=str,
        default="",
        help="Path for calibration.json (default: beside --log)",
    )
    args = ap.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = make_logger(str(log_path), args.run_id)

    stim = args.stimulus.strip()
    sweep = bool(stim)
    if sweep and (args.out_device is None or args.in_device is None):
        print("Stimulus mode requires --stimulus, --out-device, and --in-device")
        sys.exit(2)
    if not sweep and (args.out_device is not None or args.in_device is not None):
        print("Clap mode ignores --out-device/--in-device; omit them or pass --stimulus")

    stim_meta: dict = {}
    if sweep:
        stim_path = Path(stim)
        if not stim_path.is_file():
            print(f"Stimulus WAV not found: {stim_path}")
            sys.exit(1)
        print(
            f"Playing stimulus + recording from in={args.in_device} out={args.out_device} "
            f"@ {args.rate} Hz  play_channels={args.stimulus_play_channels} ..."
        )
        mode, x, stim_meta = record_with_stimulus(
            stimulus_path=stim_path,
            out_device=int(args.out_device),
            in_device=int(args.in_device),
            rate=args.rate,
            channels=args.channels,
            post_ms=args.post_ms,
            log=log,
            play_channels=args.stimulus_play_channels,
        )
        # Drop first ~50 ms settle (already partly outside play); trim 1 s if long enough.
        drop = min(args.rate, max(0, x.shape[0] // 20))
        x = x[drop:]
        level_method = "rms"
        method_name = "stimulus_sweep"
    else:
        cands = find_device(args.channels)
        log("E", "device_candidates", {"cands": [c[1:] for c in cands]})
        if not cands:
            print("Pico mic device not found")
            sys.exit(1)
        _, dev, name, api = cands[0]
        print(f"Recording {args.secs}s from [{dev}] {name} ({api}) - clap now...")
        mode, x = record(dev, api, args.rate, args.channels, args.secs, log)
        if x is None:
            print("All capture modes failed (is Audacity/another app holding the device?)")
            sys.exit(1)
        x = x[args.rate :]  # drop first second (startup / recovery blend)
        log(
            "E",
            "capture_mode",
            {"mode": mode, "device": name, "hostapi": api, "frames": int(len(x))},
        )
        level_method = "p99"
        method_name = "clap"

    log(
        "E",
        "capture_mode",
        {"mode": mode, "frames": int(len(x)), "method": method_name},
    )

    wav_out = Path(args.wav_out) if args.wav_out.strip() else None
    _maybe_write_wav(wav_out, x, args.rate)

    gains, summary = analyse(x, args.channels, args.active, log, level_method=level_method)

    cal_path = (
        Path(args.calibration_json)
        if args.calibration_json.strip()
        else log_path.with_name("calibration.json")
    )
    write_calibration_json(
        cal_path,
        method=method_name,
        summary=summary,
        rate_hz=args.rate,
        active_channels=args.active,
        stimulus_path=str(Path(stim).resolve()) if sweep else None,
        stimulus_sha256_prefix=stim_meta.get("stimulus_sha256_prefix") if sweep else None,
        run_id=args.run_id,
    )

    print("Suggested k_mic_cal_gain[6] =", gains)
    print(f"spread_db_before={summary['spread_db_before']}  level_db_rel={summary['level_db_rel']}")
    print("Logged to", args.log)
    print("calibration.json ->", cal_path)
    if wav_out:
        print("WAV ->", wav_out)


if __name__ == "__main__":
    main()
