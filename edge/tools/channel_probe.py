#!/usr/bin/env python3
"""Hardware probe for B5 (channel order) and B6 (inter-channel sync).

Runs on the Pi against the Pico UAC2 8-channel device. Does not modify audio
firmware or model weights.

Modes:
  tap   Guided per-mic tap → recommended --channel-map (B5).
  clap  Record while clapping in front → cross-correlation lags (B6).

Examples:
  python -m edge.tools.channel_probe --list-devices
  python -m edge.tools.channel_probe tap --device 1 --sr 48000
  python -m edge.tools.channel_probe clap --device 1 --sr 48000 --repeats 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install sounddevice first: pip install sounddevice") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Model mic names in training order (mic00 … mic05).
MODEL_MIC_NAMES = (
    "left ear (mic00)",
    "headband 1 (mic01)",
    "headband 2 (mic02)",
    "headband 3 (mic03)",
    "headband 4 (mic04)",
    "right ear (mic05)",
)


def _rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))
    return float(20.0 * np.log10(max(rms, 1e-10)))


def _record(device: int | None, channels: int, sr: int, seconds: float) -> np.ndarray:
    """Return float32 array [T, C]."""
    frames = int(round(seconds * sr))
    print(f"  recording {seconds:.2f}s @ {sr} Hz, {channels} ch …", flush=True)
    data = sd.rec(
        frames,
        samplerate=sr,
        channels=channels,
        dtype="float32",
        device=device,
        blocking=True,
    )
    return np.asarray(data, dtype=np.float32)


def cmd_list_devices(_: argparse.Namespace) -> int:
    print(sd.query_devices())
    return 0


def cmd_tap(args: argparse.Namespace) -> int:
    print("=== Channel tap probe (B5) ===")
    print("For each model mic, tap that physical mic once when prompted.")
    print("Idle levels are measured first (useful for B8 quiet-mic checks).\n")

    idle = _record(args.device, args.channels, args.sr, args.idle_sec)
    idle_db = [_rms_dbfs(idle[:, ch]) for ch in range(args.channels)]
    print("Idle dBFS per device channel:")
    for ch, db in enumerate(idle_db):
        print(f"  ch{ch}: {db:7.1f} dBFS")
    print()

    mapping: list[int | None] = [None] * 6
    used: set[int] = set()

    for model_idx, name in enumerate(MODEL_MIC_NAMES):
        input(f"[{model_idx}/5] Tap physical mic for {name}, then press Enter … ")
        # Small settle so Enter key noise decays.
        time.sleep(0.15)
        tap = _record(args.device, args.channels, args.sr, args.tap_sec)
        tap_db = [_rms_dbfs(tap[:, ch]) for ch in range(min(6, args.channels))]
        # Prefer channels 0..5 (active mics); ignore padding 6..7.
        best = int(np.argmax(tap_db))
        delta = tap_db[best] - idle_db[best]
        print(f"  peak device ch{best} = {tap_db[best]:.1f} dBFS (Δ {delta:+.1f} vs idle)")
        for ch, db in enumerate(tap_db):
            mark = " <--" if ch == best else ""
            print(f"    ch{ch}: {db:7.1f} dBFS{mark}")
        if best in used:
            print(f"  WARNING: device ch{best} already assigned to model mic {mapping.index(best)}")
        if delta < args.min_delta_db:
            print(
                f"  WARNING: Δ {delta:.1f} dB < --min-delta-db {args.min_delta_db}; "
                "tap may be too soft or wrong mic."
            )
        mapping[model_idx] = best
        used.add(best)
        print()

    if None in mapping or len(used) < 6:
        print("FAIL: could not assign a unique device channel to every model mic.")
        print(f"Partial map: {mapping}")
        return 2

    channel_map = ",".join(str(ch) for ch in mapping)
    print("=== Result ===")
    print(f"Recommended --channel-map {channel_map}")
    print("Identity map is 0,1,2,3,4,5. Use the recommended value in run_pi_live_pipeline.ps1 -ChannelMap.")
    for model_idx, name in enumerate(MODEL_MIC_NAMES):
        print(f"  model[{model_idx}] {name:24s} <- device ch{mapping[model_idx]}")
    quiet = [ch for ch, db in enumerate(idle_db[:6]) if db < -50.0]
    if quiet:
        print(
            f"Note (B8): idle channels {quiet} are below -50 dBFS. "
            "Launcher now defaults --silence-dbfs -100 so these are not zeroed."
        )
    return 0


def _xcorr_lag_samples(ref: np.ndarray, sig: np.ndarray, max_lag: int) -> tuple[float, float]:
    """Return (lag_samples, peak_corr). Positive lag => sig lags ref (arrives later)."""
    ref = ref.astype(np.float64) - float(np.mean(ref))
    sig = sig.astype(np.float64) - float(np.mean(sig))
    n = min(ref.size, sig.size)
    ref = ref[:n]
    sig = sig[:n]
    corr = np.correlate(sig, ref, mode="full")
    mid = n - 1
    lo = mid - max_lag
    hi = mid + max_lag + 1
    window = corr[lo:hi]
    peak_i = int(np.argmax(window))
    # Parabolic sub-sample interpolation around the peak.
    if 0 < peak_i < window.size - 1:
        y0, y1, y2 = float(window[peak_i - 1]), float(window[peak_i]), float(window[peak_i + 1])
        denom = y0 - 2.0 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        frac = 0.0
    lag = (peak_i - max_lag) + frac
    peak = float(window[peak_i]) / (np.linalg.norm(ref) * np.linalg.norm(sig) + 1e-12)
    return float(lag), float(peak)


def cmd_clap(args: argparse.Namespace) -> int:
    print("=== Channel sync clap probe (B6) ===")
    print("Stand ~1 m directly in front of the headset. Clap once per take when prompted.")
    print("Pass: all |lag| < 1 ms; pattern stable across repeats.\n")

    max_lag = int(round(args.max_lag_ms * 1e-3 * args.sr))
    all_lags: list[list[float]] = []

    for rep in range(args.repeats):
        input(f"Take {rep + 1}/{args.repeats}: clap once, then press Enter … ")
        time.sleep(0.1)
        rec = _record(args.device, args.channels, args.sr, args.clap_sec)
        # Use channels 0..5 only.
        active = min(6, args.channels)
        ref = rec[:, 0]
        lags: list[float] = []
        print(f"  lags vs device ch0 (max_lag={max_lag} samples / {args.max_lag_ms:.2f} ms):")
        for ch in range(active):
            lag, peak = _xcorr_lag_samples(ref, rec[:, ch], max_lag=max_lag)
            lags.append(lag)
            ms = 1000.0 * lag / args.sr
            print(f"    ch{ch}: {lag:+7.2f} samples ({ms:+7.3f} ms), corr={peak:.3f}")
        all_lags.append(lags)
        print()

    arr = np.asarray(all_lags, dtype=np.float64)  # [R, C]
    mean_lag = np.mean(arr, axis=0)
    std_lag = np.std(arr, axis=0)
    print("=== Summary (mean ± std across takes) ===")
    worst_ms = 0.0
    for ch in range(mean_lag.size):
        ms = 1000.0 * mean_lag[ch] / args.sr
        worst_ms = max(worst_ms, abs(ms))
        print(
            f"  ch{ch}: {mean_lag[ch]:+7.2f} ± {std_lag[ch]:.2f} samples "
            f"({ms:+7.3f} ms)"
        )

    fail = False
    if worst_ms > 1.0:
        print(f"FAIL: max |mean lag| {worst_ms:.3f} ms > 1.0 ms.")
        print("  Likely firmware USB ring starvation / recovery, not separate I2S buses.")
        print("  Check Pico CDC telemetry for ovr / ovr_bytes; re-flash a committed UF2 if dirty.")
        fail = True
    else:
        print(f"PASS: max |mean lag| {worst_ms:.3f} ms ≤ 1.0 ms.")

    # Mirror-pair consistency if we somehow know map; with identity, ears are ch0/ch5.
    # Report only — physical layout may differ until tap probe runs.
    if mean_lag.size >= 6:
        ear_diff = abs(mean_lag[0] - mean_lag[5])
        print(f"Ear pair |lag ch0−ch5| = {ear_diff:.2f} samples (informational; map may not be identity).")

    # Stability across repeats.
    if arr.shape[0] >= 2:
        max_std = float(np.max(std_lag))
        if max_std > 2.0:
            print(f"WARN: lag std {max_std:.2f} samples across takes — pattern not stable.")
            fail = True
        else:
            print(f"PASS: lag std ≤ {max_std:.2f} samples across takes.")

    return 2 if fail else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=int, default=None, help="sounddevice input device index")
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--sr", type=int, default=48000)
    sub = p.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list-devices", help="Print sounddevice devices")
    list_p.set_defaults(func=cmd_list_devices)

    tap_p = sub.add_parser("tap", help="Guided tap → --channel-map (B5)")
    tap_p.add_argument("--idle-sec", type=float, default=1.0)
    tap_p.add_argument("--tap-sec", type=float, default=0.8)
    tap_p.add_argument("--min-delta-db", type=float, default=6.0)
    tap_p.set_defaults(func=cmd_tap)

    clap_p = sub.add_parser("clap", help="Clap cross-correlation sync check (B6)")
    clap_p.add_argument("--clap-sec", type=float, default=1.5)
    clap_p.add_argument("--repeats", type=int, default=3)
    clap_p.add_argument("--max-lag-ms", type=float, default=2.0)
    clap_p.set_defaults(func=cmd_clap)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
