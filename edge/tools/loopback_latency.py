#!/usr/bin/env python3
"""Measure round-trip click latency through a sounddevice in/out pair.

Plays a short impulse on the output device while recording the input device,
then reports the cross-correlation peak delay in milliseconds.

Physical loopback (DAC -> ADC cable) or acoustic path both work. Run on the Pi:

  python -m edge.tools.loopback_latency --input-device 1 --output-device 0 \\
      --input-sr 44100 --output-sr 48000 --channels-out 2 --channels-in 1
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("sounddevice is required") from exc


def _cross_latency_ms(tx: np.ndarray, rx: np.ndarray, sr: float) -> tuple[float, float]:
    """Return (latency_ms, peak_correlation) from mono tx/rx at the same rate."""
    tx = np.asarray(tx, dtype=np.float64).reshape(-1)
    rx = np.asarray(rx, dtype=np.float64).reshape(-1)
    n = min(tx.shape[0], rx.shape[0])
    tx = tx[:n] - tx[:n].mean()
    rx = rx[:n] - rx[:n].mean()
    corr = np.correlate(rx, tx, mode="full")
    lag = int(np.argmax(corr) - (n - 1))
    peak = float(corr[np.argmax(corr)] / (np.linalg.norm(tx) * np.linalg.norm(rx) + 1e-12))
    return 1000.0 * float(lag) / float(sr), peak


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-device", type=int, required=True)
    p.add_argument("--output-device", type=int, required=True)
    p.add_argument("--input-sr", type=int, default=48000)
    p.add_argument("--output-sr", type=int, default=48000)
    p.add_argument("--channels-in", type=int, default=1)
    p.add_argument("--channels-out", type=int, default=2)
    p.add_argument("--duration-sec", type=float, default=1.0)
    p.add_argument("--click-delay-sec", type=float, default=0.2)
    p.add_argument("--click-samples", type=int, default=8)
    p.add_argument("--amplitude", type=float, default=0.8)
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    out_n = int(round(args.duration_sec * args.output_sr))
    in_n = int(round(args.duration_sec * args.input_sr))
    click_at = int(round(args.click_delay_sec * args.output_sr))
    tx = np.zeros((out_n, args.channels_out), dtype=np.float32)
    tx[click_at : click_at + args.click_samples, :] = float(args.amplitude)

    print(
        f"[loopback] play out={args.output_device}@{args.output_sr} "
        f"rec in={args.input_device}@{args.input_sr} for {args.duration_sec:.2f}s"
    )
    # Separate streams: start recording slightly before playback.
    rec = sd.rec(
        in_n,
        samplerate=args.input_sr,
        channels=args.channels_in,
        dtype="float32",
        device=args.input_device,
    )
    time.sleep(0.02)
    sd.play(tx, samplerate=args.output_sr, device=args.output_device, blocking=True)
    sd.wait()

    # Resample the shorter side to a common rate for correlation (linear).
    common_sr = float(max(args.input_sr, args.output_sr))
    t_tx = np.arange(out_n) / float(args.output_sr)
    t_rx = np.arange(in_n) / float(args.input_sr)
    t_c = np.arange(0.0, args.duration_sec, 1.0 / common_sr)
    tx_m = np.interp(t_c, t_tx, tx[:, 0].astype(np.float64), left=0.0, right=0.0)
    rx_m = np.interp(t_c, t_rx, rec[:, 0].astype(np.float64), left=0.0, right=0.0)
    # Align so the known click time is the reference zero in tx.
    latency_ms, peak = _cross_latency_ms(tx_m, rx_m, common_sr)
    # Subtract the intentional click delay so the report is path delay only.
    path_ms = latency_ms - 1000.0 * float(args.click_delay_sec)
    print(f"[result] path_latency_ms={path_ms:.2f}  raw_peak_lag_ms={latency_ms:.2f}  corr={peak:.3f}")
    if peak < 0.05:
        print("[warn] correlation peak is weak; check cabling/gain/device indices", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
