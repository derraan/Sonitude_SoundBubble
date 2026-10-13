from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class StreamPair:
    input_stream: Any
    output_stream: Any


def _device_default_sr(sd_module: Any, device: int | None) -> float | None:
    if device is None:
        return None
    try:
        info = sd_module.query_devices(device)
    except Exception:
        return None
    default_sr = info.get("default_samplerate")
    return float(default_sr) if default_sr is not None else None


def preflight_sounddevice(
    sd_module: Any,
    input_device: int | None,
    input_channels: int,
    input_sr: int,
    output_device: int | None,
    output_channels: int,
    output_sr: int,
) -> None:
    """Validate PortAudio can open the requested rates (UAC2 host selects Pico clock)."""
    in_default = _device_default_sr(sd_module, input_device)
    if in_default is not None:
        print(
            f"[audio] input device default_samplerate={in_default:g} Hz; "
            f"requesting input_sr={int(input_sr)} Hz (Pico offers 16/24/44.1/48 kHz)"
        )
        if abs(in_default - float(input_sr)) > 1.0:
            print(
                f"[audio] note: device default differs from --input-sr; "
                f"opening at {int(input_sr)} Hz so UAC2 should renegotiate the Pico clock"
            )
    try:
        sd_module.check_input_settings(
            device=input_device,
            channels=int(input_channels),
            samplerate=int(input_sr),
            dtype="float32",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Input device cannot open at {int(input_sr)} Hz ({exc}). "
            "Pico firmware must offer this rate; prefer --input-sr 48000 (exact 2× model_sr)."
        ) from exc
    sd_module.check_output_settings(
        device=output_device,
        channels=int(output_channels),
        samplerate=int(output_sr),
        dtype="float32",
    )


def assert_stream_samplerate(stream: Any, *, expected_sr: int, label: str) -> None:
    """Fail fast if PortAudio did not bind the negotiated rate (B7)."""
    actual = float(getattr(stream, "samplerate", expected_sr))
    if abs(actual - float(expected_sr)) > 1.0:
        raise RuntimeError(
            f"{label} stream samplerate is {actual:g} Hz but --*-sr requested {expected_sr}. "
            "Sample-rate mismatch scales ITD/ILD (fault B7). Fix device negotiation or --input-sr."
        )
    print(f"[audio] {label} stream samplerate={actual:g} Hz (ok)")


def create_streams(
    sd_module: Any,
    *,
    input_sr: int,
    input_block: int,
    input_channels: int,
    input_latency: str,
    input_callback: Callable,
    input_device: int | None,
    output_sr: int,
    output_block: int,
    output_channels: int,
    output_latency: str,
    output_callback: Callable,
    output_device: int | None,
) -> StreamPair:
    input_stream = sd_module.InputStream(
        samplerate=int(input_sr),
        blocksize=int(input_block),
        channels=int(input_channels),
        dtype="float32",
        latency=input_latency,
        callback=input_callback,
        device=input_device,
    )
    output_stream = sd_module.OutputStream(
        samplerate=int(output_sr),
        blocksize=int(output_block),
        channels=int(output_channels),
        dtype="float32",
        latency=output_latency,
        callback=output_callback,
        device=output_device,
    )
    return StreamPair(input_stream=input_stream, output_stream=output_stream)

