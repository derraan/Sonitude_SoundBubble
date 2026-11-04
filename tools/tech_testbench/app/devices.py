"""PortAudio device listing (same idea as the Sonitude testbench device_manager)."""

from __future__ import annotations

from dataclasses import dataclass

import sounddevice as sd


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate_hz: float
    hostapi: str


def list_devices() -> list[DeviceInfo]:
    apis = sd.query_hostapis()
    out: list[DeviceInfo] = []
    for index, device in enumerate(sd.query_devices()):
        api = apis[device["hostapi"]]["name"]
        out.append(
            DeviceInfo(
                index=index,
                name=str(device["name"]),
                max_input_channels=int(device.get("max_input_channels", 0)),
                max_output_channels=int(device.get("max_output_channels", 0)),
                default_sample_rate_hz=float(device.get("default_samplerate", 0.0)),
                hostapi=str(api),
            )
        )
    return out


def list_input_devices(min_channels: int = 1) -> list[DeviceInfo]:
    return [d for d in list_devices() if d.max_input_channels >= min_channels]


def list_output_devices(min_channels: int = 1) -> list[DeviceInfo]:
    return [d for d in list_devices() if d.max_output_channels >= min_channels]


def format_device(d: DeviceInfo) -> str:
    return (
        f"[{d.index}] {d.name}  "
        f"(in={d.max_input_channels} out={d.max_output_channels}  "
        f"{d.default_sample_rate_hz:.0f} Hz  {d.hostapi})"
    )


def device_by_index(index: int | None) -> DeviceInfo | None:
    if index is None:
        return None
    for d in list_devices():
        if d.index == int(index):
            return d
    return None


def check_rate(
    *,
    device: int,
    samplerate: int,
    channels: int,
    kind: str,
) -> str | None:
    """Return an error string if PortAudio rejects this rate, else None.

    Note: on Windows WDM-KS, an unsupported rate often surfaces later as
    ``Invalid device`` from Pa_OpenStream — not ``Invalid sample rate``.
    """
    try:
        if kind == "input":
            sd.check_input_settings(
                device=device,
                channels=channels,
                samplerate=samplerate,
                dtype="float32",
            )
        else:
            sd.check_output_settings(
                device=device,
                channels=channels,
                samplerate=samplerate,
                dtype="float32",
            )
        return None
    except Exception as exc:  # noqa: BLE001 — PortAudioError / ValueError
        return str(exc)


def suggested_pipeline_sr(device: DeviceInfo | None, fallback: int = 24000) -> int:
    """Native device rate when known; Pico Class B is 24000-only."""
    if device is None:
        return int(fallback)
    native = int(round(float(device.default_sample_rate_hz or 0.0)))
    if native > 0:
        return native
    if "INMP441" in device.name.upper():
        return 24000
    return int(fallback)


def validate_pipeline_rates(
    *,
    pipeline_in: int | None,
    pipeline_out: int | None,
    input_sr: int,
    output_sr: int,
    input_channels: int = 8,
    output_channels: int = 2,
) -> tuple[list[str], int, int]:
    """Validate / snap pipeline SR to device-native rates.

    Returns (errors, input_sr, output_sr). On soft mismatch with a clear native
    rate, snaps silently and returns no errors (caller should apply snapped SR).
    Hard failures (missing device, check fails with no native hint) return errors.
    """
    errs: list[str] = []
    in_dev = device_by_index(pipeline_in)
    out_dev = device_by_index(pipeline_out)
    in_sr = int(input_sr)
    out_sr = int(output_sr)

    if pipeline_in is None:
        errs.append("Pipeline input device is not set.")
    elif in_dev is None:
        errs.append(f"Pipeline input device index {pipeline_in} is not present.")
    else:
        want = suggested_pipeline_sr(in_dev, in_sr)
        bad = check_rate(
            device=int(pipeline_in),
            samplerate=in_sr,
            channels=input_channels,
            kind="input",
        )
        if bad or in_sr != want:
            # Snap to native; re-check.
            in_sr = want
            bad2 = check_rate(
                device=int(pipeline_in),
                samplerate=in_sr,
                channels=input_channels,
                kind="input",
            )
            if bad2:
                errs.append(
                    f"Pipeline input [{pipeline_in}] {in_dev.name} rejects "
                    f"{want} Hz ({input_channels}ch): {bad2}. "
                    "Pico INMP441 WDM-KS is 24000 Hz only; wrong SR often "
                    "shows up later as PortAudio 'Invalid device'."
                )

    if pipeline_out is None:
        errs.append("Pipeline output device is not set.")
    elif out_dev is None:
        errs.append(f"Pipeline output device index {pipeline_out} is not present.")
    else:
        want = suggested_pipeline_sr(out_dev, out_sr)
        bad = check_rate(
            device=int(pipeline_out),
            samplerate=out_sr,
            channels=output_channels,
            kind="output",
        )
        if bad or out_sr != want:
            out_sr = want
            bad2 = check_rate(
                device=int(pipeline_out),
                samplerate=out_sr,
                channels=output_channels,
                kind="output",
            )
            if bad2:
                errs.append(
                    f"Pipeline output [{pipeline_out}] {out_dev.name} rejects "
                    f"{want} Hz ({output_channels}ch): {bad2}. "
                    "Pico DAC WDM-KS is 24000 Hz only; wrong SR often "
                    "shows up later as PortAudio 'Invalid device'."
                )

    if in_sr != out_sr:
        # Two-stream mode allows different rates in theory, but Class B Pico
        # and the RT resampler path expect matched device rates on this bench.
        errs.append(
            f"Pipeline input SR ({in_sr}) and output SR ({out_sr}) differ — "
            "set both to the Pico native rate (usually 24000)."
        )

    return errs, in_sr, out_sr
