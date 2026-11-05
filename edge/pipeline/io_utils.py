from __future__ import annotations

from typing import Any

import numpy as np


def list_audio_devices(sd_module: Any) -> None:
    if sd_module is None:
        raise RuntimeError("sounddevice is not installed. Please install it: pip install sounddevice")
    devices = sd_module.query_devices()
    hostapis = sd_module.query_hostapis()
    print("Audio devices:\n")
    for index, device in enumerate(devices):
        hostapi_name = hostapis[device["hostapi"]]["name"]
        print(
            f"[{index}] {device['name']} | "
            f"hostapi={hostapi_name} | "
            f"in={device['max_input_channels']} "
            f"out={device['max_output_channels']} "
            f"default_sr={device['default_samplerate']}"
        )


def parse_channel_map(channel_map_arg: str | None, input_channels: int, model_num_ch: int) -> list[int]:
    if channel_map_arg is None:
        return list(range(min(input_channels, model_num_ch)))

    channel_map = []
    for token in channel_map_arg.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --channel-map value {channel_map_arg!r}. Use comma-separated integers, e.g. 0,1,2,3,4,5."
            ) from exc
        if idx < 0 or idx >= input_channels:
            raise ValueError(f"Channel index {idx} in --channel-map is out of range for input_channels={input_channels}.")
        channel_map.append(idx)

    if not channel_map:
        raise ValueError("--channel-map did not contain any valid channel indices.")
    if len(channel_map) > model_num_ch:
        raise ValueError(
            f"--channel-map selects {len(channel_map)} channels, but the model expects at most {model_num_ch}."
        )
    return channel_map


def fit_channels(x: np.ndarray, target_channels: int) -> np.ndarray:
    # x: [C, T]
    c, _ = x.shape
    if c == target_channels:
        return x
    if c > target_channels:
        return x[:target_channels]
    repeat = target_channels - c
    reps = int(np.ceil(repeat / c))
    cycled = np.tile(x, (reps + 1, 1))
    return np.concatenate([x, cycled[:repeat]], axis=0)


def to_stereo_or_multich(y: np.ndarray, out_channels: int) -> np.ndarray:
    # y: [T], returns [T, out_channels]
    y = y.astype(np.float32)
    if out_channels == 1:
        return y[:, None]
    return np.tile(y[:, None], (1, out_channels))
