"""Generate small DSP golden arrays for the C++ parity test.

Does not commit the arrays. CI caches the output directory. ONNX hop goldens are
written only when --model is passed and onnxruntime imports.
"""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import numpy as np


def write_tone_wav(path: Path, seconds: float = 2.0, sr: int = 24000, channels: int = 8) -> None:
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    mono = (0.1 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
    interleaved = np.tile(mono[:, None], (1, channels))
    pcm = np.clip(interleaved, -1, 1)
    pcm_i16 = (pcm * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_i16.tobytes())


def dc_filter(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.zeros_like(x)
    prev_x = np.zeros(x.shape[0], dtype=np.float32)
    prev_y = np.zeros(x.shape[0], dtype=np.float32)
    for t in range(x.shape[1]):
        y[:, t] = x[:, t] - prev_x + alpha * prev_y
        prev_x = x[:, t]
        prev_y = y[:, t]
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    wav = out / "tone_2s.wav"
    write_tone_wav(wav)
    with wave.open(str(wav), "rb") as wf:
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).reshape(-1, wf.getnchannels())
    x = (raw.astype(np.float32) / 32768.0).T
    y = dc_filter(x[:1], 0.5)
    np.save(out / "dc_ch0.npy", y[0, :32])
    if args.model is not None:
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
        (out / "onnx_inputs.txt").write_text("\n".join(i.name for i in session.get_inputs()), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
