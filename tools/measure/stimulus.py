"""Thin stimulus player for the shared REW ESS + timing-ref WAV.

Routes to Pico DAC (windows_pico) or speaker/amp device (pi_hats).
Does not regenerate sweeps — plays the operator-owned file as-is.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None

try:
    from scipy.io.wavfile import read as wavread
except ImportError:  # pragma: no cover
    wavread = None


@dataclass
class StimulusInfo:
    path: Path
    sample_rate: int
    channels: int
    frames: int
    duration_s: float


def load_stimulus_wav(path: str | Path) -> tuple[StimulusInfo, np.ndarray]:
    """Load stimulus as float32 ``[T, C]``."""
    if wavread is None:
        raise RuntimeError("scipy is required to load stimulus WAVs")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"stimulus WAV not found: {p}")
    sr, audio = wavread(str(p))
    x = np.asarray(audio)
    if x.ndim == 1:
        x = x[:, None]
    if np.issubdtype(x.dtype, np.integer):
        denom = float(max(abs(int(np.iinfo(x.dtype).min)), int(np.iinfo(x.dtype).max)))
        x = x.astype(np.float32) / denom
    else:
        x = x.astype(np.float32, copy=False)
    info = StimulusInfo(
        path=p.resolve(),
        sample_rate=int(sr),
        channels=int(x.shape[1]),
        frames=int(x.shape[0]),
        duration_s=float(x.shape[0]) / float(sr) if sr else 0.0,
    )
    return info, x


def normalize_play_channels(value: str | None) -> str:
    """Return ``both`` | ``left`` | ``right`` (default ``both``)."""
    v = (value or "both").strip().lower()
    if v in ("both", "left", "right", "l", "r", "lr", "stereo"):
        if v in ("l",):
            return "left"
        if v in ("r",):
            return "right"
        if v in ("lr", "stereo"):
            return "both"
        return v
    raise ValueError(f"play_channels must be both|left|right, got {value!r}")


class StimulusPlayer:
    """Blocking or background playback of a stimulus WAV to one output device."""

    def __init__(
        self,
        path: str | Path,
        *,
        device: int | None = None,
        output_channels: int | None = None,
        repeat: int = 1,
        play_channels: str = "both",
    ) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is required for stimulus playback")
        self.info, self._audio = load_stimulus_wav(path)
        self.device = device
        self.output_channels = int(output_channels or self.info.channels)
        self.repeat = max(1, int(repeat))
        self.play_channels = normalize_play_channels(play_channels)
        # Effective play length after repeats (join timeout / HATS arming).
        self.info = StimulusInfo(
            path=self.info.path,
            sample_rate=self.info.sample_rate,
            channels=self.info.channels,
            frames=int(self.info.frames) * self.repeat,
            duration_s=float(self.info.duration_s) * float(self.repeat),
        )
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def _fit_channels(self) -> np.ndarray:
        x = self._audio
        c = self.output_channels
        if x.shape[1] == c:
            fitted = x
        elif x.shape[1] == 1 and c > 1:
            fitted = np.repeat(x, c, axis=1)
        elif x.shape[1] > c:
            fitted = x[:, :c]
        else:
            pad = np.zeros((x.shape[0], c - x.shape[1]), dtype=np.float32)
            fitted = np.concatenate([x, pad], axis=1)
        fitted = np.asarray(fitted, dtype=np.float32).copy()
        # Mute unused DAC legs so stereo ESS (e.g. *_refL) can drive L-only timing ref.
        if c >= 2 and self.play_channels == "left":
            fitted[:, 1:] = 0.0
        elif c >= 2 and self.play_channels == "right":
            fitted[:, 0] = 0.0
            if fitted.shape[1] > 2:
                fitted[:, 2:] = 0.0
        if self.repeat <= 1:
            return fitted
        return np.concatenate([fitted] * self.repeat, axis=0)

    def play_blocking(self) -> StimulusInfo:
        data = self._fit_channels()
        sd.play(data, samplerate=self.info.sample_rate, device=self.device, blocking=True)
        return self.info

    def start_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("stimulus already playing")
        self._error = None

        def _run() -> None:
            try:
                self.play_blocking()
            except BaseException as exc:  # noqa: BLE001 — surface to join()
                self._error = exc

        self._thread = threading.Thread(target=_run, name="stimulus-player", daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is None:
            return
        self._thread.join(timeout=timeout)
        if self._error is not None:
            raise RuntimeError(f"stimulus playback failed: {self._error}") from self._error

    def stop(self) -> None:
        try:
            sd.stop()
        except Exception:
            pass

    def as_dict(self) -> dict[str, Any]:
        return {
            "stimulus_path": str(self.info.path),
            "stimulus_sr": self.info.sample_rate,
            "stimulus_channels": self.info.channels,
            "stimulus_frames": self.info.frames,
            "stimulus_duration_s": self.info.duration_s,
            "stimulus_out_device": self.device,
            "stimulus_repeat": self.repeat,
            "stimulus_play_channels": self.play_channels,
        }


# ---------------------------------------------------------------------------
# In-context speech: random utterance loop (2026-11-04)
# ---------------------------------------------------------------------------
SPEECH_EXTS = (".flac", ".wav")
SPEECH_LOOP_SR = 44100
SPEECH_LOOP_SECONDS = 15.0     # ≈ ESS take length (13.4 s sweep)
SPEECH_LOOP_GAP_S = 0.35
SPEECH_LOOP_LEVEL_DBFS = -26.0  # per-utterance active RMS (≈ ESS −20 dBFS peak sweep loudness)
SPEECH_LOOP_MAX_UTT_S = 6.0     # long utterances are cut (at a quiet point 4–6 s) for more talkers per take


def list_speech_files(src_dir: str | Path) -> list[Path]:
    """All speech files (flac/wav) under ``src_dir`` (recursive, sorted)."""
    root = Path(src_dir)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in SPEECH_EXTS and p.is_file())


def build_speech_loop(
    src_dir: str | Path,
    out_path: str | Path,
    *,
    seed: int | None = None,
    target_s: float = SPEECH_LOOP_SECONDS,
    sample_rate: int = SPEECH_LOOP_SR,
    gap_s: float = SPEECH_LOOP_GAP_S,
    level_dbfs: float = SPEECH_LOOP_LEVEL_DBFS,
    channels: int = 2,
    max_utt_s: float = SPEECH_LOOP_MAX_UTT_S,
) -> dict[str, Any]:
    """Write a random-utterance speech loop WAV (float32) and return its metadata.

    Random utterances from ``src_dir`` are mono-mixed, resampled to ``sample_rate``,
    levelled to ``level_dbfs`` active RMS (peaks capped at −1 dBFS), and joined with
    ``gap_s`` of silence until ``target_s`` is reached. Same signal on every channel;
    StimulusPlayer's play_channels still routes left/right/both.
    """
    import random
    import time
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly

    files = list_speech_files(src_dir)
    if not files:
        raise FileNotFoundError(f"no speech files (*.flac/*.wav) under {src_dir}")
    seed = int(seed) if seed is not None else int(time.time_ns() % (2**31))
    rng = random.Random(seed)
    gap = np.zeros(int(round(gap_s * sample_rate)), np.float32)
    pieces: list[np.ndarray] = [gap]
    used: list[dict[str, Any]] = []
    total = len(gap)
    guard = 0
    while total < target_s * sample_rate and guard < 200:
        guard += 1
        f = rng.choice(files)
        x, sr = sf.read(str(f), always_2d=True, dtype="float32")
        x = x.mean(axis=1)
        if sr != sample_rate:
            g = gcd(int(sr), int(sample_rate))
            x = resample_poly(x, sample_rate // g, int(sr) // g).astype(np.float32)
        w = max(1, sample_rate // 50)  # 20 ms frames for active-RMS levelling
        e = np.array([np.mean(x[i:i + w] ** 2) for i in range(0, max(1, len(x) - w), w)])
        act = e[e > np.percentile(e, 40)] if len(e) > 4 else e
        rms = float(np.sqrt(np.mean(act))) if len(act) else 0.0
        if rms <= 1e-8:
            continue
        x = x * (10 ** (level_dbfs / 20) / rms)
        peak = float(np.max(np.abs(x)))
        if peak > 10 ** (-1 / 20):
            x = x * (10 ** (-1 / 20) / peak)
        if max_utt_s and len(x) > max_utt_s * sample_rate:
            lo, hi = int(0.67 * max_utt_s * sample_rate), int(max_utt_s * sample_rate)
            seg = x[lo:hi] ** 2
            k = int(np.argmin(np.convolve(seg, np.ones(w) / w, mode="same"))) if len(seg) > w else 0
            x = x[: lo + k].copy()
            fade = min(len(x), sample_rate // 100)  # 10 ms fade-out at the cut
            x[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        pieces += [x.astype(np.float32), gap]
        total += len(x) + len(gap)
        try:
            rel = str(f.relative_to(Path(src_dir)))
        except ValueError:
            rel = str(f)
        used.append({"file": rel, "source_sr": int(sr), "duration_s": round(len(x) / sample_rate, 3)})
    y = np.concatenate(pieces)
    out = np.repeat(y[:, None], max(1, int(channels)), axis=1)
    sf.write(str(out_path), out, sample_rate, subtype="FLOAT")
    return {
        "type": "random_utterance_loop",
        "source_dir": str(Path(src_dir)),
        "pool_size": len(files),
        "seed": seed,
        "sample_rate": sample_rate,
        "level_dbfs_active_rms": level_dbfs,
        "gap_s": gap_s,
        "duration_s": round(len(y) / sample_rate, 3),
        "utterances": used,
    }
