"""Parallel HATS / simulated-head / ear-rig capture.

Records from a separate audio-interface input while ONNX RT + stimulus run.
Clock notes (also in tools/measure/README.md):
- Pico Class B USB and the external interface are **independent clocks**.
- Prefer matching sample rates when possible; fall back to the device default
  (many WDM-KS interfaces are fixed at 44.1 kHz and reject 48 kHz with
  PortAudio -9994 "Sample format not supported").
- Align start by arming the recorder, then starting the pipeline, then playing
  the stimulus (measure core order).

Stereo contract (hard):
- HATS is always opened as a **stereo** PortAudio input (channels=2).
- Internally every buffer is float32 ``[T, 2]`` (L=ch0, R=ch1).
- On Windows Roland QUAD-CAPTURE, WDM-KS / WASAPI often report 2ch but deliver
  a dead left channel; DirectSound / MME deliver real stereo. We therefore
  remap the selected endpoint to the best stereo-capable host API sibling.
"""

from __future__ import annotations

import queue
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
    from scipy.io.wavfile import write as wavwrite
except ImportError:  # pragma: no cover
    wavwrite = None

# Prefer 48 kHz when the interface allows it; otherwise probe defaults / commons.
_HATS_SR_CANDIDATES = (48000, 44100, 96000, 32000, 16000)
_HATS_DTYPE_CANDIDATES = ("float32", "int32", "int16")
_HATS_STEREO_CHANNELS = 2

# Host APIs that actually deliver both QUAD 1-2 legs as independent channels.
# WDM-KS / WASAPI often open as "stereo" with ch0 permanently zero on Roland.
_HATS_HOSTAPI_PREFERENCE = (
    "Windows DirectSound",
    "MME",
    "Windows WASAPI",
    "Windows WDM-KS",
)


def _device_default_sr(device: int | None) -> int | None:
    if sd is None or device is None:
        return None
    try:
        info = sd.query_devices(device)
        sr = float(info.get("default_samplerate") or 0.0)
        return int(round(sr)) if sr > 0 else None
    except Exception:
        return None


def _hostapi_name(device_info: dict[str, Any]) -> str:
    if sd is None:
        return ""
    try:
        return str(sd.query_hostapis(int(device_info["hostapi"]))["name"])
    except Exception:
        return ""


def _endpoint_key(name: str) -> str:
    """Collapse PortAudio naming variants onto a stable endpoint identity."""
    n = (name or "").lower().replace(" ", "")
    if "quad" in n or "quad-capture" in n or "quad_capture" in n:
        if "1-2" in n or "1/2" in n:
            return "quad-1-2"
        if "3-4" in n or "3/4" in n:
            return "quad-3-4"
        if "main" in n:
            return "quad-main"
    return n


def _hostapi_rank(hostapi: str) -> int:
    h = (hostapi or "").strip()
    try:
        return _HATS_HOSTAPI_PREFERENCE.index(h)
    except ValueError:
        return len(_HATS_HOSTAPI_PREFERENCE)


def resolve_hats_stereo_device(device: int | None) -> int:
    """Pick a stereo-capable sibling of ``device`` on the best host API.

    Always requires ``max_input_channels >= 2``. Remaps WDM-KS/WASAPI QUAD
    endpoints to DirectSound/MME when a same-endpoint sibling exists.
    """
    if sd is None:
        raise RuntimeError("sounddevice is required for HATS capture")
    if device is None:
        raise RuntimeError("HATS device index is required")
    devices = list(sd.query_devices())
    try:
        selected = devices[int(device)]
    except Exception as exc:
        raise RuntimeError(f"HATS device {device} not found") from exc
    if int(selected.get("max_input_channels") or 0) < _HATS_STEREO_CHANNELS:
        raise RuntimeError(
            f"HATS device {device} ({selected.get('name')!r}) is not stereo "
            f"(max_input_channels={selected.get('max_input_channels')}); "
            f"open a 2-channel input endpoint"
        )
    key = _endpoint_key(str(selected.get("name") or ""))
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for idx, info in enumerate(devices):
        if int(info.get("max_input_channels") or 0) < _HATS_STEREO_CHANNELS:
            continue
        if _endpoint_key(str(info.get("name") or "")) != key:
            continue
        rank = _hostapi_rank(_hostapi_name(info))
        candidates.append((rank, idx, info))
    if not candidates:
        return int(device)
    candidates.sort(key=lambda t: (t[0], t[1]))
    best_rank, best_idx, _ = candidates[0]
    # Keep the operator selection when it is already on a preferred API or
    # when no better sibling exists.
    selected_rank = _hostapi_rank(_hostapi_name(selected))
    if int(device) == best_idx or selected_rank <= best_rank:
        return int(device)
    return int(best_idx)


def resolve_hats_input_settings(
    device: int | None,
    channels: int,
    preferred_sr: int,
) -> tuple[int, str]:
    """Pick (samplerate, dtype) PortAudio will actually open on ``device``."""
    if sd is None:
        raise RuntimeError("sounddevice is required for HATS capture")
    channels = int(channels)
    if channels != _HATS_STEREO_CHANNELS:
        raise RuntimeError(
            f"HATS must open as stereo (channels={_HATS_STEREO_CHANNELS}), "
            f"got channels={channels}"
        )
    rates: list[int] = []
    for sr in (preferred_sr, _device_default_sr(device), *_HATS_SR_CANDIDATES):
        if sr is None:
            continue
        sr_i = int(sr)
        if sr_i > 0 and sr_i not in rates:
            rates.append(sr_i)
    errors: list[str] = []
    for sr in rates:
        for dtype in _HATS_DTYPE_CANDIDATES:
            try:
                sd.check_input_settings(
                    device=device,
                    channels=_HATS_STEREO_CHANNELS,
                    samplerate=sr,
                    dtype=dtype,
                )
                return sr, dtype
            except Exception as exc:
                errors.append(f"sr={sr} dtype={dtype}: {exc}")
    detail = "; ".join(errors[:6]) if errors else "no candidates tried"
    raise RuntimeError(
        f"HATS device {device} stereo: no supported sample format ({detail})"
    )


def _as_stereo_float32(indata: np.ndarray) -> np.ndarray:
    """Normalize a PortAudio callback/block to float32 ``[T, 2]``."""
    audio = np.asarray(indata)
    if audio.ndim == 1:
        raise RuntimeError(
            f"HATS stream delivered mono frames (shape={audio.shape}); "
            "device must be opened and processed as stereo"
        )
    if audio.ndim != 2:
        raise RuntimeError(f"HATS stream unexpected ndim={audio.ndim} shape={audio.shape}")
    if audio.shape[1] < _HATS_STEREO_CHANNELS:
        raise RuntimeError(
            f"HATS stream delivered {audio.shape[1]} channel(s); need stereo"
        )
    # Keep only L/R even if a driver reports >2.
    return np.asarray(audio[:, :_HATS_STEREO_CHANNELS], dtype=np.float32).copy()


@dataclass
class HatsCaptureConfig:
    device: int | None
    # Stereo is mandatory; the field remains for trial.json / callers.
    channels: int = _HATS_STEREO_CHANNELS
    sample_rate: int = 48000
    # Optional channel reorder after capture, still length-2 stereo.
    # e.g. (1, 0) swaps L/R. Mono subsets are rejected.
    channel_indices: tuple[int, ...] | None = None


class HatsRecorder:
    """Background stereo InputStream that accumulates float32 ``[T, 2]``."""

    def __init__(self, cfg: HatsCaptureConfig) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is required for HATS capture")
        if wavwrite is None:
            raise RuntimeError("scipy is required to write HATS WAVs")
        if int(cfg.channels) != _HATS_STEREO_CHANNELS:
            raise RuntimeError(
                f"HatsCaptureConfig.channels must be {_HATS_STEREO_CHANNELS} "
                f"(stereo); got {cfg.channels}"
            )
        if cfg.channel_indices is not None and len(cfg.channel_indices) != _HATS_STEREO_CHANNELS:
            raise RuntimeError(
                "hats channel_indices must select exactly 2 channels "
                f"(got {cfg.channel_indices!r})"
            )
        self.cfg = cfg
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._chunks: list[np.ndarray] = []
        self._stream: Any = None
        self._lock = threading.Lock()
        self._started = False
        self._dtype: str = "float32"
        self._opened_device: int | None = None
        self._requested_device: int | None = cfg.device
        self._hostapi: str = ""

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        _ = frames, time_info, status
        try:
            self._q.put(_as_stereo_float32(indata))
        except Exception:
            # Never throw out of the PortAudio callback; drop the bad block.
            pass

    @property
    def is_started(self) -> bool:
        return bool(self._started)

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._q = queue.Queue()
            self._chunks = []
            requested = self.cfg.device
            opened = resolve_hats_stereo_device(requested)
            self.cfg.channels = _HATS_STEREO_CHANNELS
            sr, dtype = resolve_hats_input_settings(
                opened,
                _HATS_STEREO_CHANNELS,
                int(self.cfg.sample_rate),
            )
            self.cfg.sample_rate = int(sr)
            self.cfg.device = int(opened)
            self._dtype = dtype
            self._opened_device = int(opened)
            self._requested_device = None if requested is None else int(requested)
            try:
                self._hostapi = _hostapi_name(sd.query_devices(int(opened)))
            except Exception:
                self._hostapi = ""
            self._stream = sd.InputStream(
                samplerate=int(sr),
                channels=_HATS_STEREO_CHANNELS,
                device=int(opened),
                dtype=dtype,
                callback=self._callback,
            )
            # Fail closed if PortAudio silently opened mono.
            opened_ch = int(getattr(self._stream, "channels", _HATS_STEREO_CHANNELS) or 0)
            if opened_ch != _HATS_STEREO_CHANNELS:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
                raise RuntimeError(
                    f"HATS InputStream opened with channels={opened_ch}; "
                    f"stereo ({_HATS_STEREO_CHANNELS}) required"
                )
            self._stream.start()
            self._started = True

    def stop(self) -> np.ndarray:
        """Stop stream and return interleaved float32 ``[T, 2]``."""
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            was_started = self._started
            self._started = False
            while True:
                try:
                    self._chunks.append(self._q.get_nowait())
                except queue.Empty:
                    break
            if not self._chunks:
                audio = np.zeros((0, _HATS_STEREO_CHANNELS), dtype=np.float32)
            else:
                audio = _as_stereo_float32(np.concatenate(self._chunks, axis=0))
                idxs = self.cfg.channel_indices
                if idxs is not None:
                    audio = audio[:, list(idxs)]
                    audio = _as_stereo_float32(audio)
            self._chunks = []
            if not was_started and audio.size == 0:
                return audio
            return audio

    def write_wavs(self, trial_dir: str | Path, audio: np.ndarray | None = None) -> list[Path]:
        """Write stereo ``hats_stereo.wav`` plus ``hats_l.wav`` / ``hats_r.wav``."""
        trial_dir = Path(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)
        if audio is None:
            audio = self.stop()
        written: list[Path] = []
        if audio.size == 0:
            return written
        audio = _as_stereo_float32(audio)
        sr = int(self.cfg.sample_rate)
        stereo_path = trial_dir / "hats_stereo.wav"
        wavwrite(str(stereo_path), sr, audio.astype(np.float32, copy=False))
        written.append(stereo_path)
        p_l = trial_dir / "hats_l.wav"
        wavwrite(str(p_l), sr, audio[:, 0].astype(np.float32, copy=False))
        written.append(p_l)
        p_r = trial_dir / "hats_r.wav"
        wavwrite(str(p_r), sr, audio[:, 1].astype(np.float32, copy=False))
        written.append(p_r)
        return written

    def as_dict(self) -> dict[str, Any]:
        return {
            "hats_in_device": self.cfg.device,
            "hats_in_device_requested": self._requested_device,
            "hats_channels": _HATS_STEREO_CHANNELS,
            "hats_sr": self.cfg.sample_rate,
            "hats_dtype": self._dtype,
            "hats_hostapi": self._hostapi,
            "hats_channel_indices": (
                list(self.cfg.channel_indices) if self.cfg.channel_indices else None
            ),
            "hats_clock_note": (
                "Independent clock vs Pico USB; short trials; match SR when possible; "
                "not sample-locked. HATS always opens as stereo; Windows remaps to a "
                "DirectSound/MME sibling when WDM-KS/WASAPI would deliver a dead L."
            ),
        }
