"""WAV load + band / FR helpers for offline measure analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.io.wavfile import read as wavread
except ImportError:  # pragma: no cover
    wavread = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None


def load_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Load WAV as float32 ``[T, C]`` in roughly ``[-1, 1]``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if sf is not None:
        audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
        return int(sr), np.asarray(audio, dtype=np.float32)

    if wavread is None:
        raise RuntimeError("need soundfile or scipy to read WAVs")
    sr, audio = wavread(str(path))
    x = np.asarray(audio)
    if x.ndim == 1:
        x = x[:, None]
    if np.issubdtype(x.dtype, np.integer):
        denom = float(max(abs(int(np.iinfo(x.dtype).min)), int(np.iinfo(x.dtype).max)))
        x = x.astype(np.float32) / denom
    else:
        x = x.astype(np.float32, copy=False)
    return int(sr), x


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.astype(np.float64, copy=False)
    return x.mean(axis=1).astype(np.float64, copy=False)


def rms_dbfs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    rms = float(np.sqrt(np.mean(x * x) + 1e-20))
    return 20.0 * float(np.log10(max(rms, 1e-20)))


def fullband_level_db(x: np.ndarray, sr: int | None = None) -> float:
    """Absolute full-spectrum level (RMS dBFS) for polar scalars.

    ``sr`` is accepted for API symmetry with ``band_level_db`` but unused —
    RMS covers the whole take / Nyquist band without a speech-band mask.
    """
    _ = sr
    return rms_dbfs(x)


def band_level_db(
    x: np.ndarray,
    sr: int,
    f_lo: float = 300.0,
    f_hi: float = 3400.0,
) -> float:
    """Hann-windowed FFT band energy → dB (relative, comparable dry vs wet)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return -120.0
    # Full-spectrum shortcut when the requested band covers DC→Nyquist.
    nyq = 0.5 * float(sr)
    if f_lo <= 0.0 and f_hi >= nyq * 0.99:
        return fullband_level_db(x, sr)
    n = int(2 ** np.ceil(np.log2(max(256, x.size))))
    w = np.hanning(x.size)
    spec = np.fft.rfft(x * w, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        power = float(np.mean(x * x) + 1e-20)
    else:
        power = float(np.mean(np.abs(spec[mask]) ** 2) + 1e-20)
    return 10.0 * float(np.log10(power))


def suppression_db(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    f_lo: float = 300.0,
    f_hi: float = 3400.0,
) -> dict[str, float]:
    n = min(dry.size, wet.size)
    dry_l = band_level_db(dry[:n], sr, f_lo, f_hi)
    wet_l = band_level_db(wet[:n], sr, f_lo, f_hi)
    return {
        "dry_band_db": dry_l,
        "wet_band_db": wet_l,
        "suppression_db": dry_l - wet_l,
        "dry_rms_dbfs": rms_dbfs(dry[:n]),
        "wet_rms_dbfs": rms_dbfs(wet[:n]),
        "f_lo_hz": float(f_lo),
        "f_hi_hz": float(f_hi),
    }


def magnitude_fr_db(
    x: np.ndarray,
    sr: int,
    *,
    n_fft: int = 4096,
    f_lo: float = 20.0,
    f_hi: float | None = 20000.0,
) -> dict[str, Any]:
    """Smoothed magnitude spectrum (dB) for FR plots / series reload.

    Default band is 20 Hz–20 kHz, clipped to Nyquist when ``sr`` is lower
    (e.g. ~12 kHz max at 24 kHz capture).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {"freq_hz": [], "mag_db": [], "sr": sr}
    nyq = 0.5 * float(sr)
    requested_hi = float(f_hi if f_hi is not None else 20000.0)
    f_hi = min(requested_hi, nyq)
    n_fft = int(max(n_fft, 256))
    hop = max(n_fft // 4, 1)
    # Average |STFT| power across frames (Welch-ish).
    if x.size < n_fft:
        pad = np.zeros(n_fft, dtype=np.float64)
        pad[: x.size] = x
        frames = [pad]
    else:
        frames = [x[i : i + n_fft] for i in range(0, x.size - n_fft + 1, hop)]
    window = np.hanning(n_fft)
    acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for fr in frames:
        if len(fr) < n_fft:
            continue
        acc += np.abs(np.fft.rfft(fr * window)) ** 2
    acc /= max(len(frames), 1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    freqs = freqs[mask]
    mag_db = 10.0 * np.log10(acc[mask] + 1e-20)
    # Normalize to band peak for readable overlays (shape, not absolute cal).
    mag_db = mag_db - float(np.max(mag_db))
    return {
        "freq_hz": [float(v) for v in freqs],
        "mag_db": [float(v) for v in mag_db],
        "sr": int(sr),
        "n_fft": n_fft,
        "f_lo_hz": float(f_lo),
        "f_hi_hz": float(f_hi),
        "ref": "peak_normalized",
    }


def write_wav_float32(path: str | Path, sr: int, audio: np.ndarray) -> Path:
    """Write float32 WAV (clipped to [-1, 1]). Prefer soundfile, else scipy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    x = np.clip(x, -1.0, 1.0)
    if sf is not None:
        sf.write(str(path), x, int(sr), subtype="FLOAT")
        return path
    from scipy.io.wavfile import write as wavwrite  # lazy

    # scipy float write is float32 PCM in [-1,1]
    wavwrite(str(path), int(sr), x)
    return path
