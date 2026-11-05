from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

try:
    import soxr  # type: ignore
except Exception:
    soxr = None


def dbfs_to_linear(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def rms_dbfs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
    return float(20.0 * np.log10(max(rms, 1e-10)))


def peak_dbfs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return -120.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return float(20.0 * np.log10(max(peak, 1e-10)))


def render_monitor(levels: np.ndarray, statuses: list[str]) -> str:
    split = max(1, len(levels) // 2)
    line1: list[str] = []
    line2: list[str] = []
    for ch in range(len(levels)):
        token = f"ch{ch}: {levels[ch]:6.1f} dBFS {statuses[ch]:7s}"
        if ch < split:
            line1.append(token)
        else:
            line2.append(token)
    if line2:
        return "    ".join(line1) + "\n" + "    ".join(line2)
    return "    ".join(line1)


def resample_poly_exact(x: np.ndarray, src_sr: int, dst_sr: int, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)
    if src_sr <= 0 or dst_sr <= 0:
        raise ValueError("Sample rates must be positive.")
    if x.shape[axis] == 0 or src_sr == dst_sr:
        return x.astype(np.float32, copy=False)

    g = math.gcd(int(src_sr), int(dst_sr))
    up = int(dst_sr) // g
    down = int(src_sr) // g
    y = resample_poly(x, up=up, down=down, axis=axis, window=("kaiser", 8.6))

    in_len = x.shape[axis]
    out_len = int(round(in_len * dst_sr / src_sr))
    if y.shape[axis] > out_len:
        slicer = [slice(None)] * y.ndim
        slicer[axis] = slice(0, out_len)
        y = y[tuple(slicer)]
    elif y.shape[axis] < out_len:
        pad_width = [(0, 0)] * y.ndim
        pad_width[axis] = (0, out_len - y.shape[axis])
        y = np.pad(y, pad_width, mode="constant")
    return y.astype(np.float32, copy=False)


class HalfbandResampler:
    """Causal, stateful 2x up- or down-sampler (linear-phase FIR, ~0.5 ms delay at 48 kHz).

    Output length is exactly proportional to input length on every block, unlike SoXR's
    streaming mode, which releases ~30 ms bursts.
    """

    def __init__(self, *, up: bool, num_channels: int, num_taps: int = 48) -> None:
        self.up = bool(up)
        self.num_channels = int(num_channels)
        # Cutoff at half the high-rate Nyquist (= Nyquist of the low rate).
        h = firwin(int(num_taps), 0.5, window=("kaiser", 8.0)).astype(np.float64)
        self._h = h * 2.0 if self.up else h
        self._zi = np.zeros((self.num_channels, len(h) - 1), dtype=np.float64)
        self._consumed = 0

    def reset(self) -> None:
        self._zi.fill(0.0)
        self._consumed = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = int(x.shape[1])
        if n == 0:
            return np.zeros((self.num_channels, 0), dtype=np.float32)
        if self.up:
            z = np.zeros((self.num_channels, 2 * n), dtype=np.float64)
            z[:, ::2] = x
            y, self._zi = lfilter(self._h, 1.0, z, axis=-1, zi=self._zi)
            return y.astype(np.float32)
        y, self._zi = lfilter(self._h, 1.0, x, axis=-1, zi=self._zi)
        start = (-self._consumed) % 2
        self._consumed += n
        return y[:, start::2].astype(np.float32)


class FixedRatioASRC:
    """Causal linear ASRC with long-term exact sample accounting.

    Per-block output length varies by at most ~1 sample (not SoXR's ~30 ms bursts).
    `correction_ratio` scales the destination rate (1 - ppm*1e-6 from DriftMonitor).
    """

    def __init__(self, *, src_sr: int, dst_sr: int, num_channels: int) -> None:
        self.src_sr = float(src_sr)
        self.dst_sr_nominal = float(dst_sr)
        self.dst_sr = float(dst_sr)
        self.num_channels = int(num_channels)
        self._correction = 1.0
        self._ratio = self.src_sr / max(1e-6, self.dst_sr)
        self._buf = np.zeros((self.num_channels, 0), dtype=np.float64)
        self._buf_start = 0.0
        self._next_src_pos = 0.0
        self._src_total = 0
        self._dst_total = 0

    def set_correction_ratio(self, ratio: float) -> None:
        self._correction = float(np.clip(ratio, 0.98, 1.02))
        self.dst_sr = self.dst_sr_nominal * self._correction
        self._ratio = self.src_sr / max(1e-6, self.dst_sr)

    def reset(self) -> None:
        self._buf = np.zeros((self.num_channels, 0), dtype=np.float64)
        self._buf_start = 0.0
        self._next_src_pos = 0.0
        self._src_total = 0
        self._dst_total = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != self.num_channels:
            raise ValueError(f"expected [C,T] with C={self.num_channels}, got {x.shape}")
        n = int(x.shape[1])
        if n == 0:
            return np.zeros((self.num_channels, 0), dtype=np.float32)
        if abs(self.src_sr - self.dst_sr) < 1e-9 and abs(self._correction - 1.0) < 1e-12:
            return x.astype(np.float32)

        self._buf = np.concatenate([self._buf, x], axis=1) if self._buf.shape[1] else x.copy()
        self._src_total += n
        target_dst = int(math.floor(self._src_total * self.dst_sr / self.src_sr + 1e-12))
        want = max(0, target_dst - self._dst_total)
        # Largest n_out such that next_src_pos + (n_out-1)*ratio + 1 stays inside the buffer.
        span = float(self._buf.shape[1]) - 1.0 - (self._next_src_pos - self._buf_start)
        max_out = 0 if span < 0.0 else int(math.floor(span / self._ratio + 1e-12)) + 1
        n_out = min(want, max(0, max_out))
        if n_out <= 0:
            return np.zeros((self.num_channels, 0), dtype=np.float32)

        positions = self._next_src_pos + self._ratio * np.arange(n_out, dtype=np.float64)
        local = positions - self._buf_start
        i0 = np.floor(local).astype(np.int64)
        frac = (local - i0).astype(np.float64)
        y = (1.0 - frac) * self._buf[:, i0] + frac * self._buf[:, i0 + 1]
        self._next_src_pos = float(positions[-1] + self._ratio)
        self._dst_total += n_out
        keep_from = max(0, int(math.floor(self._next_src_pos - self._buf_start)) - 1)
        if keep_from > 0:
            self._buf = self._buf[:, keep_from:]
            self._buf_start += float(keep_from)
        return y.astype(np.float32)


class StreamingResampler:
    """
    Stateful streaming SRC wrapper.

    - 44.1 kHz <-> 24 kHz: ASRC to 48 kHz then causal half-band (smooth blocks).
    - Exact 2:1 / 1:2: causal half-band FIR.
    - 44.1 kHz <-> 48 kHz: FixedRatioASRC alone.
    - Else: SoXR stream, or polyphase fallback if SoXR is missing.
    """

    def __init__(self, *, src_sr: int, dst_sr: int, num_channels: int, quality: str = "VHQ") -> None:
        self.src_sr = int(src_sr)
        self.dst_sr = int(dst_sr)
        self.num_channels = int(num_channels)
        self.quality = str(quality)

        if self.src_sr <= 0 or self.dst_sr <= 0:
            raise ValueError("Sample rates must be positive.")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive.")

        self._asrc: FixedRatioASRC | None = None
        self._halfband: HalfbandResampler | None = None
        self._rate_trim: FixedRatioASRC | None = None
        self._stream = None
        self._use_soxr = False

        pair = {self.src_sr, self.dst_sr}
        if pair == {44100, 24000}:
            # Capture/model bridge: ASRC 44.1<->48, then exact half-band 48<->24.
            up = self.dst_sr > self.src_sr
            if up:
                self._halfband = HalfbandResampler(up=True, num_channels=self.num_channels)
                self._asrc = FixedRatioASRC(src_sr=48000, dst_sr=44100, num_channels=self.num_channels)
                self.backend = "halfband-asrc"
            else:
                self._asrc = FixedRatioASRC(src_sr=44100, dst_sr=48000, num_channels=self.num_channels)
                self._halfband = HalfbandResampler(up=False, num_channels=self.num_channels)
                self.backend = "asrc-halfband"
        elif pair == {44100, 48000}:
            self._asrc = FixedRatioASRC(src_sr=self.src_sr, dst_sr=self.dst_sr, num_channels=self.num_channels)
            self.backend = "asrc-linear"
        elif self.dst_sr == 2 * self.src_sr or self.src_sr == 2 * self.dst_sr:
            self._halfband = HalfbandResampler(up=self.dst_sr > self.src_sr, num_channels=self.num_channels)
            # Micro rate trim for clock drift (identity when correction_ratio == 1).
            self._rate_trim = FixedRatioASRC(src_sr=self.dst_sr, dst_sr=self.dst_sr, num_channels=self.num_channels)
            self.backend = "halfband-fir"
        else:
            self._use_soxr = soxr is not None and hasattr(soxr, "ResampleStream")
            self.backend = "soxr-stream" if self._use_soxr else "polyphase-fallback"

        # Polyphase streaming fallback state.
        g = math.gcd(self.src_sr, self.dst_sr)
        self._up = self.dst_sr // g
        self._down = self.src_sr // g
        self._hist_len = int(max(32, 16 * max(self._up, self._down)))
        self._hist = np.zeros((self.num_channels, self._hist_len), dtype=np.float32)

        if self._use_soxr:
            self._stream = soxr.ResampleStream(self.src_sr, self.dst_sr, self.num_channels, dtype="float32", quality=self.quality)

    def set_correction_ratio(self, ratio: float) -> None:
        if self._rate_trim is not None:
            self._rate_trim.set_correction_ratio(ratio)
        elif self._asrc is not None and self._halfband is None:
            self._asrc.set_correction_ratio(ratio)

    def reset(self) -> None:
        if self._asrc is not None:
            self._asrc.reset()
        if self._halfband is not None:
            self._halfband.reset()
        if self._rate_trim is not None:
            self._rate_trim.reset()
        if self._use_soxr:
            self._stream = soxr.ResampleStream(self.src_sr, self.dst_sr, self.num_channels, dtype="float32", quality=self.quality)
        self._hist.fill(0.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] != self.num_channels:
            raise ValueError(f"expected [C,T] with C={self.num_channels}, got {x.shape}")
        if x.shape[1] == 0 or self.src_sr == self.dst_sr:
            return x.astype(np.float32, copy=False)

        if self.backend == "asrc-halfband":
            assert self._asrc is not None and self._halfband is not None
            return self._halfband.process(self._asrc.process(x))
        if self.backend == "halfband-asrc":
            assert self._asrc is not None and self._halfband is not None
            return self._asrc.process(self._halfband.process(x))
        if self._halfband is not None:
            y = self._halfband.process(x)
            if self._rate_trim is not None and abs(self._rate_trim._correction - 1.0) > 1e-9:
                y = self._rate_trim.process(y)
            return y
        if self._asrc is not None:
            return self._asrc.process(x)

        if self._use_soxr and self._stream is not None:
            y_tc = self._stream.resample_chunk(x.T, last=False)
            return np.asarray(y_tc, dtype=np.float32).T

        # Polyphase fallback: prepend history and discard its resampled contribution.
        x_ext = np.concatenate([self._hist, x], axis=1)
        y_ext = resample_poly(x_ext, up=self._up, down=self._down, axis=-1, window=("kaiser", 8.6)).astype(np.float32, copy=False)

        in_len = int(x_ext.shape[-1])
        out_len = int(round(in_len * self.dst_sr / self.src_sr))
        if y_ext.shape[-1] > out_len:
            y_ext = y_ext[..., :out_len]
        elif y_ext.shape[-1] < out_len:
            y_ext = np.pad(y_ext, ((0, 0), (0, out_len - y_ext.shape[-1])), mode="constant")

        hist_out_len = int(round(self._hist_len * self.dst_sr / self.src_sr))
        y = y_ext[:, hist_out_len:]

        if x.shape[-1] >= self._hist_len:
            self._hist[:, :] = x[:, -self._hist_len :]
        else:
            self._hist = np.concatenate([self._hist[:, x.shape[-1] :], x], axis=1)
        return y.astype(np.float32, copy=False)


class DCOffsetFilter:
    def __init__(self, num_channels: int, alpha: float) -> None:
        self.num_channels = int(num_channels)
        self.alpha = float(alpha)
        self.prev_x = np.zeros(self.num_channels, dtype=np.float32)
        self.prev_y = np.zeros(self.num_channels, dtype=np.float32)

        # Difference equation:
        #   y[n] = x[n] - x[n-1] + alpha * y[n-1]
        # Implemented via scipy.signal.lfilter with per-channel state.
        self._b = np.array([1.0, -1.0], dtype=np.float32)
        self._a = np.array([1.0, -self.alpha], dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        # x: [C, T]
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] != self.num_channels:
            raise ValueError(f"expected [C,T] with C={self.num_channels}, got {x.shape}")
        if x.shape[1] == 0:
            return x.astype(np.float32, copy=False)

        # For this 2-tap IIR, lfilter's initial condition maps as:
        #   y[0] = b0*x[0] + zi0, with b0=1
        # We want y[0] = x[0] - prev_x + alpha*prev_y, so zi0 = -prev_x + alpha*prev_y.
        zi0 = (-self.prev_x + self.alpha * self.prev_y).astype(np.float32, copy=False)[:, None]
        y, zf = lfilter(self._b, self._a, x, axis=-1, zi=zi0)
        # Maintain prior semantics for other code that inspects these.
        self.prev_x[:] = x[:, -1]
        self.prev_y[:] = y[:, -1]
        return y.astype(np.float32, copy=False)


class ChannelLevelValidator:
    def __init__(
        self,
        num_channels: int,
        silence_dbfs: float,
        fault_min_dbfs: float,
        fault_max_dbfs: float,
        silence_window_frames: int,
    ) -> None:
        self.num_channels = int(num_channels)
        self.silence_dbfs = float(silence_dbfs)
        self.fault_min_dbfs = float(fault_min_dbfs)
        self.fault_max_dbfs = float(fault_max_dbfs)
        self.silence_window_frames = max(1, int(silence_window_frames))
        self.silence_counts = np.zeros(self.num_channels, dtype=np.int64)
        self.fault_counts = np.zeros(self.num_channels, dtype=np.int64)
        self.power_history: deque[np.ndarray] = deque(maxlen=self.silence_window_frames)
        self.power_sum = np.zeros(self.num_channels, dtype=np.float64)

    @staticmethod
    def _rms_dbfs(channel: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(channel), dtype=np.float64) + 1e-12))
        return float(20.0 * np.log10(max(rms, 1e-10)))

    def validate(self, x: np.ndarray, frame_idx: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        y = x.copy()
        frame_levels = np.zeros(self.num_channels, dtype=np.float32)
        statuses: list[str] = ["ok"] * self.num_channels
        frame_power = np.mean(np.square(y, dtype=np.float64), axis=1).astype(np.float32)
        if len(self.power_history) == self.power_history.maxlen:
            oldest = self.power_history.popleft()
            self.power_sum -= oldest.astype(np.float64)
        self.power_history.append(frame_power)
        self.power_sum += frame_power.astype(np.float64)
        rolling_power = (self.power_sum / max(1, len(self.power_history))).astype(np.float32)
        rolling_levels = np.array(
            [float(20.0 * np.log10(max(float(np.sqrt(p + 1e-12)), 1e-10))) for p in rolling_power],
            dtype=np.float32,
        )
        for ch in range(self.num_channels):
            dbfs = self._rms_dbfs(y[ch])
            frame_levels[ch] = dbfs
            if rolling_levels[ch] < self.silence_dbfs:
                y[ch, :] = 0.0
                statuses[ch] = "silence"
                self.silence_counts[ch] += 1
            elif self.fault_min_dbfs <= dbfs <= self.fault_max_dbfs:
                y[ch, :] = 0.0
                statuses[ch] = "FAULT"
                self.fault_counts[ch] += 1
            else:
                statuses[ch] = "ok"
        return y, frame_levels, statuses


class DriftMonitor:
    def __init__(
        self,
        block_duration_sec: float,
        target_fill: int,
        window_sec: float,
        update_every: int,
        max_ppm: float,
    ) -> None:
        import time

        self._time = time
        self.block_duration_sec = float(block_duration_sec)
        self.target_fill = max(1, int(target_fill))
        self.window_sec = float(window_sec)
        self.update_every = max(1, int(update_every))
        self.max_ppm = float(max_ppm)
        self.samples: deque[tuple[float, float]] = deque()
        self.counter = 0
        self.current_ppm = 0.0
        self.current_drift_ppm = 0.0

    def update(self, fill_level: int) -> tuple[float, float]:
        self.counter += 1
        now = self._time.perf_counter()
        self.samples.append((now, float(fill_level)))
        while len(self.samples) > 2 and (now - self.samples[0][0]) > self.window_sec:
            self.samples.popleft()

        if self.counter % self.update_every != 0:
            return self.current_ppm, (1.0 - self.current_ppm * 1e-6)

        if len(self.samples) >= 2:
            t0, f0 = self.samples[0]
            t1, f1 = self.samples[-1]
            dt = max(1e-6, t1 - t0)
            slope_blocks_per_sec = (f1 - f0) / dt
            nominal_blocks_per_sec = 1.0 / max(1e-6, self.block_duration_sec)
            self.current_drift_ppm = float((slope_blocks_per_sec / nominal_blocks_per_sec) * 1e6)
        else:
            self.current_drift_ppm = 0.0

        error_blocks = float(fill_level - self.target_fill)
        error_ppm = (error_blocks / self.target_fill) * (self.block_duration_sec / self.window_sec) * 1e6
        desired_ppm = self.current_drift_ppm + error_ppm
        desired_ppm = float(np.clip(desired_ppm, -self.max_ppm, self.max_ppm))
        self.current_ppm = 0.9 * self.current_ppm + 0.1 * desired_ppm
        self.current_ppm = float(np.clip(self.current_ppm, -self.max_ppm, self.max_ppm))
        return self.current_ppm, (1.0 - self.current_ppm * 1e-6)


@dataclass
class ProfileStats:
    capture_to_frame_ms: float = 0.0
    dc_ms: float = 0.0
    level_ms: float = 0.0
    frame_to_infer_ms: float = 0.0
    infer_ms: float = 0.0
    post_ms: float = 0.0
    e2e_ms: float = 0.0
    drift_ppm: float = 0.0

    def update(self, other: "ProfileStats", alpha: float = 0.1) -> None:
        self.capture_to_frame_ms = (1 - alpha) * self.capture_to_frame_ms + alpha * other.capture_to_frame_ms
        self.dc_ms = (1 - alpha) * self.dc_ms + alpha * other.dc_ms
        self.level_ms = (1 - alpha) * self.level_ms + alpha * other.level_ms
        self.frame_to_infer_ms = (1 - alpha) * self.frame_to_infer_ms + alpha * other.frame_to_infer_ms
        self.infer_ms = (1 - alpha) * self.infer_ms + alpha * other.infer_ms
        self.post_ms = (1 - alpha) * self.post_ms + alpha * other.post_ms
        self.e2e_ms = (1 - alpha) * self.e2e_ms + alpha * other.e2e_ms
        self.drift_ppm = (1 - alpha) * self.drift_ppm + alpha * other.drift_ppm

