"""Relative mic delay from multi-channel ESS captures (Part A jig day)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audio import load_wav


def gcc_phat_delay_samples(
    ref: np.ndarray,
    sig: np.ndarray,
    *,
    max_tau_s: float | None = None,
    sr: int = 24000,
) -> tuple[float, float]:
    """Return (delay_samples, peak_corr) of ``sig`` relative to ``ref`` (positive = sig later).

    Uses GCC-PHAT; falls back to plain cross-correlation if the PHAT peak is
    weak (common for narrowband / pure tones).
    """
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    n = int(min(ref.size, sig.size))
    if n < 16:
        return 0.0, 0.0
    ref = ref[:n] - float(np.mean(ref[:n]))
    sig = sig[:n] - float(np.mean(sig[:n]))

    nfft = int(2 ** np.ceil(np.log2(max(32, 2 * n - 1))))
    R = np.fft.rfft(ref, n=nfft)
    S = np.fft.rfft(sig, n=nfft)
    cross = S * np.conj(R)
    denom = np.maximum(np.abs(cross), 1e-20)
    cc_phat = np.fft.irfft(cross / denom, n=nfft)
    cc_plain = np.fft.irfft(cross, n=nfft)

    lags = np.arange(nfft, dtype=np.int64)
    lags[lags > nfft // 2] -= nfft
    if max_tau_s is not None and sr > 0:
        max_lag = max(1, int(max_tau_s * float(sr)))
        mask = np.abs(lags) <= max_lag
    else:
        mask = np.ones(nfft, dtype=bool)

    def _best(cc: np.ndarray) -> tuple[float, float]:
        scored = np.where(mask, cc, -np.inf)
        idx = int(np.argmax(scored))
        lag = float(lags[idx])
        peak = float(cc[idx]) if np.isfinite(cc[idx]) else 0.0
        return lag, peak

    lag_p, peak_p = _best(cc_phat)
    lag_c, peak_c = _best(cc_plain)
    # Prefer PHAT when its peak is decisive; else plain xcorr (better on tones).
    if abs(peak_p) >= 0.15 * max(abs(peak_c), 1e-12) and abs(peak_p) > 1e-6:
        return lag_p, peak_p
    return lag_c, peak_c


def relative_delays_from_multich(
    audio: np.ndarray,
    sr: int,
    *,
    ref_channel: int = 0,
    max_channels: int = 6,
    max_tau_s: float = 0.005,
) -> dict[str, Any]:
    """Compute relative delay (ms) of channels 0..N-1 vs ``ref_channel``."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n_ch = min(int(x.shape[1]), int(max_channels))
    ref_channel = int(np.clip(ref_channel, 0, max(0, n_ch - 1)))
    ref = x[:, ref_channel]
    delays_ms: list[float] = []
    peaks: list[float] = []
    delays_samples: list[float] = []
    for ch in range(n_ch):
        if ch == ref_channel:
            delays_samples.append(0.0)
            delays_ms.append(0.0)
            peaks.append(1.0)
            continue
        lag, peak = gcc_phat_delay_samples(
            ref, x[:, ch], max_tau_s=max_tau_s, sr=sr
        )
        delays_samples.append(lag)
        delays_ms.append(1000.0 * lag / float(sr) if sr else 0.0)
        peaks.append(peak)
    return {
        "type": "mic_delay",
        "method": "gcc_phat_or_xcorr",
        "ref_channel": ref_channel,
        "sample_rate": int(sr),
        "n_channels": n_ch,
        "max_tau_s": max_tau_s,
        "relative_delay_samples": [round(v, 3) for v in delays_samples],
        "relative_delay_ms": [round(v, 4) for v in delays_ms],
        "peak_corr": [round(float(v), 5) for v in peaks],
    }


def analyze_mic_delay_trial(
    trial_dir: str | Path,
    *,
    ref_channel: int = 0,
    max_channels: int = 6,
    write_plots: bool = True,
) -> dict[str, Any] | None:
    """Find multi-ch capture; write metrics/plots. Returns delay dict or None."""
    trial_dir = Path(trial_dir)
    candidates = (
        "levels_8ch.wav",
        "raw_polar_8ch.wav",
        "raw_8ch.wav",
        "mapped_6ch.wav",
    )
    wav_path = None
    for name in candidates:
        p = trial_dir / name
        if p.is_file():
            wav_path = p
            break
    if wav_path is None:
        return None

    sr, audio = load_wav(wav_path)
    result = relative_delays_from_multich(
        audio,
        sr,
        ref_channel=ref_channel,
        max_channels=max_channels,
    )
    result["source_wav"] = wav_path.name

    plots_dir = trial_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = "mic_delay"
    csv_path = plots_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["channel", "relative_delay_ms", "relative_delay_samples", "peak_corr"],
        )
        w.writeheader()
        for i, (ms, samp, pk) in enumerate(
            zip(
                result["relative_delay_ms"],
                result["relative_delay_samples"],
                result["peak_corr"],
            )
        ):
            w.writerow(
                {
                    "channel": i,
                    "relative_delay_ms": ms,
                    "relative_delay_samples": samp,
                    "peak_corr": pk,
                }
            )
    (plots_dir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    png = None
    if write_plots:
        png = _plot_delays(result, plots_dir / f"{stem}.png")

    metrics_path = trial_dir / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    except json.JSONDecodeError:
        metrics = {}
    metrics["relative_delay_ms"] = result["relative_delay_ms"]
    metrics["relative_delay_samples"] = result["relative_delay_samples"]
    metrics["ref_channel"] = result["ref_channel"]
    metrics["mic_delay"] = result
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    tj = trial_dir / "trial.json"
    if tj.is_file():
        try:
            doc = json.loads(tj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc = {}
        arts = doc.setdefault("artifacts", {})
        plots = list(arts.get("plots") or [])
        for rel in (f"plots/{stem}.csv", f"plots/{stem}.json"):
            if rel not in plots:
                plots.append(rel)
        if png is not None and f"plots/{stem}.png" not in plots:
            plots.append(f"plots/{stem}.png")
        arts["plots"] = plots
        doc["relative_delay_ms"] = result["relative_delay_ms"]
        doc["ref_channel"] = result["ref_channel"]
        tj.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    try:
        from ..storage.trial_store import TrialStore

        run_dir = trial_dir.parent
        store = TrialStore(run_dir.parent)
        paths = store.trial_paths(run_dir.name, trial_dir.name)
        store.register_plot(
            paths,
            stem=stem,
            image=png,
            series_csv=csv_path,
            series_json=result,
        )
        store.save_metrics(paths, metrics)
    except Exception:
        pass

    return result


def _plot_delays(result: dict[str, Any], dest: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    ms = result.get("relative_delay_ms") or []
    ch = list(range(len(ms)))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(ch, ms, color="#4a90d9")
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Relative delay (ms)")
    ax.set_title(f"Mic delay vs ch{result.get('ref_channel', 0)}")
    ax.set_xticks(ch)
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest
