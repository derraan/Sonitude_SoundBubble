"""ESS acoustic timing-ref detection (controlled_ess only).

Extracts the timing-ref output leg from ``stimulus_used.wav`` (Left for
``*_refL``), builds a short click template, and GCC-PHAT-locates that marker
in each capture stream. Absolute stim→stream lags feed E2E latency; align
offsets support IR registration. Does **not** replace mic↔mic TDOA.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audio import load_wav, to_mono
from .mic_delay import gcc_phat_delay_samples

METHOD = "gcc_phat_template_from_stimulus_ref_leg"

# Capture WAVs to scan (first existing multi-ch wins for raw_* fan-out).
_RAW_CANDIDATES = (
    "levels_8ch.wav",
    "raw_polar_8ch.wav",
    "raw_8ch.wav",
    "mapped_6ch.wav",
)
_MONO_STREAMS = (
    ("dry_ref", "dry_ref.wav"),
    ("wet_out", "wet_out.wav"),
    ("hats_l", "hats_l.wav"),
    ("hats_r", "hats_r.wav"),
    ("hats_stereo", "hats_stereo.wav"),
)


def ref_leg_from_play_channels(play_channels: str | None) -> str:
    """Map Setup play_channels to stimulus channel used as timing-ref leg."""
    v = (play_channels or "left").strip().lower()
    if v in ("right", "r"):
        return "right"
    # left | both | default → Left (REW *_refL convention)
    return "left"


def _channel_index(ref_leg: str) -> int:
    return 1 if ref_leg == "right" else 0


def _resample_mono(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if sr_in == sr_out or x.size == 0 or sr_in <= 0 or sr_out <= 0:
        return x
    n_out = max(1, int(round(x.size * float(sr_out) / float(sr_in))))
    t_in = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float64, copy=False)


def extract_timing_template(
    stimulus: np.ndarray,
    sr: int,
    *,
    ref_leg: str = "left",
    template_ms: float = 25.0,
    search_s: float = 2.0,
) -> tuple[np.ndarray, int] | None:
    """Return (template, stimulus_t0_sample) from the timing-ref leg, or None.

    ``stimulus`` is ``[T, C]`` or ``[T]``. Finds the peak of a short energy
    envelope in the first ``search_s`` seconds of the ref leg.
    """
    x = np.asarray(stimulus, dtype=np.float64)
    if x.ndim == 1:
        leg = x
    else:
        ch = _channel_index(ref_leg)
        if x.shape[1] <= ch:
            if x.shape[1] < 1:
                return None
            leg = x[:, 0]
        else:
            leg = x[:, ch]

    if leg.size < 32 or sr <= 0:
        return None

    search_n = min(leg.size, max(32, int(float(search_s) * float(sr))))
    seg = leg[:search_n]
    # Prefer raw |sample| peak for sharp REW-style clicks; fall back to energy envelope.
    abs_seg = np.abs(seg)
    peak_abs = int(np.argmax(abs_seg))
    peak_abs_val = float(abs_seg[peak_abs])
    floor_abs = float(np.median(abs_seg)) + 1e-20
    win = max(1, int(0.002 * float(sr)))
    env = np.convolve(seg * seg, np.ones(win) / float(win), mode="same")
    peak_env = int(np.argmax(env))
    peak_env_val = float(env[peak_env])
    floor_env = float(np.median(env)) + 1e-20
    if peak_abs_val >= 10.0 * floor_abs and peak_abs_val >= 1e-6:
        peak = peak_abs
        peak_val = peak_abs_val
        floor = floor_abs
    else:
        peak = peak_env
        peak_val = peak_env_val
        floor = floor_env
    # Require a decisive transient (click / timing marker).
    if peak_val < 20.0 * floor and peak_val < 1e-6:
        return None

    half = max(8, int(0.5 * float(template_ms) * 1e-3 * float(sr)))
    start = max(0, peak - half)
    end = min(leg.size, peak + half)
    template = leg[start:end].copy()
    if template.size < 16:
        return None
    template -= float(np.mean(template))
    if float(np.max(np.abs(template))) < 1e-9:
        return None
    return template, int(peak)


def locate_timing_ref(
    stream: np.ndarray,
    sr: int,
    template: np.ndarray,
    *,
    max_tau_s: float = 1.0,
    stimulus_t0_sample: int = 0,
    stimulus_ref_leg: np.ndarray | None = None,
    stimulus_sr: int | None = None,
) -> dict[str, float]:
    """Locate timing marker in ``stream``; positive lag = stream later than stim t0.

    Prefer GCC of the full stimulus ref-leg (resampled) against the stream when
    provided; otherwise pad the short template at sample 0 and correlate.
    """
    sig = np.asarray(stream, dtype=np.float64).reshape(-1)
    tmpl = np.asarray(template, dtype=np.float64).reshape(-1)
    if sig.size < 16 or tmpl.size < 8 or sr <= 0:
        return {"lag_samples": 0.0, "lag_ms": 0.0, "peak_corr": 0.0}

    if stimulus_ref_leg is not None and stimulus_sr is not None and stimulus_sr > 0:
        # Full ref-leg from sample 0 (same buffer alignment as e2e stim↔stream GCC).
        # Do not slice from stimulus_t0 — that would add t0 into the reported lag.
        ref = _resample_mono(stimulus_ref_leg, int(stimulus_sr), int(sr))
        _ = stimulus_t0_sample  # retained for API / callers that pass t0 for metadata
    else:
        # Template at index 0 of a zero-padded ref (click-only match).
        ref = np.zeros(max(sig.size, tmpl.size), dtype=np.float64)
        ref[: tmpl.size] = tmpl

    # Match lengths for gcc_phat_delay_samples (uses min length).
    n = int(min(ref.size, sig.size))
    lag, peak = gcc_phat_delay_samples(ref[:n], sig[:n], max_tau_s=max_tau_s, sr=sr)
    ms = 1000.0 * lag / float(sr)
    return {
        "lag_samples": round(float(lag), 3),
        "lag_ms": round(float(ms), 4),
        "peak_corr": round(float(peak), 5),
    }


def _load_trial_meta(trial_dir: Path) -> dict[str, Any]:
    tj = trial_dir / "trial.json"
    if not tj.is_file():
        return {}
    try:
        return json.loads(tj.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def analyze_timing_ref_trial(
    trial_dir: str | Path,
    *,
    campaign_kind: str | None = None,
    play_channels: str | None = None,
    max_tau_s: float = 1.0,
    max_raw_channels: int = 6,
    write_plots: bool = True,
) -> dict[str, Any] | None:
    """Detect ESS timing ref in a trial; write plots/metrics. Returns dict or None.

    Skips when ``campaign_kind`` is not ``controlled_ess`` (reads trial.json when
    argument is None). Speech / in-context campaigns never enable this path.
    """
    trial_dir = Path(trial_dir)
    doc = _load_trial_meta(trial_dir)
    kind = (campaign_kind if campaign_kind is not None else doc.get("campaign_kind")) or ""
    kind = str(kind).strip().lower()
    if kind != "controlled_ess":
        disabled = {
            "type": "timing_ref",
            "enabled": False,
            "reason": f"campaign_kind={kind or 'unset'} (requires controlled_ess)",
            "method": METHOD,
        }
        # Do not write plots for skipped campaigns; leave metrics alone unless caller merges.
        return disabled

    stim_path = trial_dir / "stimulus_used.wav"
    if not stim_path.is_file():
        return {
            "type": "timing_ref",
            "enabled": False,
            "reason": "stimulus_used.wav missing",
            "method": METHOD,
        }

    play = play_channels if play_channels is not None else doc.get("stimulus_play_channels")
    ref_leg = ref_leg_from_play_channels(str(play) if play is not None else "left")

    stim_sr, stim_audio = load_wav(stim_path)
    extracted = extract_timing_template(stim_audio, stim_sr, ref_leg=ref_leg)
    if extracted is None:
        return {
            "type": "timing_ref",
            "enabled": False,
            "reason": "no decisive timing-ref transient on stimulus ref leg",
            "method": METHOD,
            "ref_leg": ref_leg,
            "stimulus_sr": int(stim_sr),
        }

    template, stim_t0 = extracted
    if stim_audio.ndim == 1:
        stim_leg = stim_audio.astype(np.float64, copy=False)
    else:
        ch = _channel_index(ref_leg)
        ch = min(ch, stim_audio.shape[1] - 1)
        stim_leg = stim_audio[:, ch].astype(np.float64, copy=False)

    streams: dict[str, dict[str, float]] = {}
    align_offsets: dict[str, int] = {}
    capture_srs: list[int] = []

    # Multi-channel raw / levels
    raw_path = None
    for name in _RAW_CANDIDATES:
        p = trial_dir / name
        if p.is_file():
            raw_path = p
            break
    if raw_path is not None:
        sr, audio = load_wav(raw_path)
        capture_srs.append(int(sr))
        x = np.asarray(audio, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        n_ch = min(int(x.shape[1]), int(max_raw_channels))
        for i in range(n_ch):
            key = f"raw_ch{i}"
            loc = locate_timing_ref(
                x[:, i],
                int(sr),
                template,
                max_tau_s=max_tau_s,
                stimulus_t0_sample=stim_t0,
                stimulus_ref_leg=stim_leg,
                stimulus_sr=int(stim_sr),
            )
            streams[key] = loc
            align_offsets[key] = int(round(float(loc["lag_samples"])))

    for key, fname in _MONO_STREAMS:
        p = trial_dir / fname
        if not p.is_file():
            continue
        sr, audio = load_wav(p)
        capture_srs.append(int(sr))
        mono = to_mono(audio)
        loc = locate_timing_ref(
            mono,
            int(sr),
            template,
            max_tau_s=max_tau_s,
            stimulus_t0_sample=stim_t0,
            stimulus_ref_leg=stim_leg,
            stimulus_sr=int(stim_sr),
        )
        streams[key] = loc
        align_offsets[key] = int(round(float(loc["lag_samples"])))

    if not streams:
        return {
            "type": "timing_ref",
            "enabled": False,
            "reason": "no capture WAVs found to locate timing ref",
            "method": METHOD,
            "ref_leg": ref_leg,
            "stimulus_t0_sample": int(stim_t0),
            "stimulus_sr": int(stim_sr),
        }

    capture_sr = int(capture_srs[0]) if capture_srs else int(stim_sr)
    sr_mismatch = bool(capture_srs) and any(s != int(stim_sr) for s in capture_srs)

    stim_to_raw_ms = None
    if "raw_ch0" in streams:
        stim_to_raw_ms = streams["raw_ch0"]["lag_ms"]
    else:
        raw_keys = sorted(k for k in streams if k.startswith("raw_ch"))
        if raw_keys:
            stim_to_raw_ms = streams[raw_keys[0]]["lag_ms"]

    result: dict[str, Any] = {
        "type": "timing_ref",
        "enabled": True,
        "method": METHOD,
        "ref_leg": ref_leg,
        "stimulus_t0_sample": int(stim_t0),
        "stimulus_sr": int(stim_sr),
        "capture_sr": capture_sr,
        "sample_rate_mismatch": sr_mismatch,
        "warning": (
            "stimulus_sr != capture_sr; absolute timing claims may be biased by host resample"
            if sr_mismatch
            else None
        ),
        "max_tau_s": float(max_tau_s),
        "streams": streams,
        "stim_to_raw_ms": stim_to_raw_ms,
        "stim_to_dry_ms": (streams.get("dry_ref") or {}).get("lag_ms"),
        "stim_to_wet_ms": (streams.get("wet_out") or {}).get("lag_ms"),
        "align_offsets_samples": align_offsets,
        "source_stimulus": stim_path.name,
        "source_raw_wav": raw_path.name if raw_path else None,
    }

    plots_dir = trial_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = "timing_ref"
    csv_path = plots_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["stream", "lag_ms", "lag_samples", "peak_corr", "align_offset_samples"],
        )
        w.writeheader()
        for name, loc in streams.items():
            w.writerow(
                {
                    "stream": name,
                    "lag_ms": loc["lag_ms"],
                    "lag_samples": loc["lag_samples"],
                    "peak_corr": loc["peak_corr"],
                    "align_offset_samples": align_offsets.get(name),
                }
            )
    (plots_dir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    png = None
    if write_plots:
        png = _plot_timing_ref(result, plots_dir / f"{stem}.png")

    metrics_path = trial_dir / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    except json.JSONDecodeError:
        metrics = {}
    metrics["timing_ref"] = result
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if trial_dir.joinpath("trial.json").is_file():
        try:
            tdoc = json.loads((trial_dir / "trial.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            tdoc = {}
        arts = tdoc.setdefault("artifacts", {})
        plots = list(arts.get("plots") or [])
        for rel in (f"plots/{stem}.csv", f"plots/{stem}.json"):
            if rel not in plots:
                plots.append(rel)
        if png is not None and f"plots/{stem}.png" not in plots:
            plots.append(f"plots/{stem}.png")
        arts["plots"] = plots
        tdoc["timing_ref_enabled"] = True
        tdoc["stim_to_raw_ms"] = stim_to_raw_ms
        (trial_dir / "trial.json").write_text(json.dumps(tdoc, indent=2), encoding="utf-8")

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


def _plot_timing_ref(result: dict[str, Any], dest: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    streams = result.get("streams") or {}
    if not streams:
        return None
    names = list(streams.keys())
    ms = [float(streams[n]["lag_ms"]) for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(range(len(ms)), ms, color="#e76f51")
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("stim → stream (ms)")
    ax.set_title("Timing-ref absolute lags (controlled_ess)")
    if result.get("sample_rate_mismatch"):
        ax.annotate(
            "SR mismatch warning",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=8,
            color="#9b2226",
        )
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest
