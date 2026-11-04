"""End-to-end latency across stimulus → raw → dry → wet taps (ms)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audio import load_wav, to_mono
from .mic_delay import gcc_phat_delay_samples

# Stage order for the chain report.
STAGE_KEYS = ("stimulus", "raw", "dry", "wet")

# On-disk names under a trial folder.
WAV_BY_STAGE = {
    "stimulus": "stimulus_used.wav",
    "raw": "raw_8ch.wav",
    "dry": "dry_ref.wav",
    "wet": "wet_out.wav",
}


def _resample_mono(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interp resample to ``sr_out`` (good enough for GCC-PHAT lag)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if sr_in == sr_out or x.size == 0:
        return x
    if sr_in <= 0 or sr_out <= 0:
        return x
    n_out = max(1, int(round(x.size * float(sr_out) / float(sr_in))))
    t_in = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float64, copy=False)


def _pair_latency_ms(
    ref: np.ndarray,
    sig: np.ndarray,
    *,
    sr: int,
    max_tau_s: float,
) -> dict[str, float]:
    """Latency of ``sig`` relative to ``ref`` in ms (positive = sig later)."""
    lag, peak = gcc_phat_delay_samples(ref, sig, max_tau_s=max_tau_s, sr=sr)
    ms = 1000.0 * lag / float(sr) if sr else 0.0
    return {
        "latency_ms": round(ms, 3),
        "latency_samples": round(lag, 3),
        "peak_corr": round(float(peak), 5),
    }


def measure_e2e_latency(
    signals: dict[str, np.ndarray],
    sample_rates: dict[str, int],
    *,
    analysis_sr: int | None = None,
    max_tau_s: float = 1.0,
) -> dict[str, Any]:
    """Cross-correlate available stages; return pairwise + chain latencies in ms.

    ``signals`` keys are subset of STAGE_KEYS; values are mono float arrays.
    Missing stages are skipped; pairs that need a missing stage are omitted.
    """
    present = [k for k in STAGE_KEYS if k in signals and signals[k] is not None]
    if len(present) < 2:
        raise ValueError(
            f"need at least two of {STAGE_KEYS} to measure latency; got {present}"
        )

    srs = [int(sample_rates[k]) for k in present if sample_rates.get(k)]
    if not srs:
        raise ValueError("sample rates missing for loaded stages")
    target_sr = int(analysis_sr) if analysis_sr else int(min(srs))

    aligned: dict[str, np.ndarray] = {}
    for k in present:
        aligned[k] = _resample_mono(signals[k], int(sample_rates[k]), target_sr)

    pairs: dict[str, dict[str, Any]] = {}
    # Adjacent chain + cumulative from stimulus when present.
    chain_specs = [
        ("stimulus", "raw"),
        ("raw", "dry"),
        ("dry", "wet"),
        ("stimulus", "dry"),
        ("stimulus", "wet"),
        ("raw", "wet"),
    ]
    for a, b in chain_specs:
        if a not in aligned or b not in aligned:
            continue
        key = f"{a}_to_{b}"
        pairs[key] = {
            "from": a,
            "to": b,
            **_pair_latency_ms(aligned[a], aligned[b], sr=target_sr, max_tau_s=max_tau_s),
        }

    # Adjacent step table in pipeline order.
    steps: list[dict[str, Any]] = []
    for a, b in (("stimulus", "raw"), ("raw", "dry"), ("dry", "wet")):
        key = f"{a}_to_{b}"
        if key in pairs:
            steps.append({"step": key, **pairs[key]})

    # Sum of adjacent hops when all three exist (approx; xcorr hops may disagree slightly).
    hop_sum_ms: float | None = None
    if all(f"{a}_to_{b}" in pairs for a, b in (("stimulus", "raw"), ("raw", "dry"), ("dry", "wet"))):
        hop_sum_ms = round(
            float(pairs["stimulus_to_raw"]["latency_ms"])
            + float(pairs["raw_to_dry"]["latency_ms"])
            + float(pairs["dry_to_wet"]["latency_ms"]),
            3,
        )

    summary_ms = {
        "stimulus_to_raw_ms": (pairs.get("stimulus_to_raw") or {}).get("latency_ms"),
        "raw_to_dry_ms": (pairs.get("raw_to_dry") or {}).get("latency_ms"),
        "dry_to_wet_ms": (pairs.get("dry_to_wet") or {}).get("latency_ms"),
        "stimulus_to_dry_ms": (pairs.get("stimulus_to_dry") or {}).get("latency_ms"),
        "stimulus_to_wet_ms": (pairs.get("stimulus_to_wet") or {}).get("latency_ms"),
        "raw_to_wet_ms": (pairs.get("raw_to_wet") or {}).get("latency_ms"),
        "adjacent_hop_sum_ms": hop_sum_ms,
    }

    return {
        "type": "e2e_latency",
        "method": "gcc_phat_or_xcorr",
        "unit": "ms",
        "analysis_sr": target_sr,
        "max_tau_s": max_tau_s,
        "stages_present": present,
        "pairs": pairs,
        "steps": steps,
        "summary_ms": summary_ms,
    }


def load_trial_stage_signals(
    trial_dir: str | Path,
    *,
    raw_channel: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, str]]:
    """Load mono stage signals from a trial folder.

    ``raw_channel``: if set, use that channel of raw_8ch; else mean of all channels.
    Returns (signals, sample_rates, source_wav_names).
    """
    trial_dir = Path(trial_dir)
    signals: dict[str, np.ndarray] = {}
    rates: dict[str, int] = {}
    sources: dict[str, str] = {}
    for stage, name in WAV_BY_STAGE.items():
        path = trial_dir / name
        if not path.is_file():
            continue
        sr, audio = load_wav(path)
        if stage == "raw" and raw_channel is not None and audio.ndim == 2:
            ch = int(np.clip(raw_channel, 0, audio.shape[1] - 1))
            mono = audio[:, ch].astype(np.float64, copy=False)
        else:
            mono = to_mono(audio)
        signals[stage] = mono
        rates[stage] = int(sr)
        sources[stage] = name
    return signals, rates, sources


def _apply_timing_ref_absolute_legs(
    result: dict[str, Any],
    timing_ref: dict[str, Any] | None,
) -> dict[str, Any]:
    """Prefer ESS timing-ref absolute stim→* ms when available; keep tap↔tap hops."""
    if not timing_ref or not timing_ref.get("enabled"):
        return result

    summary = dict(result.get("summary_ms") or {})
    pairs = dict(result.get("pairs") or {})
    used: list[str] = []

    def _override(pair_key: str, summary_key: str, ms_key: str) -> None:
        ms = timing_ref.get(ms_key)
        if ms is None:
            return
        ms_f = float(ms)
        summary[summary_key] = round(ms_f, 3)
        used.append(pair_key)
        # Update / insert pair entry for plots + CSV.
        frm, to = pair_key.split("_to_", 1)
        prev = pairs.get(pair_key) or {}
        pairs[pair_key] = {
            "from": frm,
            "to": to,
            "latency_ms": round(ms_f, 3),
            "latency_samples": prev.get("latency_samples"),
            "peak_corr": prev.get("peak_corr"),
            "source": "timing_ref",
        }

    _override("stimulus_to_raw", "stimulus_to_raw_ms", "stim_to_raw_ms")
    _override("stimulus_to_dry", "stimulus_to_dry_ms", "stim_to_dry_ms")
    _override("stimulus_to_wet", "stimulus_to_wet_ms", "stim_to_wet_ms")

    # Refresh adjacent hop sum if all three adjacent legs exist.
    hop_sum_ms: float | None = None
    if all(
        f"{a}_to_{b}" in pairs
        for a, b in (("stimulus", "raw"), ("raw", "dry"), ("dry", "wet"))
    ):
        hop_sum_ms = round(
            float(pairs["stimulus_to_raw"]["latency_ms"])
            + float(pairs["raw_to_dry"]["latency_ms"])
            + float(pairs["dry_to_wet"]["latency_ms"]),
            3,
        )
    summary["adjacent_hop_sum_ms"] = hop_sum_ms

    steps: list[dict[str, Any]] = []
    for a, b in (("stimulus", "raw"), ("raw", "dry"), ("dry", "wet")):
        key = f"{a}_to_{b}"
        if key in pairs:
            steps.append({"step": key, **pairs[key]})

    result = dict(result)
    result["summary_ms"] = summary
    result["pairs"] = pairs
    result["steps"] = steps
    result["timing_ref_absolute_legs"] = used
    if used:
        result["method"] = "timing_ref_absolute+gcc_phat_adjacent"
    return result


def analyze_e2e_latency_trial(
    trial_dir: str | Path,
    *,
    max_tau_s: float = 1.0,
    analysis_sr: int | None = None,
    raw_channel: int | None = None,
    write_plots: bool = True,
) -> dict[str, Any] | None:
    """Measure e2e latency for a trial; write metrics/plots. Returns result or None."""
    trial_dir = Path(trial_dir)
    signals, rates, sources = load_trial_stage_signals(
        trial_dir, raw_channel=raw_channel
    )
    if len(signals) < 2:
        return None

    result = measure_e2e_latency(
        signals,
        rates,
        analysis_sr=analysis_sr,
        max_tau_s=max_tau_s,
    )
    result["source_wavs"] = sources
    if raw_channel is not None:
        result["raw_channel"] = int(raw_channel)

    # Prefer controlled_ess timing-ref absolute stim→* when present (or compute now).
    timing_ref = None
    metrics_path_pre = trial_dir / "metrics.json"
    if metrics_path_pre.is_file():
        try:
            timing_ref = (json.loads(metrics_path_pre.read_text(encoding="utf-8")) or {}).get(
                "timing_ref"
            )
        except json.JSONDecodeError:
            timing_ref = None
    if timing_ref is None or not timing_ref.get("enabled"):
        try:
            from .timing_ref import analyze_timing_ref_trial

            tj = trial_dir / "trial.json"
            doc: dict[str, Any] = {}
            if tj.is_file():
                try:
                    doc = json.loads(tj.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    doc = {}
            timing_ref = analyze_timing_ref_trial(
                trial_dir,
                campaign_kind=str(doc.get("campaign_kind") or ""),
                play_channels=str(doc.get("stimulus_play_channels") or "left"),
                max_tau_s=max_tau_s,
                write_plots=write_plots,
            )
        except Exception:
            timing_ref = None
    result = _apply_timing_ref_absolute_legs(result, timing_ref)

    plots_dir = trial_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = "e2e_latency"
    csv_path = plots_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["step", "from", "to", "latency_ms", "latency_samples", "peak_corr"],
        )
        w.writeheader()
        for step in result["steps"]:
            w.writerow(
                {
                    "step": step["step"],
                    "from": step["from"],
                    "to": step["to"],
                    "latency_ms": step["latency_ms"],
                    "latency_samples": step["latency_samples"],
                    "peak_corr": step["peak_corr"],
                }
            )
        # Also dump non-adjacent pairs for completeness.
        for key, pair in result["pairs"].items():
            if any(s["step"] == key for s in result["steps"]):
                continue
            w.writerow(
                {
                    "step": key,
                    "from": pair["from"],
                    "to": pair["to"],
                    "latency_ms": pair["latency_ms"],
                    "latency_samples": pair["latency_samples"],
                    "peak_corr": pair["peak_corr"],
                }
            )
    (plots_dir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    png = None
    if write_plots:
        png = _plot_e2e(result, plots_dir / f"{stem}.png")

    metrics_path = trial_dir / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    except json.JSONDecodeError:
        metrics = {}
    metrics["e2e_latency"] = result
    metrics["e2e_latency_ms"] = result["summary_ms"]
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
        doc["e2e_latency_ms"] = result["summary_ms"]
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
        if tj.is_file():
            store.save_trial_json(paths, json.loads(tj.read_text(encoding="utf-8")))
    except Exception:
        pass

    return result


def _plot_e2e(result: dict[str, Any], dest: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    steps = result.get("steps") or []
    labels = [str(s.get("step", "?")) for s in steps]
    values = [float(s.get("latency_ms") or 0.0) for s in steps]
    if not labels:
        sm = result.get("summary_ms") or {}
        for k in (
            "stimulus_to_raw_ms",
            "raw_to_dry_ms",
            "dry_to_wet_ms",
            "stimulus_to_wet_ms",
        ):
            if sm.get(k) is not None:
                labels.append(k.replace("_ms", ""))
                values.append(float(sm[k]))

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(range(len(values)), values, color="#2a9d8f")
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("E2E latency (GCC-PHAT)")
    e2e = (result.get("summary_ms") or {}).get("stimulus_to_wet_ms")
    if e2e is not None:
        ax.annotate(
            f"stimulus→wet {e2e} ms",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            color="#264653",
        )
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest
