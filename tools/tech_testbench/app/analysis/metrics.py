"""Per-trial metrics from dry_ref / wet_out (+ mics / hats / stimulus)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .audio import (
    band_level_db,
    fullband_level_db,
    load_wav,
    magnitude_fr_db,
    rms_dbfs,
    suppression_db,
    to_mono,
)


def _load_trial_json(trial_dir: Path) -> dict[str, Any]:
    path = trial_dir / "trial.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _find_hats(trial_dir: Path) -> Path | None:
    for name in ("hats_stereo.wav", "hats_l.wav", "hats_r.wav", "hats_mono.wav"):
        p = trial_dir / name
        if p.is_file():
            return p
    # Any hats_*.wav
    matches = sorted(trial_dir.glob("hats_*.wav"))
    return matches[0] if matches else None


def analyze_trial_dir(
    trial_dir: str | Path,
    *,
    f_lo: float = 300.0,
    f_hi: float = 3400.0,
    compute_fr: bool = True,
) -> dict[str, Any]:
    """Compute suppression / levels / optional FR for one TrialStore trial folder."""
    trial_dir = Path(trial_dir)
    doc = _load_trial_json(trial_dir)
    out: dict[str, Any] = {
        "run_id": doc.get("run_id") or trial_dir.parent.name,
        "trial_id": doc.get("trial_id") or trial_dir.name,
        "kind": doc.get("kind"),
        "acoustic_profile": doc.get("acoustic_profile"),
        "host": doc.get("host") or doc.get("host_target"),
        "runtime": doc.get("runtime"),
        "distance_m": doc.get("distance_m"),
        "azimuth_deg": doc.get("azimuth_deg", doc.get("angle_deg")),
        "elevation_deg": doc.get("elevation_deg", 0.0 if doc.get("angle_deg") is not None else None),
        "gains": doc.get("gains"),
    }

    dry_p = trial_dir / "dry_ref.wav"
    wet_p = trial_dir / "wet_out.wav"
    dry = wet = None
    sr = None
    if dry_p.is_file():
        sr_d, dry_a = load_wav(dry_p)
        dry = to_mono(dry_a)
        sr = sr_d
        out["dry_fullband_db"] = fullband_level_db(dry, sr_d)
        if compute_fr:
            out["dry_fr"] = magnitude_fr_db(dry, sr_d)
    if wet_p.is_file():
        sr_w, wet_a = load_wav(wet_p)
        wet = to_mono(wet_a)
        sr = sr if sr is not None else sr_w
        out["wet_fullband_db"] = fullband_level_db(wet, sr_w)
        if compute_fr:
            out["wet_fr"] = magnitude_fr_db(wet, sr_w)
    if dry is not None and wet is not None:
        if sr is None:
            sr = 24000
        # Keep speech-band suppression for legacy aggregates; also fullband delta.
        internal = suppression_db(dry, wet, int(sr), f_lo=f_lo, f_hi=f_hi)
        out["internal_suppression"] = internal
        out["suppression_db"] = internal["suppression_db"]
        out["dry_band_db"] = internal["dry_band_db"]
        out["wet_band_db"] = internal["wet_band_db"]
        out["suppression_fullband_db"] = float(out["dry_fullband_db"]) - float(
            out["wet_fullband_db"]
        )
        if compute_fr:
            dry_fr = out.get("dry_fr") or magnitude_fr_db(dry, int(sr))
            wet_fr = out.get("wet_fr") or magnitude_fr_db(wet, int(sr))
            n = min(len(dry_fr["freq_hz"]), len(wet_fr["freq_hz"]))
            out["fr"] = {
                "freq_hz": dry_fr["freq_hz"][:n],
                "dry_mag_db": dry_fr["mag_db"][:n],
                "wet_mag_db": wet_fr["mag_db"][:n],
                "suppression_fr_db": [
                    float(dry_fr["mag_db"][i] - wet_fr["mag_db"][i]) for i in range(n)
                ],
                "sr": sr,
                "note": "peak-normalized magnitude; compare relative shapes",
            }
    elif dry_p.is_file() or wet_p.is_file():
        out["warn"] = "need both dry_ref.wav and wet_out.wav for internal suppression"

    # Mapped 6ch preferred; fall back to raw polar / raw 8ch (first 6 channels).
    mic_wav = None
    for name in ("mapped_6ch.wav", "raw_polar_8ch.wav", "raw_8ch.wav"):
        cand = trial_dir / name
        if cand.is_file():
            mic_wav = cand
            break
    if mic_wav is not None:
        sr_m, mic_a = load_wav(mic_wav)
        if mic_a.ndim == 1:
            mic_a = mic_a[:, None]
        n_ch = min(6, int(mic_a.shape[1]))
        mic_levels: list[float] = []
        mic_frs: list[dict[str, Any]] = []
        for ch in range(n_ch):
            ch_x = mic_a[:, ch].astype("float64", copy=False)
            mic_levels.append(fullband_level_db(ch_x, sr_m))
            if compute_fr:
                mic_frs.append(magnitude_fr_db(ch_x, sr_m))
        out["mic_wav"] = mic_wav.name
        out["mic_fullband_db"] = mic_levels
        out["mic_n_channels"] = n_ch
        if compute_fr and mic_frs:
            out["mic_fr"] = mic_frs

    # Ear-rig L/R (prefer split files; fall back to stereo channels).
    hats_l_p = trial_dir / "hats_l.wav"
    hats_r_p = trial_dir / "hats_r.wav"
    hats_stereo_p = trial_dir / "hats_stereo.wav"
    if hats_l_p.is_file():
        sr_l, left_a = load_wav(hats_l_p)
        left = to_mono(left_a)
        out["hats_l_fullband_db"] = fullband_level_db(left, sr_l)
        if compute_fr:
            out["hats_l_fr"] = magnitude_fr_db(left, sr_l)
    if hats_r_p.is_file():
        sr_r, right_a = load_wav(hats_r_p)
        right = to_mono(right_a)
        out["hats_r_fullband_db"] = fullband_level_db(right, sr_r)
        if compute_fr:
            out["hats_r_fr"] = magnitude_fr_db(right, sr_r)
    if not hats_l_p.is_file() and not hats_r_p.is_file() and hats_stereo_p.is_file():
        sr_s, stereo = load_wav(hats_stereo_p)
        if stereo.ndim == 1:
            stereo = stereo[:, None]
        if stereo.shape[1] >= 1:
            out["hats_l_fullband_db"] = fullband_level_db(stereo[:, 0], sr_s)
            if compute_fr:
                out["hats_l_fr"] = magnitude_fr_db(stereo[:, 0], sr_s)
        if stereo.shape[1] >= 2:
            out["hats_r_fullband_db"] = fullband_level_db(stereo[:, 1], sr_s)
            if compute_fr:
                out["hats_r_fr"] = magnitude_fr_db(stereo[:, 1], sr_s)

    hats_p = _find_hats(trial_dir)
    if hats_p is not None:
        sr_h, hats_a = load_wav(hats_p)
        h = to_mono(hats_a)
        out["hats"] = {
            "file": hats_p.name,
            "sr": sr_h,
            "band_db": band_level_db(h, sr_h, f_lo=f_lo, f_hi=f_hi),
            "fullband_db": fullband_level_db(h, sr_h),
            "rms_dbfs": rms_dbfs(h),
            "l_fullband_db": out.get("hats_l_fullband_db"),
            "r_fullband_db": out.get("hats_r_fullband_db"),
        }
        if compute_fr:
            out["hats_fr"] = magnitude_fr_db(h, sr_h)
        out["product_hats_band_db"] = out["hats"]["band_db"]
        out["hats_fullband_db"] = out["hats"]["fullband_db"]

    stim_p = trial_dir / "stimulus_used.wav"
    if stim_p.is_file():
        sr_s, stim_a = load_wav(stim_p)
        stim = to_mono(stim_a)
        out["stimulus_fullband_db"] = fullband_level_db(stim, sr_s)
        if compute_fr:
            out["stimulus_fr"] = magnitude_fr_db(stim, sr_s)

    return out


def write_trial_analysis(
    trial_dir: str | Path,
    *,
    f_lo: float = 300.0,
    f_hi: float = 3400.0,
    write_plots: bool = True,
    merge_existing_metrics: bool = True,
) -> dict[str, Any]:
    """Analyze one trial; write ``metrics.json`` + ``plots/`` series (+ PNG if possible)."""
    trial_dir = Path(trial_dir)
    plots_dir = trial_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = analyze_trial_dir(trial_dir, f_lo=f_lo, f_hi=f_hi, compute_fr=True)

    existing: dict[str, Any] = {}
    metrics_path = trial_dir / "metrics.json"
    if merge_existing_metrics and metrics_path.is_file():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    merged = {**existing, **metrics, "analysis": "offline_taps_v1"}
    metrics_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    store_pack = _trial_store_for(trial_dir)

    # --- FR series ---
    fr = metrics.get("fr")
    if fr:
        stem = "suppression_fr"
        csv_path = plots_dir / f"{stem}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["freq_hz", "dry_mag_db", "wet_mag_db", "suppression_fr_db"]
            )
            w.writeheader()
            for i, freq in enumerate(fr["freq_hz"]):
                w.writerow(
                    {
                        "freq_hz": freq,
                        "dry_mag_db": fr["dry_mag_db"][i],
                        "wet_mag_db": fr["wet_mag_db"][i],
                        "suppression_fr_db": fr["suppression_fr_db"][i],
                    }
                )
        png: Path | None = None
        if write_plots:
            png = _plot_fr(fr, plots_dir / f"{stem}.png", title=f"{trial_dir.name} FR")
        _register_plot_artifacts(
            store_pack,
            trial_dir,
            stem=stem,
            image=png,
            series_csv=csv_path,
            series_json=fr,
        )

    hats_fr = metrics.get("hats_fr")
    if hats_fr:
        stem = "hats_fr"
        csv_path = plots_dir / f"{stem}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["freq_hz", "mag_db"])
            w.writeheader()
            for freq, mag in zip(hats_fr["freq_hz"], hats_fr["mag_db"]):
                w.writerow({"freq_hz": freq, "mag_db": mag})
        png = None
        if write_plots:
            png = _plot_single_fr(hats_fr, plots_dir / f"{stem}.png", title="HATS FR")
        _register_plot_artifacts(
            store_pack,
            trial_dir,
            stem=stem,
            image=png,
            series_csv=csv_path,
            series_json=hats_fr,
        )

    # Update metrics_summary on trial.json when present.
    tj = trial_dir / "trial.json"
    if tj.is_file():
        try:
            doc = json.loads(tj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc = {}
        doc["metrics_summary"] = {
            "suppression_db": metrics.get("suppression_db"),
            "dry_band_db": metrics.get("dry_band_db"),
            "wet_band_db": metrics.get("wet_band_db"),
            "dry_fullband_db": metrics.get("dry_fullband_db"),
            "wet_fullband_db": metrics.get("wet_fullband_db"),
            "mic_fullband_db": metrics.get("mic_fullband_db"),
            "hats_l_fullband_db": metrics.get("hats_l_fullband_db"),
            "hats_r_fullband_db": metrics.get("hats_r_fullband_db"),
            "stimulus_fullband_db": metrics.get("stimulus_fullband_db"),
            "product_hats_band_db": metrics.get("product_hats_band_db"),
        }
        tj.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    # Relative mic delay for levels / raw_polar / raw taps (Part A).
    try:
        from .mic_delay import analyze_mic_delay_trial

        delay = analyze_mic_delay_trial(trial_dir, write_plots=write_plots)
        if delay:
            merged["relative_delay_ms"] = delay.get("relative_delay_ms")
            merged["ref_channel"] = delay.get("ref_channel")
            merged["mic_delay"] = delay
            metrics_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    except Exception:
        pass

    # ESS acoustic timing-ref (controlled_ess only; skipped for in_context_speech).
    try:
        from .timing_ref import analyze_timing_ref_trial

        tj_doc = _load_trial_json(trial_dir)
        timing = analyze_timing_ref_trial(
            trial_dir,
            campaign_kind=str(tj_doc.get("campaign_kind") or ""),
            play_channels=str(tj_doc.get("stimulus_play_channels") or "left"),
            write_plots=write_plots,
        )
        if timing is not None:
            merged["timing_ref"] = timing
            metrics_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    except Exception:
        pass

    return merged


def _trial_store_for(trial_dir: Path):
    """Return (TrialStore, TrialPaths) when layout looks like measure/run/trial."""
    try:
        from ..storage.trial_store import TrialStore
    except ImportError:  # pragma: no cover
        try:
            from tools.tech_testbench.app.storage.trial_store import TrialStore
        except ImportError:
            return None
    run_dir = trial_dir.parent
    measure_root = run_dir.parent
    if not (trial_dir / "trial.json").is_file():
        return None
    store = TrialStore(measure_root)
    return store, store.trial_paths(run_dir.name, trial_dir.name)


def _register_plot_artifacts(
    store_pack,
    trial_dir: Path,
    *,
    stem: str,
    image: Path | None,
    series_csv: Path,
    series_json: dict[str, Any],
) -> None:
    """Index CSV (+ optional PNG) via TrialStore.register_plot, else fall back."""
    if store_pack is not None:
        store, paths = store_pack
        store.register_plot(
            paths,
            stem=stem,
            image=image,
            series_csv=series_csv,
            series_json=series_json,
        )
        return
    # Fallback when TrialStore unavailable: keep series JSON + trial.json index.
    plots_dir = trial_dir / "plots"
    json_path = plots_dir / f"{stem}.json"
    json_path.write_text(json.dumps(series_json, indent=2), encoding="utf-8")
    plot_rels = [f"plots/{stem}.csv", f"plots/{stem}.json"]
    if image is not None:
        plot_rels.append(f"plots/{stem}.png")
    tj = trial_dir / "trial.json"
    if not tj.is_file():
        return
    try:
        doc = json.loads(tj.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        doc = {}
    arts = doc.setdefault("artifacts", {})
    plots = list(arts.get("plots") or [])
    for rel in plot_rels:
        if rel not in plots:
            plots.append(rel)
    arts["plots"] = plots
    tj.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _plot_fr(fr: dict[str, Any], dest: Path, *, title: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(fr["freq_hz"], fr["dry_mag_db"], label="dry_ref", lw=1.2)
    ax.semilogx(fr["freq_hz"], fr["wet_mag_db"], label="wet_out", lw=1.2)
    ax.semilogx(fr["freq_hz"], fr["suppression_fr_db"], label="dry−wet", lw=1.0, alpha=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB, peak-norm)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest


def _plot_single_fr(fr: dict[str, Any], dest: Path, *, title: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(fr["freq_hz"], fr["mag_db"], label="hats", lw=1.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB, peak-norm)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest
