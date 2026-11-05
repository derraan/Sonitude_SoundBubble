"""Cross-trial aggregates: suppression-vs-distance and polar az/el (+ FR polar)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import analyze_trial_dir, write_trial_analysis

# Log-spaced grid for angle×freq aggregation (Hz). Full audio band;
# per-trial spectra are clipped at Nyquist when sr < 40 kHz.
_FR_GRID_HZ = np.geomspace(20.0, 20000.0, 128)
_FR_BANDS = (
    ("0.5-1k", 500.0, 1000.0),
    ("1-2k", 1000.0, 2000.0),
    ("2-4k", 2000.0, 4000.0),
    ("4-8k", 4000.0, 8000.0),
    ("8-20k", 8000.0, 20000.0),
)


def _host_of(doc: dict[str, Any]) -> str | None:
    h = doc.get("host")
    if h:
        return str(h)
    ht = doc.get("host_target")
    if ht == "pi_rig":
        return "pi_rig"
    if ht in ("local_windows", "windows_testbench"):
        return "windows_testbench"
    return str(ht) if ht else None


def iter_trials(
    measure_root: str | Path,
    *,
    run_id: str | None = None,
    acoustic_profile: str | None = None,
    host: str | None = None,
    runtime: str | None = None,
    kind: str | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Newest-last listing of TrialStore trials, filtered by sidecar fields."""
    measure_root = Path(measure_root)
    rows: list[tuple[Path, dict[str, Any]]] = []
    if not measure_root.is_dir():
        return rows

    run_dirs: list[Path]
    if run_id:
        run_dirs = [measure_root / run_id]
    else:
        run_dirs = sorted(p for p in measure_root.iterdir() if p.is_dir())

    for run in run_dirs:
        if not run.is_dir():
            continue
        for trial in sorted(run.iterdir()):
            tj = trial / "trial.json"
            if not tj.is_file():
                continue
            try:
                doc = json.loads(tj.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if acoustic_profile and doc.get("acoustic_profile") != acoustic_profile:
                continue
            if runtime and doc.get("runtime") != runtime:
                continue
            if kind and doc.get("kind") != kind:
                continue
            if host:
                trial_host = _host_of(doc)
                # Accept either canonical host or host_target spelling.
                if trial_host != host and doc.get("host_target") != host and doc.get("host") != host:
                    continue
            rows.append((trial, doc))
    return rows


def _interp_fr(fr: dict[str, Any], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Interpolate suppression_fr_db and wet_mag_db onto ``grid`` (Hz)."""
    freqs = np.asarray(fr.get("freq_hz") or [], dtype=np.float64)
    supp = np.asarray(fr.get("suppression_fr_db") or [], dtype=np.float64)
    wet = np.asarray(fr.get("wet_mag_db") or [], dtype=np.float64)
    if freqs.size < 2 or supp.size != freqs.size:
        return None
    order = np.argsort(freqs)
    freqs = freqs[order]
    supp = supp[order]
    if wet.size == freqs.size:
        wet = wet[order]
    else:
        wet = np.full_like(supp, np.nan)
    # np.interp needs increasing x; clamp to measured band edges.
    supp_i = np.interp(grid, freqs, supp, left=np.nan, right=np.nan)
    wet_i = np.interp(grid, freqs, wet, left=np.nan, right=np.nan)
    if np.all(np.isnan(supp_i)):
        return None
    return supp_i, wet_i


def _interp_mag_fr(fr: dict[str, Any], grid: np.ndarray) -> np.ndarray | None:
    """Interpolate a single mag_db FR onto ``grid``."""
    freqs = np.asarray(fr.get("freq_hz") or [], dtype=np.float64)
    mag = np.asarray(fr.get("mag_db") or [], dtype=np.float64)
    if freqs.size < 2 or mag.size != freqs.size:
        return None
    order = np.argsort(freqs)
    freqs = freqs[order]
    mag = mag[order]
    out = np.interp(grid, freqs, mag, left=np.nan, right=np.nan)
    if np.all(np.isnan(out)):
        return None
    return out


def _row_from_metrics(trial_dir: Path, doc: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    internal = metrics.get("internal_suppression") or {}
    fr = metrics.get("fr")
    fr_on_grid = None
    if isinstance(fr, dict):
        interp = _interp_fr(fr, _FR_GRID_HZ)
        if interp is not None:
            supp_i, wet_i = interp
            fr_on_grid = {
                "freq_hz": [float(f) for f in _FR_GRID_HZ],
                "suppression_fr_db": [
                    None if np.isnan(v) else float(v) for v in supp_i
                ],
                "wet_mag_db": [None if np.isnan(v) else float(v) for v in wet_i],
            }
    dry_fr_grid = None
    wet_fr_grid = None
    if isinstance(metrics.get("dry_fr"), dict):
        di = _interp_mag_fr(metrics["dry_fr"], _FR_GRID_HZ)
        if di is not None:
            dry_fr_grid = [None if np.isnan(v) else float(v) for v in di]
    if isinstance(metrics.get("wet_fr"), dict):
        wi = _interp_mag_fr(metrics["wet_fr"], _FR_GRID_HZ)
        if wi is not None:
            wet_fr_grid = [None if np.isnan(v) else float(v) for v in wi]
    stim_fr_grid = None
    if isinstance(metrics.get("stimulus_fr"), dict):
        si = _interp_mag_fr(metrics["stimulus_fr"], _FR_GRID_HZ)
        if si is not None:
            stim_fr_grid = [None if np.isnan(v) else float(v) for v in si]
    hats_l_fr_grid = None
    hats_r_fr_grid = None
    if isinstance(metrics.get("hats_l_fr"), dict):
        hi = _interp_mag_fr(metrics["hats_l_fr"], _FR_GRID_HZ)
        if hi is not None:
            hats_l_fr_grid = [None if np.isnan(v) else float(v) for v in hi]
    if isinstance(metrics.get("hats_r_fr"), dict):
        hi = _interp_mag_fr(metrics["hats_r_fr"], _FR_GRID_HZ)
        if hi is not None:
            hats_r_fr_grid = [None if np.isnan(v) else float(v) for v in hi]
    mic_fr_grids: list[list[float | None]] = []
    for mfr in metrics.get("mic_fr") or []:
        if not isinstance(mfr, dict):
            continue
        mi = _interp_mag_fr(mfr, _FR_GRID_HZ)
        if mi is None:
            continue
        mic_fr_grids.append([None if np.isnan(v) else float(v) for v in mi])

    return {
        "run_id": doc.get("run_id") or trial_dir.parent.name,
        "trial_id": doc.get("trial_id") or trial_dir.name,
        "kind": doc.get("kind"),
        "acoustic_profile": doc.get("acoustic_profile"),
        "host": _host_of(doc),
        "runtime": doc.get("runtime"),
        "distance_m": doc.get("distance_m"),
        "azimuth_deg": doc.get("azimuth_deg", doc.get("angle_deg")),
        "elevation_deg": doc.get(
            "elevation_deg",
            0.0 if doc.get("angle_deg") is not None else None,
        ),
        "suppression_db": metrics.get("suppression_db", internal.get("suppression_db")),
        "dry_band_db": metrics.get("dry_band_db", internal.get("dry_band_db")),
        "wet_band_db": metrics.get("wet_band_db", internal.get("wet_band_db")),
        "dry_fullband_db": metrics.get("dry_fullband_db"),
        "wet_fullband_db": metrics.get("wet_fullband_db"),
        "mic_fullband_db": metrics.get("mic_fullband_db"),
        "hats_l_fullband_db": metrics.get("hats_l_fullband_db"),
        "hats_r_fullband_db": metrics.get("hats_r_fullband_db"),
        "hats_fullband_db": metrics.get("hats_fullband_db"),
        "stimulus_fullband_db": metrics.get("stimulus_fullband_db"),
        "hats_band_db": (metrics.get("hats") or {}).get("band_db", metrics.get("hats_band_db")),
        "fr_grid": fr_on_grid,
        "dry_fr_grid": dry_fr_grid,
        "wet_fr_grid": wet_fr_grid,
        "stimulus_fr_grid": stim_fr_grid,
        "hats_l_fr_grid": hats_l_fr_grid,
        "hats_r_fr_grid": hats_r_fr_grid,
        "mic_fr_grids": mic_fr_grids,
    }


def _hold_rows(rows: list[dict[str, Any]], hold_key: str, hold_value: float = 0.0) -> list[dict[str, Any]]:
    """Rows on the reference plane for a polar cut.

    Azimuth polars hold elevation at 0 deg; elevation polars hold azimuth at 0 deg.
    Without this, off-axis elevation takes (all at az 0) were pooled into the
    azimuth polar's 0 deg point, and every el-0 azimuth take into the elevation
    polar's 0 deg point. Falls back to all rows if none sit on the plane.
    """
    def on_plane(r: dict[str, Any]) -> bool:
        try:
            return abs(float(r.get(hold_key)) - hold_value) < 1e-6
        except (TypeError, ValueError):
            return False

    held = [r for r in rows if on_plane(r)]
    return held if held else rows


def _position_key(doc: dict[str, Any]) -> tuple:
    def r(v: Any) -> float | None:
        try:
            return round(float(v), 3)
        except (TypeError, ValueError):
            return None

    return (doc.get("kind"), r(doc.get("distance_m")), r(doc.get("azimuth_deg")), r(doc.get("elevation_deg")))


def select_valid_trials(trials: list[tuple[Path, dict[str, Any]]]) -> list[tuple[Path, dict[str, Any]]]:
    """Apply operator validity marks (trial.json "valid", see mark_trial_validity).

    - "valid": false  -> always excluded.
    - If any take at a (kind, distance, azimuth, elevation) position is
      "valid": true, only the valid takes at that position are used.
    - Unmarked positions keep the legacy behaviour (all takes averaged).
    """
    validated = {_position_key(d) for _, d in trials if d.get("valid") is True}
    out = []
    for trial_dir, doc in trials:
        v = doc.get("valid")
        if v is False:
            continue
        if _position_key(doc) in validated and v is not True:
            continue
        out.append((trial_dir, doc))
    return out


def mark_trial_validity(
    trial_dir: str | Path, valid: bool, note: str = "", status: str | None = None
) -> dict[str, Any]:
    """Persist an operator validity mark into trial.json (read-modify-write).

    status: "pass" (default when valid), "check" (valid and used in aggregates,
    but flagged for manual evaluation), or "invalid" (default when not valid).
    """
    from datetime import datetime, timezone

    status = status or ("pass" if valid else "invalid")
    tj = Path(trial_dir) / "trial.json"
    doc = json.loads(tj.read_text(encoding="utf-8"))
    doc["valid"] = bool(valid)
    doc["validity_status"] = status
    doc["validity"] = {
        "valid": bool(valid),
        "status": status,
        "marked_utc": datetime.now(timezone.utc).isoformat(),
        "note": note,
        "position": {
            "distance_m": doc.get("distance_m"),
            "azimuth_deg": doc.get("azimuth_deg"),
            "elevation_deg": doc.get("elevation_deg"),
        },
    }
    tj.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def analyze_run(
    measure_root: str | Path,
    *,
    run_id: str | None = None,
    acoustic_profile: str | None = None,
    host: str | None = None,
    runtime: str | None = None,
    kind: str | None = "algorithm",
    write_per_trial: bool = True,
    write_plots: bool = True,
    f_lo: float = 300.0,
    f_hi: float = 3400.0,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze filtered trials; optionally refresh each trial + write aggregates.

    When ``run_id`` is set, aggregate artifacts land under
    ``measure/<run_id>/analysis/``. Otherwise under ``out_dir`` or
    ``measure_root/analysis/``.
    """
    measure_root = Path(measure_root)
    trials = iter_trials(
        measure_root,
        run_id=run_id,
        acoustic_profile=acoustic_profile,
        host=host,
        runtime=runtime,
        kind=kind,
    )
    trials = select_valid_trials(trials)

    rows: list[dict[str, Any]] = []
    for trial_dir, doc in trials:
        if write_per_trial:
            metrics = write_trial_analysis(
                trial_dir, f_lo=f_lo, f_hi=f_hi, write_plots=write_plots
            )
        else:
            metrics = analyze_trial_dir(trial_dir, f_lo=f_lo, f_hi=f_hi)
        rows.append(_row_from_metrics(trial_dir, doc, metrics))

    if out_dir is not None:
        agg_root = Path(out_dir)
    elif run_id:
        agg_root = measure_root / run_id / "analysis"
    else:
        agg_root = measure_root / "analysis"
    agg_root.mkdir(parents=True, exist_ok=True)

    filter_meta = {
        "measure_root": str(measure_root),
        "run_id": run_id,
        "acoustic_profile": acoustic_profile,
        "host": host,
        "runtime": runtime,
        "kind": kind,
        "n_trials": len(rows),
        "note": (
            "Do not silently compare windows_pico tap curves to pi_hats HATS curves; "
            "filter by acoustic_profile. Suppression polar uses speech-band dry−wet; "
            "polar_*_{mics,dry,wet,hats,stimulus} use absolute fullband RMS dBFS; "
            "polar_*_fr = angle×frequency (suppression or per-signal mag)."
        ),
    }

    csv_path = agg_root / "suppression_vs_distance.csv"
    fieldnames = [
        "run_id",
        "trial_id",
        "kind",
        "acoustic_profile",
        "host",
        "runtime",
        "distance_m",
        "azimuth_deg",
        "elevation_deg",
        "suppression_db",
        "dry_band_db",
        "wet_band_db",
        "hats_band_db",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})

    # JSON points without bulky fr_grid (FR lives in polar_*_fr.json).
    points_light = [{k: v for k, v in r.items() if k != "fr_grid"} for r in rows]
    series = {
        "type": "suppression_vs_distance",
        "filters": filter_meta,
        "points": points_light,
    }
    (agg_root / "suppression_vs_distance.json").write_text(
        json.dumps(series, indent=2), encoding="utf-8"
    )

    polar_az = _polar_series(
        _hold_rows(rows, "elevation_deg"), angle_key="azimuth_deg", hold_key="elevation_deg", value_key="suppression_db"
    )
    polar_el = _polar_series(
        _hold_rows(rows, "azimuth_deg"), angle_key="elevation_deg", hold_key="azimuth_deg", value_key="suppression_db"
    )
    (agg_root / "polar_azimuth.json").write_text(json.dumps(polar_az, indent=2), encoding="utf-8")
    (agg_root / "polar_elevation.json").write_text(json.dumps(polar_el, indent=2), encoding="utf-8")
    _write_polar_csv(agg_root / "polar_azimuth.csv", polar_az, value_field="suppression_db_mean")
    _write_polar_csv(agg_root / "polar_elevation.csv", polar_el, value_field="suppression_db_mean")

    written = {
        "suppression_vs_distance_csv": str(csv_path),
        "suppression_vs_distance_json": str(agg_root / "suppression_vs_distance.json"),
        "polar_azimuth_csv": str(agg_root / "polar_azimuth.csv"),
        "polar_azimuth_json": str(agg_root / "polar_azimuth.json"),
        "polar_elevation_csv": str(agg_root / "polar_elevation.csv"),
        "polar_elevation_json": str(agg_root / "polar_elevation.json"),
    }

    # Fullband absolute-level polars for each signal family.
    for angle_key, hold_key, stem_base in (
        ("azimuth_deg", "elevation_deg", "polar_azimuth"),
        ("elevation_deg", "azimuth_deg", "polar_elevation"),
    ):
        for value_key, suffix, ylabel in (
            ("dry_fullband_db", "dry", "dry fullband dBFS"),
            ("wet_fullband_db", "wet", "wet fullband dBFS"),
            ("stimulus_fullband_db", "stimulus", "stimulus fullband dBFS"),
            ("hats_l_fullband_db", "hats_l", "hats L fullband dBFS"),
            ("hats_r_fullband_db", "hats_r", "hats R fullband dBFS"),
        ):
            series = _polar_series(
                _hold_rows(rows, hold_key), angle_key=angle_key, hold_key=hold_key, value_key=value_key
            )
            stem = f"{stem_base}_{suffix}"
            written.update(
                _write_level_polar_artifacts(
                    agg_root, stem, series, ylabel=ylabel, write_plots=write_plots
                )
            )

        mics = _polar_mics_series(_hold_rows(rows, hold_key), angle_key=angle_key, hold_key=hold_key)
        written.update(
            _write_mics_polar_artifacts(
                agg_root, f"{stem_base}_mics", mics, write_plots=write_plots
            )
        )
        hats = _polar_hats_pair_series(_hold_rows(rows, hold_key), angle_key=angle_key, hold_key=hold_key)
        written.update(
            _write_hats_polar_artifacts(
                agg_root, f"{stem_base}_hats", hats, write_plots=write_plots
            )
        )

        # Mag FR polars (absolute level vs angle×freq) for dry/wet/stimulus/hats/mics.
        for fr_key, suffix in (
            ("dry_fr_grid", "dry"),
            ("wet_fr_grid", "wet"),
            ("stimulus_fr_grid", "stimulus"),
            ("hats_l_fr_grid", "hats_l"),
            ("hats_r_fr_grid", "hats_r"),
        ):
            fr_series = _polar_mag_fr_series(
                _hold_rows(rows, hold_key), angle_key=angle_key, hold_key=hold_key, fr_key=fr_key
            )
            written.update(
                _write_mag_fr_polar_artifacts(
                    agg_root, f"{stem_base}_{suffix}", fr_series, write_plots=write_plots
                )
            )
        mics_fr = _polar_mics_fr_series(_hold_rows(rows, hold_key), angle_key=angle_key, hold_key=hold_key)
        written.update(
            _write_mics_fr_polar_artifacts(
                agg_root, f"{stem_base}_mics", mics_fr, write_plots=write_plots
            )
        )

    polar_az_fr = _polar_fr_series(_hold_rows(rows, "elevation_deg"), angle_key="azimuth_deg", hold_key="elevation_deg")
    polar_el_fr = _polar_fr_series(_hold_rows(rows, "azimuth_deg"), angle_key="elevation_deg", hold_key="azimuth_deg")
    written.update(_write_polar_fr_artifacts(agg_root, "polar_azimuth", polar_az_fr, write_plots=write_plots))
    written.update(_write_polar_fr_artifacts(agg_root, "polar_elevation", polar_el_fr, write_plots=write_plots))

    if write_plots:
        written.update(write_aggregate_plots(agg_root, rows, polar_az, polar_el))

    (agg_root / "analysis_summary.json").write_text(
        json.dumps({"filters": filter_meta, "artifacts": written}, indent=2),
        encoding="utf-8",
    )
    return {"filters": filter_meta, "rows": points_light, "artifacts": written, "analysis_dir": str(agg_root)}


def _polar_series(
    rows: list[dict[str, Any]],
    *,
    angle_key: str,
    hold_key: str,
    value_key: str = "suppression_db",
) -> dict[str, Any]:
    """Group mean of ``value_key`` by angle; keep hold dimension noted."""
    buckets: dict[float, list[float]] = {}
    hold_vals: list[float] = []
    mean_field = (
        "suppression_db_mean" if value_key == "suppression_db" else f"{value_key}_mean"
    )
    for row in rows:
        ang = row.get(angle_key)
        val = row.get(value_key)
        if ang is None or val is None:
            continue
        ang_f = float(ang)
        buckets.setdefault(ang_f, []).append(float(val))
        hv = row.get(hold_key)
        if hv is not None:
            hold_vals.append(float(hv))
    angles = sorted(buckets.keys())
    means = [float(sum(buckets[a]) / len(buckets[a])) for a in angles]
    counts = [len(buckets[a]) for a in angles]
    return {
        "type": f"polar_{angle_key}_{value_key}",
        "angle_key": angle_key,
        "hold_key": hold_key,
        "value_key": value_key,
        "hold_values_seen": sorted(set(hold_vals)),
        "angles_deg": angles,
        mean_field: means,
        "values_db_mean": means,
        "n": counts,
        "points": [
            {"angle_deg": a, mean_field: m, "value_db_mean": m, "n": n}
            for a, m, n in zip(angles, means, counts)
        ],
    }


def _polar_mics_series(
    rows: list[dict[str, Any]],
    *,
    angle_key: str,
    hold_key: str,
) -> dict[str, Any]:
    """Per-mic fullband means vs angle (up to 6 channels)."""
    buckets: dict[float, list[list[float]]] = {}
    hold_vals: list[float] = []
    max_ch = 0
    for row in rows:
        ang = row.get(angle_key)
        levels = row.get("mic_fullband_db")
        if ang is None or not isinstance(levels, list) or not levels:
            continue
        ang_f = float(ang)
        vals = [float(v) for v in levels]
        max_ch = max(max_ch, len(vals))
        buckets.setdefault(ang_f, []).append(vals)
        hv = row.get(hold_key)
        if hv is not None:
            hold_vals.append(float(hv))
    angles = sorted(buckets.keys())
    curves: list[list[float | None]] = [[] for _ in range(max_ch)]
    counts: list[int] = []
    for a in angles:
        stacks = buckets[a]
        counts.append(len(stacks))
        for ch in range(max_ch):
            ch_vals = [s[ch] for s in stacks if len(s) > ch]
            curves[ch].append(float(sum(ch_vals) / len(ch_vals)) if ch_vals else None)
    return {
        "type": f"polar_mics_{angle_key}",
        "angle_key": angle_key,
        "hold_key": hold_key,
        "hold_values_seen": sorted(set(hold_vals)),
        "angles_deg": angles,
        "n_channels": max_ch,
        "mic_fullband_db_mean": curves,
        "n": counts,
        "note": "absolute fullband RMS dBFS per mapped mic channel",
    }


def _polar_hats_pair_series(
    rows: list[dict[str, Any]],
    *,
    angle_key: str,
    hold_key: str,
) -> dict[str, Any]:
    left = _polar_series(
        rows, angle_key=angle_key, hold_key=hold_key, value_key="hats_l_fullband_db"
    )
    right = _polar_series(
        rows, angle_key=angle_key, hold_key=hold_key, value_key="hats_r_fullband_db"
    )
    return {
        "type": f"polar_hats_{angle_key}",
        "angle_key": angle_key,
        "hold_key": hold_key,
        "hold_values_seen": sorted(
            set((left.get("hold_values_seen") or []) + (right.get("hold_values_seen") or []))
        ),
        "angles_deg": sorted(set((left.get("angles_deg") or []) + (right.get("angles_deg") or []))),
        "hats_l": left,
        "hats_r": right,
        "note": "absolute fullband RMS dBFS; L and R ear-rig",
    }


def _polar_mag_fr_series(
    rows: list[dict[str, Any]],
    *,
    angle_key: str,
    hold_key: str,
    fr_key: str,
) -> dict[str, Any]:
    """Mean magnitude FR vs angle for a single signal (list on FR grid)."""
    grid = _FR_GRID_HZ
    buckets: dict[float, list[np.ndarray]] = {}
    hold_vals: list[float] = []
    skipped = 0
    for row in rows:
        ang = row.get(angle_key)
        frg = row.get(fr_key)
        if ang is None or not isinstance(frg, list) or len(frg) != grid.size:
            skipped += 1
            continue
        arr = np.asarray(
            [np.nan if v is None else float(v) for v in frg], dtype=np.float64
        )
        ang_f = float(ang)
        buckets.setdefault(ang_f, []).append(arr)
        hv = row.get(hold_key)
        if hv is not None:
            hold_vals.append(float(hv))
    angles = sorted(buckets.keys())
    by_angle: list[dict[str, Any]] = []
    for a in angles:
        stack = np.vstack(buckets[a])
        mean = np.nanmean(stack, axis=0)
        by_angle.append(
            {
                "angle_deg": a,
                "n": len(buckets[a]),
                "mag_db": [None if np.isnan(v) else float(v) for v in mean],
            }
        )
    return {
        "type": f"polar_mag_fr_{angle_key}_{fr_key}",
        "angle_key": angle_key,
        "hold_key": hold_key,
        "fr_key": fr_key,
        "hold_values_seen": sorted(set(hold_vals)),
        "freq_hz": [float(f) for f in grid],
        "angles_deg": angles,
        "by_angle": by_angle,
        "n_skipped_no_fr": skipped,
        "note": "Mean peak-normalized mag FR vs angle (full audio band to Nyquist).",
    }


def _polar_mics_fr_series(
    rows: list[dict[str, Any]],
    *,
    angle_key: str,
    hold_key: str,
) -> dict[str, Any]:
    """Mean-of-mics mag FR vs angle (average across channels per trial first)."""
    grid = _FR_GRID_HZ
    buckets: dict[float, list[np.ndarray]] = {}
    hold_vals: list[float] = []
    for row in rows:
        ang = row.get(angle_key)
        grids = row.get("mic_fr_grids")
        if ang is None or not isinstance(grids, list) or not grids:
            continue
        mats = []
        for g in grids:
            if not isinstance(g, list) or len(g) != grid.size:
                continue
            mats.append(
                np.asarray([np.nan if v is None else float(v) for v in g], dtype=np.float64)
            )
        if not mats:
            continue
        mean_ch = np.nanmean(np.vstack(mats), axis=0)
        ang_f = float(ang)
        buckets.setdefault(ang_f, []).append(mean_ch)
        hv = row.get(hold_key)
        if hv is not None:
            hold_vals.append(float(hv))
    angles = sorted(buckets.keys())
    by_angle: list[dict[str, Any]] = []
    for a in angles:
        stack = np.vstack(buckets[a])
        mean = np.nanmean(stack, axis=0)
        by_angle.append(
            {
                "angle_deg": a,
                "n": len(buckets[a]),
                "mag_db": [None if np.isnan(v) else float(v) for v in mean],
            }
        )
    return {
        "type": f"polar_mics_fr_{angle_key}",
        "angle_key": angle_key,
        "hold_key": hold_key,
        "hold_values_seen": sorted(set(hold_vals)),
        "freq_hz": [float(f) for f in grid],
        "angles_deg": angles,
        "by_angle": by_angle,
        "note": "Mean of per-mic mag FR (peak-norm) vs angle.",
    }


def _polar_fr_series(
    rows: list[dict[str, Any]],
    *,
    angle_key: str,
    hold_key: str,
) -> dict[str, Any]:
    """Mean suppression_fr / wet_mag vs frequency, grouped by angle."""
    grid = _FR_GRID_HZ
    buckets_supp: dict[float, list[np.ndarray]] = {}
    buckets_wet: dict[float, list[np.ndarray]] = {}
    hold_vals: list[float] = []
    skipped = 0
    for row in rows:
        ang = row.get(angle_key)
        frg = row.get("fr_grid")
        if ang is None or not isinstance(frg, dict):
            skipped += 1
            continue
        supp = np.asarray(
            [np.nan if v is None else float(v) for v in frg.get("suppression_fr_db") or []],
            dtype=np.float64,
        )
        wet = np.asarray(
            [np.nan if v is None else float(v) for v in frg.get("wet_mag_db") or []],
            dtype=np.float64,
        )
        if supp.size != grid.size:
            skipped += 1
            continue
        ang_f = float(ang)
        buckets_supp.setdefault(ang_f, []).append(supp)
        buckets_wet.setdefault(ang_f, []).append(wet)
        hv = row.get(hold_key)
        if hv is not None:
            hold_vals.append(float(hv))

    angles = sorted(buckets_supp.keys())
    by_angle: list[dict[str, Any]] = []
    for a in angles:
        stack_s = np.vstack(buckets_supp[a])
        stack_w = np.vstack(buckets_wet[a])
        mean_s = np.nanmean(stack_s, axis=0)
        mean_w = np.nanmean(stack_w, axis=0)
        by_angle.append(
            {
                "angle_deg": a,
                "n": len(buckets_supp[a]),
                "suppression_fr_db": [
                    None if np.isnan(v) else float(v) for v in mean_s
                ],
                "wet_mag_db": [None if np.isnan(v) else float(v) for v in mean_w],
            }
        )

    bands = _band_means_from_fr(by_angle, grid)
    return {
        "type": f"polar_fr_{angle_key}",
        "angle_key": angle_key,
        "hold_key": hold_key,
        "hold_values_seen": sorted(set(hold_vals)),
        "freq_hz": [float(f) for f in grid],
        "angles_deg": angles,
        "by_angle": by_angle,
        "bands": bands,
        "n_trials_with_fr": sum(p["n"] for p in by_angle),
        "n_skipped_no_fr": skipped,
        "note": "Mean dry−wet mag FR on log grid; peak-normalized per-trial shapes.",
    }


def _band_means_from_fr(by_angle: list[dict[str, Any]], grid: np.ndarray) -> dict[str, Any]:
    band_out: dict[str, Any] = {"band_defs": [], "angles_deg": [], "series": {}}
    if not by_angle:
        return band_out
    angles = [p["angle_deg"] for p in by_angle]
    band_out["angles_deg"] = angles
    for name, lo, hi in _FR_BANDS:
        mask = (grid >= lo) & (grid <= hi)
        band_out["band_defs"].append({"name": name, "f_lo_hz": lo, "f_hi_hz": hi})
        vals: list[float | None] = []
        for p in by_angle:
            arr = np.asarray(
                [np.nan if v is None else float(v) for v in p["suppression_fr_db"]],
                dtype=np.float64,
            )
            if not np.any(mask) or np.all(np.isnan(arr[mask])):
                vals.append(None)
            else:
                vals.append(float(np.nanmean(arr[mask])))
        band_out["series"][name] = vals
    return band_out


def _write_polar_csv(
    path: Path,
    series: dict[str, Any],
    *,
    value_field: str = "suppression_db_mean",
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["angle_deg", value_field, "n"])
        w.writeheader()
        for p in series.get("points") or []:
            w.writerow(
                {
                    "angle_deg": p.get("angle_deg"),
                    value_field: p.get(value_field, p.get("value_db_mean")),
                    "n": p.get("n"),
                }
            )


def _write_level_polar_artifacts(
    agg_root: Path,
    stem: str,
    series: dict[str, Any],
    *,
    ylabel: str,
    write_plots: bool,
) -> dict[str, str]:
    written: dict[str, str] = {}
    if not series.get("angles_deg"):
        return written
    jpath = agg_root / f"{stem}.json"
    jpath.write_text(json.dumps(series, indent=2), encoding="utf-8")
    written[f"{stem}_json"] = str(jpath)
    cpath = agg_root / f"{stem}.csv"
    _write_polar_csv(cpath, series, value_field="value_db_mean")
    written[f"{stem}_csv"] = str(cpath)
    if write_plots:
        png = _plot_level_polar(agg_root / f"{stem}.png", series, ylabel=ylabel)
        if png is not None:
            written[f"{stem}_png"] = str(png)
    return written


def _write_mics_polar_artifacts(
    agg_root: Path,
    stem: str,
    series: dict[str, Any],
    *,
    write_plots: bool,
) -> dict[str, str]:
    written: dict[str, str] = {}
    if not series.get("angles_deg"):
        return written
    jpath = agg_root / f"{stem}.json"
    jpath.write_text(json.dumps(series, indent=2), encoding="utf-8")
    written[f"{stem}_json"] = str(jpath)
    angles = series.get("angles_deg") or []
    curves = series.get("mic_fullband_db_mean") or []
    cpath = agg_root / f"{stem}.csv"
    fieldnames = ["angle_deg", "n"] + [f"mic{i}_fullband_db" for i in range(len(curves))]
    with cpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        ns = series.get("n") or []
        for i, ang in enumerate(angles):
            row: dict[str, Any] = {"angle_deg": ang, "n": ns[i] if i < len(ns) else None}
            for ch, curve in enumerate(curves):
                row[f"mic{ch}_fullband_db"] = curve[i] if i < len(curve) else None
            w.writerow(row)
    written[f"{stem}_csv"] = str(cpath)
    if write_plots:
        png = _plot_mics_polar(agg_root / f"{stem}.png", series)
        if png is not None:
            written[f"{stem}_png"] = str(png)
    return written


def _write_hats_polar_artifacts(
    agg_root: Path,
    stem: str,
    series: dict[str, Any],
    *,
    write_plots: bool,
) -> dict[str, str]:
    written: dict[str, str] = {}
    left = series.get("hats_l") or {}
    right = series.get("hats_r") or {}
    if not left.get("angles_deg") and not right.get("angles_deg"):
        return written
    jpath = agg_root / f"{stem}.json"
    jpath.write_text(json.dumps(series, indent=2), encoding="utf-8")
    written[f"{stem}_json"] = str(jpath)
    angles = sorted(set((left.get("angles_deg") or []) + (right.get("angles_deg") or [])))
    l_map = {
        float(p["angle_deg"]): p.get("value_db_mean", p.get("hats_l_fullband_db_mean"))
        for p in (left.get("points") or [])
    }
    r_map = {
        float(p["angle_deg"]): p.get("value_db_mean", p.get("hats_r_fullband_db_mean"))
        for p in (right.get("points") or [])
    }
    cpath = agg_root / f"{stem}.csv"
    with cpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["angle_deg", "hats_l_fullband_db", "hats_r_fullband_db"]
        )
        w.writeheader()
        for ang in angles:
            w.writerow(
                {
                    "angle_deg": ang,
                    "hats_l_fullband_db": l_map.get(float(ang)),
                    "hats_r_fullband_db": r_map.get(float(ang)),
                }
            )
    written[f"{stem}_csv"] = str(cpath)
    if write_plots:
        png = _plot_hats_polar(agg_root / f"{stem}.png", series)
        if png is not None:
            written[f"{stem}_png"] = str(png)
    return written


def _write_mag_fr_polar_artifacts(
    agg_root: Path,
    stem: str,
    series: dict[str, Any],
    *,
    write_plots: bool,
) -> dict[str, str]:
    """Write ``{stem}_fr`` mag×angle artifacts (not suppression)."""
    written: dict[str, str] = {}
    if not series.get("angles_deg"):
        return written
    fr_stem = f"{stem}_fr"
    jpath = agg_root / f"{fr_stem}.json"
    jpath.write_text(json.dumps(series, indent=2), encoding="utf-8")
    written[f"{fr_stem}_json"] = str(jpath)
    cpath = agg_root / f"{fr_stem}.csv"
    with cpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["angle_deg", "freq_hz", "mag_db", "n"])
        w.writeheader()
        freqs = series.get("freq_hz") or []
        for p in series.get("by_angle") or []:
            for i, freq in enumerate(freqs):
                w.writerow(
                    {
                        "angle_deg": p["angle_deg"],
                        "freq_hz": freq,
                        "mag_db": (p.get("mag_db") or [None])[i],
                        "n": p.get("n"),
                    }
                )
    written[f"{fr_stem}_csv"] = str(cpath)
    if write_plots:
        png = _plot_mag_fr_heatmap(agg_root / f"{fr_stem}.png", series, title=fr_stem)
        if png is not None:
            written[f"{fr_stem}_png"] = str(png)
    return written


def _write_mics_fr_polar_artifacts(
    agg_root: Path,
    stem: str,
    series: dict[str, Any],
    *,
    write_plots: bool,
) -> dict[str, str]:
    return _write_mag_fr_polar_artifacts(
        agg_root, stem, series, write_plots=write_plots
    )


def _plot_level_polar(dest: Path, series: dict[str, Any], *, ylabel: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    angles = series.get("angles_deg") or []
    means = series.get("values_db_mean") or series.get("suppression_db_mean") or []
    if not angles:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(angles, means, marker="o", lw=1.2)
    ax.set_xlabel(f"{series.get('angle_key', 'angle')} (deg)")
    ax.set_ylabel(ylabel)
    ax.set_title(dest.stem.replace("_", " "))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest


def _plot_mics_polar(dest: Path, series: dict[str, Any]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    angles = series.get("angles_deg") or []
    curves = series.get("mic_fullband_db_mean") or []
    if not angles or not curves:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ch, curve in enumerate(curves):
        ax.plot(angles, curve, marker="o", lw=1.1, label=f"mic{ch}")
    ax.set_xlabel(f"{series.get('angle_key', 'angle')} (deg)")
    ax.set_ylabel("Fullband RMS (dBFS)")
    ax.set_title(dest.stem.replace("_", " "))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest


def _plot_hats_polar(dest: Path, series: dict[str, Any]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    left = series.get("hats_l") or {}
    right = series.get("hats_r") or {}
    fig, ax = plt.subplots(figsize=(7, 4))
    if left.get("angles_deg"):
        ax.plot(
            left["angles_deg"],
            left.get("values_db_mean") or [],
            marker="o",
            lw=1.2,
            label="hats L",
        )
    if right.get("angles_deg"):
        ax.plot(
            right["angles_deg"],
            right.get("values_db_mean") or [],
            marker="s",
            lw=1.2,
            label="hats R",
        )
    ax.set_xlabel(f"{series.get('angle_key', 'angle')} (deg)")
    ax.set_ylabel("Fullband RMS (dBFS)")
    ax.set_title(dest.stem.replace("_", " "))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest


def _plot_mag_fr_heatmap(dest: Path, series: dict[str, Any], *, title: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    angles = series.get("angles_deg") or []
    freqs = np.asarray(series.get("freq_hz") or [], dtype=np.float64)
    by_angle = series.get("by_angle") or []
    if not angles or freqs.size == 0 or not by_angle:
        return None
    mat = np.full((len(freqs), len(angles)), np.nan, dtype=np.float64)
    ang_index = {a: i for i, a in enumerate(angles)}
    for p in by_angle:
        j = ang_index.get(p["angle_deg"])
        if j is None:
            continue
        vals = np.asarray(
            [np.nan if v is None else float(v) for v in p.get("mag_db") or []],
            dtype=np.float64,
        )
        if vals.size == len(freqs):
            mat[:, j] = vals
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        extent=[min(angles), max(angles), float(freqs[0]), float(freqs[-1])],
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_yscale("log")
    ax.set_xlabel(f"{series.get('angle_key', 'angle')} (deg)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title.replace("_", " "))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mag_dB")
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest


def _write_polar_fr_artifacts(
    agg_root: Path,
    stem: str,
    series: dict[str, Any],
    *,
    write_plots: bool,
) -> dict[str, str]:
    """Write polar_*_fr and polar_*_bands JSON/CSV/(PNG)."""
    written: dict[str, str] = {}
    if not series.get("angles_deg"):
        return written

    fr_json = agg_root / f"{stem}_fr.json"
    fr_json.write_text(json.dumps(series, indent=2), encoding="utf-8")
    written[f"{stem}_fr_json"] = str(fr_json)

    fr_csv = agg_root / f"{stem}_fr.csv"
    with fr_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["angle_deg", "freq_hz", "suppression_fr_db", "wet_mag_db", "n"],
        )
        w.writeheader()
        freqs = series.get("freq_hz") or []
        for p in series.get("by_angle") or []:
            for i, freq in enumerate(freqs):
                w.writerow(
                    {
                        "angle_deg": p["angle_deg"],
                        "freq_hz": freq,
                        "suppression_fr_db": (p.get("suppression_fr_db") or [None])[i],
                        "wet_mag_db": (p.get("wet_mag_db") or [None])[i],
                        "n": p.get("n"),
                    }
                )
    written[f"{stem}_fr_csv"] = str(fr_csv)

    bands = series.get("bands") or {}
    bands_doc = {
        "type": f"{stem}_bands",
        "angle_key": series.get("angle_key"),
        "hold_key": series.get("hold_key"),
        **bands,
    }
    bands_json = agg_root / f"{stem}_bands.json"
    bands_json.write_text(json.dumps(bands_doc, indent=2), encoding="utf-8")
    written[f"{stem}_bands_json"] = str(bands_json)

    bands_csv = agg_root / f"{stem}_bands.csv"
    with bands_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["angle_deg"] + [b["name"] for b in bands.get("band_defs") or []]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        angles = bands.get("angles_deg") or []
        for i, ang in enumerate(angles):
            row: dict[str, Any] = {"angle_deg": ang}
            for b in bands.get("band_defs") or []:
                name = b["name"]
                series_vals = (bands.get("series") or {}).get(name) or []
                row[name] = series_vals[i] if i < len(series_vals) else None
            w.writerow(row)
    written[f"{stem}_bands_csv"] = str(bands_csv)

    if write_plots:
        written.update(_plot_polar_fr(agg_root, stem, series))
        written.update(_plot_polar_bands(agg_root, stem, bands_doc))
    return written


def _plot_polar_fr(agg_root: Path, stem: str, series: dict[str, Any]) -> dict[str, str]:
    written: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return written

    angles = series.get("angles_deg") or []
    freqs = np.asarray(series.get("freq_hz") or [], dtype=np.float64)
    by_angle = series.get("by_angle") or []
    if not angles or freqs.size == 0 or not by_angle:
        return written

    # Rows = freq (low→high bottom→top via extent), cols = angle order.
    mat = np.full((len(freqs), len(angles)), np.nan, dtype=np.float64)
    ang_index = {a: i for i, a in enumerate(angles)}
    for p in by_angle:
        j = ang_index.get(p["angle_deg"])
        if j is None:
            continue
        vals = np.asarray(
            [np.nan if v is None else float(v) for v in p.get("suppression_fr_db") or []],
            dtype=np.float64,
        )
        if vals.size == len(freqs):
            mat[:, j] = vals

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Use midpoints for pcolormesh-friendly imshow with log y via extent + set_yscale.
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        extent=[min(angles), max(angles), float(freqs[0]), float(freqs[-1])],
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_yscale("log")
    ax.set_xlabel(f"{series.get('angle_key', 'angle')} (deg)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{stem.replace('_', ' ')} — suppression FR (dry−wet dB)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("suppression_fr_dB")
    fig.tight_layout()
    dest = agg_root / f"{stem}_fr.png"
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    written[f"{stem}_fr_png"] = str(dest)
    return written


def _plot_polar_bands(agg_root: Path, stem: str, bands_doc: dict[str, Any]) -> dict[str, str]:
    written: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return written

    angles = bands_doc.get("angles_deg") or []
    if not angles:
        return written
    fig, ax = plt.subplots(figsize=(7, 4))
    for b in bands_doc.get("band_defs") or []:
        name = b["name"]
        ys = (bands_doc.get("series") or {}).get(name) or []
        if len(ys) != len(angles):
            continue
        ax.plot(angles, ys, marker="o", lw=1.2, label=name)
    ax.set_xlabel(f"{bands_doc.get('angle_key', 'angle')} (deg)")
    ax.set_ylabel("Mean suppression_fr (dB)")
    ax.set_title(f"{stem.replace('_', ' ')} — multi-band suppression")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    dest = agg_root / f"{stem}_bands.png"
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    written[f"{stem}_bands_png"] = str(dest)
    return written


def write_aggregate_plots(
    agg_root: str | Path,
    rows: list[dict[str, Any]],
    polar_az: dict[str, Any],
    polar_el: dict[str, Any],
) -> dict[str, str]:
    """Write aggregate PNGs plus matching CSV series (never PNG-only)."""
    agg_root = Path(agg_root)
    written: dict[str, str] = {}

    # Matching CSV beside every image (never PNG-only). Prefer an existing
    # full suppression table from analyze_run; otherwise write the plotted series.
    scatter_rows = [
        r
        for r in rows
        if r.get("distance_m") is not None and r.get("suppression_db") is not None
    ]
    csv_path = agg_root / "suppression_vs_distance.csv"
    if scatter_rows and not csv_path.is_file():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["distance_m", "suppression_db", "run_id", "trial_id"]
            )
            w.writeheader()
            for r in scatter_rows:
                w.writerow(
                    {
                        "distance_m": r.get("distance_m"),
                        "suppression_db": r.get("suppression_db"),
                        "run_id": r.get("run_id"),
                        "trial_id": r.get("trial_id"),
                    }
                )
    if csv_path.is_file():
        written["suppression_vs_distance_csv"] = str(csv_path)

    for stem, series in (("polar_azimuth", polar_az), ("polar_elevation", polar_el)):
        if series.get("angles_deg"):
            polar_csv = agg_root / f"{stem}.csv"
            _write_polar_csv(polar_csv, series, value_field="suppression_db_mean")
            written[f"{stem}_csv"] = str(polar_csv)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return written

    # Suppression vs distance
    xs = [r["distance_m"] for r in scatter_rows]
    ys = [r["suppression_db"] for r in scatter_rows]
    if xs:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(xs, ys, s=36, alpha=0.85)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Suppression (dB) = dry − wet band")
        ax.set_title("Suppression vs distance (internal taps)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        dest = agg_root / "suppression_vs_distance.png"
        fig.savefig(dest, dpi=120)
        plt.close(fig)
        written["suppression_vs_distance_png"] = str(dest)

    for name, series, title in (
        ("polar_azimuth.png", polar_az, "Polar azimuth — mean suppression"),
        ("polar_elevation.png", polar_el, "Polar elevation — mean suppression"),
    ):
        angles = series.get("angles_deg") or []
        means = series.get("suppression_db_mean") or []
        if not angles:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(angles, means, marker="o", lw=1.2)
        ax.set_xlabel(f"{series.get('angle_key', 'angle')} (deg)")
        ax.set_ylabel("Mean suppression (dB)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        dest = agg_root / name
        fig.savefig(dest, dpi=120)
        plt.close(fig)
        written[name.replace(".png", "_png")] = str(dest)

    return written
