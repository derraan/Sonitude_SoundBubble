"""Host-agnostic measure trial orchestrator.

Supports ``acoustic_profile`` windows_pico | pi_hats:
  1) start selectable ONNX RT (cpp|python) with gains + --tap-dir
  2) play shared stimulus on profile stimulus device
  3) when hats_in_device is set, record HATS in parallel (start-of-take sync)
  4) stop / flush taps + hats_*.wav; write trial.json + telemetry.jsonl

Multi-interface sync is start-of-take (shared host timeline), not bit-clock lock.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.measure.hats import HatsCaptureConfig, HatsRecorder
from tools.measure.runtime import Gains, RuntimeLaunch, RuntimeProcess, sleep_ms
from tools.measure.stimulus import StimulusPlayer

try:
    from tools.tech_testbench.app.storage.trial_store import TrialStore, new_run_id, new_trial_id
except ImportError:  # pragma: no cover — allow running without UI package layout
    TrialStore = None  # type: ignore[misc, assignment]
    new_run_id = None  # type: ignore[assignment]
    new_trial_id = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEECH_DIR = REPO_ROOT / "datasets" / "LibriSpeech" / "dev-clean"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _auto_suppression_fr(trial_root: Path) -> list[str]:
    """Write suppression_fr plots/metrics when dry_ref + wet_out exist. Returns plot paths."""
    dry = trial_root / "dry_ref.wav"
    wet = trial_root / "wet_out.wav"
    if not dry.is_file() or not wet.is_file():
        return []
    try:
        from tools.tech_testbench.app.analysis.metrics import write_trial_analysis
    except ImportError:
        return []
    try:
        write_trial_analysis(trial_root, write_plots=True, merge_existing_metrics=True)
    except Exception:  # noqa: BLE001 — trial still valid without FR
        return []
    artifacts: list[str] = []
    for name in (
        "plots/suppression_fr.csv",
        "plots/suppression_fr.json",
        "plots/suppression_fr.png",
        "plots/hats_fr.csv",
        "plots/hats_fr.json",
        "plots/hats_fr.png",
    ):
        if (trial_root / name).is_file():
            artifacts.append(name)
    return artifacts


def _file_sha256(path: Path, max_bytes: int = 8 * 1024 * 1024) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            remaining = max_bytes
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
        return h.hexdigest()
    except OSError:
        return None


@dataclass
class TrialGeometry:
    distance_m: float = 1.5
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0


@dataclass
class MeasureConfig:
    acoustic_profile: str = "windows_pico"  # windows_pico | pi_hats
    host: str = "windows_testbench"  # windows_testbench | pi_rig
    measure_root: str = str(REPO_ROOT / "measure")
    run_id: str = ""
    kind: str = "algorithm"
    stimulus_path: str = ""
    stimulus_out_device: int | None = None
    hats_in_device: int | None = None
    hats_channels: int = 2
    hats_sr: int = 48000
    geometry: TrialGeometry = field(default_factory=TrialGeometry)
    runtime: RuntimeLaunch = field(default_factory=RuntimeLaunch)
    settle_ms: float = 250.0
    post_stimulus_ms: float = 200.0
    notes: str = ""
    campaign_kind: str = "controlled_ess"  # controlled_ess | in_context_speech
    motion_direction: str = "static"  # static | into_bubble | out_of_bubble
    fixed_source_m: float | None = None
    moving_source_m: float | None = None
    stimulus_repeat: int = 1
    stimulus_play_channels: str = "both"  # both | left | right


def _ensure_trial_dir(cfg: MeasureConfig) -> tuple[Path, Path, dict[str, Any]]:
    root = Path(cfg.measure_root)
    root.mkdir(parents=True, exist_ok=True)
    if TrialStore is not None:
        store = TrialStore(root)
        run_id = cfg.run_id or None
        if run_id is None:
            run_root = store.new_run(
                meta={"acoustic_profile": cfg.acoustic_profile, "host": cfg.host, "kind": cfg.kind}
            )
            run_id = run_root.name
        else:
            run_root = root / run_id
            if not run_root.is_dir():
                store.new_run(run_id=run_id, meta={"acoustic_profile": cfg.acoustic_profile, "host": cfg.host})
        paths = store.new_trial(
            run_id=run_id,
            kind=cfg.kind,
            trial_fields={
                "acoustic_profile": cfg.acoustic_profile,
                "host": cfg.host,
                "kind": cfg.kind,
            },
        )
        return paths.root, paths.run_root, {
            "run_id": paths.run_id,
            "trial_id": paths.trial_id,
            "store": store,
            "paths": paths,
        }

    # Minimal fallback without TrialStore
    run_id = cfg.run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_root = root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run.json").write_text(
        json.dumps({"run_id": run_id, "created_utc": _utc_now()}, indent=2), encoding="utf-8"
    )
    trial_id = f"{cfg.kind}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    trial_root = run_root / trial_id
    (trial_root / "plots").mkdir(parents=True, exist_ok=True)
    return trial_root, run_root, {"run_id": run_id, "trial_id": trial_id, "store": None, "paths": None}


def append_telemetry(run_root: Path, record: dict[str, Any]) -> Path:
    """Append one compact metrics line (never WAV/PCM payloads)."""
    path = run_root / "telemetry.jsonl"
    payload = dict(record)
    for banned in ("wav", "pcm", "audio", "samples"):
        payload.pop(banned, None)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    return path


def _make_hats_recorder(cfg: MeasureConfig) -> HatsRecorder | None:
    """Arm ear-rig capture whenever hats_in_device is set (any acoustic profile)."""
    if cfg.hats_in_device is None:
        return None
    # HATS is always opened/processed as stereo internally (see tools.measure.hats).
    cfg.hats_channels = 2
    return HatsRecorder(
        HatsCaptureConfig(
            device=cfg.hats_in_device,
            channels=2,
            sample_rate=int(cfg.hats_sr),
        )
    )


def _stimulus_ok(cfg: MeasureConfig) -> bool:
    """A WAV file, or (in_context_speech only) a folder of speech utterances."""
    p = Path(cfg.stimulus_path) if cfg.stimulus_path else None
    if p is None:
        return False
    return p.is_file() or (cfg.campaign_kind == "in_context_speech" and p.is_dir())


def _copy_stimulus(cfg: MeasureConfig, trial_root: Path) -> tuple[Path, Path | None]:
    """Copy operator stimulus into the trial as ``stimulus_used.wav`` (never move).

    in_context_speech + a folder: build a fresh random-utterance loop for this take
    (``stimulus_used.wav`` + ``stimulus_speech.json``) and play that.
    """
    stim_src = Path(cfg.stimulus_path)
    stim_copy = trial_root / "stimulus_used.wav"
    note = None
    if (
        cfg.campaign_kind == "in_context_speech"
        and stim_src.is_file()
        and "MeasSweep" in stim_src.name
        and DEFAULT_SPEECH_DIR.is_dir()
    ):
        # Safety net: a speech campaign must not silently play the ESS sweep.
        note = f"in_context_speech with ESS stimulus {stim_src.name}: used default speech folder"
        stim_src = DEFAULT_SPEECH_DIR
    if cfg.campaign_kind == "in_context_speech" and stim_src.is_dir():
        from tools.measure.stimulus import build_speech_loop

        meta = build_speech_loop(stim_src, stim_copy)
        if note:
            meta["note"] = note
        (trial_root / "stimulus_speech.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return stim_copy, stim_copy
    if stim_src.is_file():
        shutil.copy2(stim_src, stim_copy)
        return stim_src, stim_copy
    return stim_src, None


def _new_sync() -> dict[str, Any]:
    return {
        "mode": "start_of_take",
        "bit_clock_locked": False,
        "note": (
            "Independent USB clocks; WAVs start-aligned on host timeline; "
            "not sample-locked."
        ),
    }


def _base_trial_doc(
    cfg: MeasureConfig,
    meta: dict[str, Any],
    *,
    stim_src: Path,
    wav_artifacts: list[str],
    sync: dict[str, Any],
    player: StimulusPlayer,
    elapsed_s: float,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared trial.json fields for algorithm + raw_polar paths."""
    doc: dict[str, Any] = {
        "run_id": meta["run_id"],
        "trial_id": meta["trial_id"],
        "kind": cfg.kind,
        "created_utc": _utc_now(),
        "host": cfg.host,
        "acoustic_profile": cfg.acoustic_profile,
        "distance_m": cfg.geometry.distance_m,
        "azimuth_deg": cfg.geometry.azimuth_deg,
        "elevation_deg": cfg.geometry.elevation_deg,
        "campaign_kind": cfg.campaign_kind,
        "motion_direction": cfg.motion_direction,
        "fixed_source_m": cfg.fixed_source_m,
        "moving_source_m": (
            cfg.moving_source_m
            if cfg.moving_source_m is not None
            else cfg.geometry.distance_m
        ),
        "stimulus_path": str(stim_src),
        "stimulus_sha256_prefix": _file_sha256(stim_src),
        "stimulus_out_device": cfg.stimulus_out_device,
        "stimulus_repeat": int(cfg.stimulus_repeat or 1),
        "stimulus_play_channels": str(cfg.stimulus_play_channels or "both"),
        "hats_in_device": cfg.hats_in_device,
        "notes": cfg.notes,
        "elapsed_s": elapsed_s,
        "artifacts": {"wavs": list(wav_artifacts), "plots": [], "files": []},
        "stimulus": player.as_dict(),
        "sync": sync,
    }
    speech_meta = Path(stim_src).parent / "stimulus_speech.json"
    if Path(stim_src).name == "stimulus_used.wav" and speech_meta.is_file():
        doc["stimulus_speech"] = json.loads(speech_meta.read_text(encoding="utf-8"))
        doc["artifacts"]["files"].append("stimulus_speech.json")
    if error:
        doc["error"] = error
    if extra:
        doc.update(extra)
    return doc


def _union_artifacts(
    existing: dict[str, Any] | None,
    overlay: dict[str, Any] | None,
) -> dict[str, Any]:
    """Union wavs/plots/files; prefer TrialStore.merge_artifacts when available."""
    if TrialStore is not None:
        return TrialStore.merge_artifacts(existing, overlay)
    out: dict[str, Any] = dict(existing or {})
    over = dict(overlay or {})
    for bucket in ("wavs", "plots", "files"):
        merged: list[str] = []
        for src in (out.get(bucket), over.get(bucket)):
            for item in list(src or []):
                s = str(item)
                if s not in merged:
                    merged.append(s)
        if merged or bucket in out or bucket in over:
            out[bucket] = merged
    for k, v in over.items():
        if k in ("wavs", "plots", "files"):
            continue
        out[k] = v
    return out


def _finalize_trial(
    *,
    trial_root: Path,
    run_root: Path,
    meta: dict[str, Any],
    trial_doc: dict[str, Any],
    metrics: dict[str, Any],
    wav_artifacts: list[str],
    telemetry: dict[str, Any] | None = None,
    file_artifacts: list[str] | None = None,
) -> dict[str, Any]:
    """Write trial.json + metrics, append telemetry, deep-merge artifacts into store.

    Loads any on-disk ``trial.json`` **before** writing so analysis-appended
    plots (mic_delay / timing_ref / FR) are unioned rather than clobbered.
    """
    arts = dict(trial_doc.get("artifacts") or {})
    arts["wavs"] = list(dict.fromkeys(list(arts.get("wavs") or []) + list(wav_artifacts)))
    if file_artifacts:
        files = list(arts.get("files") or [])
        for name in file_artifacts:
            if name not in files:
                files.append(name)
        arts["files"] = files
    trial_doc["artifacts"] = arts

    # Prefer store/on-disk artifacts written by write_trial_analysis earlier.
    on_disk: dict[str, Any] = {}
    store = meta.get("store")
    paths = meta.get("paths")
    if store is not None and paths is not None:
        try:
            on_disk = store.load_trial_json(paths)
        except (OSError, json.JSONDecodeError):
            on_disk = {}
    elif (trial_root / "trial.json").is_file():
        try:
            on_disk = json.loads((trial_root / "trial.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            on_disk = {}

    merged_doc = {**on_disk, **trial_doc}
    merged_doc["artifacts"] = _union_artifacts(
        on_disk.get("artifacts"),
        trial_doc.get("artifacts"),
    )
    trial_doc = merged_doc

    (trial_root / "trial.json").write_text(json.dumps(trial_doc, indent=2), encoding="utf-8")
    (trial_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if telemetry is not None:
        append_telemetry(run_root, telemetry)

    if store is not None and paths is not None:
        if hasattr(store, "register_wavs"):
            store.register_wavs(paths, wav_artifacts)
        else:
            for name in wav_artifacts:
                if (trial_root / name).is_file():
                    store.register_wav(paths, name)
        for name in file_artifacts or []:
            if (trial_root / name).is_file():
                try:
                    store.register_file(paths, name, None)
                except Exception:
                    pass
        # Re-union after register_wavs (which may reload/save) so plots stay.
        latest = store.load_trial_json(paths)
        final_doc = {**latest, **trial_doc}
        final_doc["artifacts"] = _union_artifacts(
            latest.get("artifacts"),
            trial_doc.get("artifacts"),
        )
        store.save_trial_json(paths, final_doc)
        try:
            merged_metrics = json.loads((trial_root / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            merged_metrics = metrics
        store.save_metrics(paths, merged_metrics)
        trial_doc = final_doc
    return trial_doc


def run_algorithm_trial(
    cfg: MeasureConfig,
    *,
    pipeline_session: Any | None = None,
) -> dict[str, Any]:
    """Execute one algorithm / suppressed trial; returns trial.json dict.

    When ``pipeline_session`` is a live Ready session, retarget taps for this
    take instead of spawning/stopping soundbubble-rt.
    """
    if cfg.acoustic_profile not in ("windows_pico", "pi_hats"):
        raise ValueError(f"unknown acoustic_profile: {cfg.acoustic_profile}")
    if not cfg.stimulus_path:
        raise ValueError("stimulus_path is required")
    if not _stimulus_ok(cfg):
        raise FileNotFoundError(f"stimulus WAV (or speech folder) not found: {cfg.stimulus_path}")
    if cfg.acoustic_profile == "pi_hats" and cfg.hats_in_device is None:
        raise ValueError("pi_hats requires hats_in_device")
    if cfg.stimulus_out_device is None:
        raise ValueError("stimulus_out_device is required")
    if cfg.runtime.pipeline_in_device is None or cfg.runtime.pipeline_out_device is None:
        raise ValueError("pipeline_in_device and pipeline_out_device are required")
    if (
        cfg.stimulus_out_device is not None
        and cfg.runtime.pipeline_out_device is not None
        and int(cfg.stimulus_out_device) == int(cfg.runtime.pipeline_out_device)
    ):
        raise ValueError(
            "stimulus_out_device == pipeline_out_device: PortAudio exclusive open will conflict. "
            "For windows_pico use a separate stimulus speaker; for pi_hats route speakers vs "
            "Pico DAC to different devices."
        )

    live = pipeline_session is not None
    if live and not getattr(pipeline_session, "is_ready", False):
        raise RuntimeError(
            "Global pipeline is not Ready — use Start Pipeline and wait before taking."
        )

    trial_root, run_root, meta = _ensure_trial_dir(cfg)
    taps_dir = trial_root  # taps land beside trial.json
    cfg.runtime.tap_dir = str(taps_dir)
    if not cfg.runtime.tap_select:
        cfg.runtime.tap_select = "raw,mapped,dry,wet"

    stim_src, stim_copy = _copy_stimulus(cfg, trial_root)
    player = StimulusPlayer(
        str(stim_src),
        device=cfg.stimulus_out_device,
        repeat=int(cfg.stimulus_repeat or 1),
        play_channels=str(cfg.stimulus_play_channels or "both"),
    )
    hats = _make_hats_recorder(cfg)

    pipeline_log = trial_root / "pipeline.log"
    rt: RuntimeProcess | None = None if live else RuntimeProcess(cfg.runtime, log_path=pipeline_log)
    wav_artifacts: list[str] = ["stimulus_used.wav"] if stim_copy is not None else []
    sync = _new_sync()
    error: str | None = None
    rc = 0 if live else -1
    try:
        # Order: arm HATS → (start or retarget) ONNX RT → settle → play stimulus → stop/flush.
        if hats is not None:
            sync["hats_armed_perf"] = time.perf_counter()
            hats.start()
        sync["pipeline_started_perf"] = time.perf_counter()
        if live:
            pipeline_session.tap_begin(trial_root, select=cfg.runtime.tap_select)
            pipeline_log.write_text(
                f"[session] live take using global pipeline\n"
                f"[session] tap_dir={trial_root}\n",
                encoding="utf-8",
            )
        else:
            assert rt is not None
            rt.start()
        sleep_ms(cfg.settle_ms)
        if not live:
            assert rt is not None
            if rt.poll() is not None:
                raise RuntimeError(f"pipeline exited early rc={rt.poll()}")
        if hats is not None and not hats.is_started:
            raise RuntimeError("hats_in_device set but HATS stream failed to start before stimulus")
        sync["t0_perf"] = time.perf_counter()
        sync["stimulus_started_perf"] = sync["t0_perf"]
        player.start_background()
        player.join(timeout=player.info.duration_s + 30.0)
        sleep_ms(cfg.post_stimulus_ms)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        try:
            player.stop()
        except Exception:
            pass
    finally:
        if live:
            try:
                pipeline_session.tap_end(trial_dir=trial_root)
            except Exception as flush_exc:  # noqa: BLE001
                if error is None:
                    error = f"tap flush failed: {flush_exc}"
            try:
                pipeline_session.append_profile_to_trial_log(pipeline_log)
            except Exception:
                pass
            rc = 0
        else:
            assert rt is not None
            rc = rt.stop(grace_s=4.0)
        sync["pipeline_stopped_perf"] = time.perf_counter()
        if hats is not None:
            audio = hats.stop()
            sync["hats_stopped_perf"] = time.perf_counter()
            for p in hats.write_wavs(trial_root, audio):
                if p.name not in wav_artifacts:
                    wav_artifacts.append(p.name)

    # Register tap WAVs if present (ignore header-only stubs from a raced flush).
    tap_names = ("raw_8ch.wav", "mapped_6ch.wav", "dry_ref.wav", "wet_out.wav")
    found_taps = 0
    for name in tap_names:
        p = trial_root / name
        if p.is_file() and p.stat().st_size > 1024:
            found_taps += 1
            if name not in wav_artifacts:
                wav_artifacts.append(name)
    # Late flush: give disk a moment if the take ran live and files are still landing.
    if live and found_taps == 0:
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline and found_taps == 0:
            time.sleep(0.05)
            found_taps = 0
            for name in tap_names:
                p = trial_root / name
                if p.is_file() and p.stat().st_size > 1024:
                    found_taps += 1
                    if name not in wav_artifacts:
                        wav_artifacts.append(name)
    taps_missing = found_taps == 0
    tap_truncated = taps_missing  # back-compat key (historical name)
    if cfg.hats_in_device is not None:
        hats_ok = any((trial_root / n).is_file() for n in ("hats_l.wav", "hats_r.wav", "hats_stereo.wav"))
        if not hats_ok and error is None:
            error = "hats_in_device set but missing hats_*.wav"

    elapsed_s = float(sync.get("pipeline_stopped_perf") or 0.0) - float(
        sync.get("pipeline_started_perf") or 0.0
    )
    file_artifacts: list[str] = []
    if pipeline_log.is_file():
        file_artifacts.append("pipeline.log")

    # Live takes run on the already-started global pipeline: record what it was
    # actually launched with (Setup edits do not reach a running pipeline).
    eff = cfg.runtime
    settings_mismatch: dict[str, Any] = {}
    if live:
        run_launch = getattr(pipeline_session, "launch", None)
        if run_launch is not None:
            eff = run_launch
            for key, a, b in (
                ("gains", cfg.runtime.gains.to_dict(), run_launch.gains.to_dict()),
                ("model", cfg.runtime.model_path, run_launch.model_path),
                ("contract", cfg.runtime.contract_path, run_launch.contract_path),
                ("channel_map", cfg.runtime.channel_map, run_launch.channel_map),
                ("bubble_radius_m", cfg.runtime.bubble_radius_m, run_launch.bubble_radius_m),
            ):
                if a != b:
                    settings_mismatch[key] = {"setup": a, "running_pipeline": b}
            if settings_mismatch:
                with pipeline_log.open("a", encoding="utf-8") as fh:
                    fh.write(
                        "[session] WARNING: Setup differs from the RUNNING pipeline for "
                        + ", ".join(settings_mismatch)
                        + " -> trial records the running values; restart the pipeline to apply Setup.\n"
                    )
    trial_doc = _base_trial_doc(
        cfg,
        meta,
        stim_src=stim_src,
        wav_artifacts=wav_artifacts,
        sync=sync,
        player=player,
        elapsed_s=elapsed_s,
        error=error,
        extra={
            "runtime": eff.runtime,
            "binary_or_entry": eff.as_dict().get("binary_or_entry"),
            "pipeline_device": {
                "in": eff.pipeline_in_device,
                "out": eff.pipeline_out_device,
            },
            "model": eff.model_path,
            "contract": eff.contract_path,
            "zoo_dir": eff.zoo_dir if eff.use_zoo else "",
            "bubble_radius_m": eff.bubble_radius_m,
            "gains": eff.gains.to_dict(),
            **({"settings_mismatch": settings_mismatch} if settings_mismatch else {}),
            "sr": {"input": eff.input_sr, "output": eff.output_sr},
            "channel_map": eff.channel_map,
            "tap_select": cfg.runtime.tap_select,
            "tap_caps": {
                "max_seconds": cfg.runtime.tap_max_seconds,
                "max_mb": cfg.runtime.tap_max_mb,
            },
            "tap_truncated": tap_truncated,
            "taps_missing": taps_missing,
            "runtime_exit_code": rc,
            "live_pipeline": bool(live),
            "pipeline_log": "pipeline.log" if pipeline_log.is_file() else None,
        },
    )
    if hats is not None:
        trial_doc["hats"] = hats.as_dict()

    metrics = {
        "elapsed_s": elapsed_s,
        "runtime_exit_code": rc,
        "error": error,
        "sync": sync,
        "tap_truncated": tap_truncated,
        "taps_missing": taps_missing,
    }
    # Write early so analysis can merge into metrics/plots on disk.
    (trial_root / "trial.json").write_text(json.dumps(trial_doc, indent=2), encoding="utf-8")
    (trial_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Auto wet / suppression FR (+ mic_delay / timing_ref via write_trial_analysis).
    fr_artifacts = _auto_suppression_fr(trial_root)
    if fr_artifacts:
        plots = list(trial_doc["artifacts"].get("plots") or [])
        for rel in fr_artifacts:
            if rel not in plots:
                plots.append(rel)
        trial_doc["artifacts"]["plots"] = plots

    # Reload plots appended by write_trial_analysis (mic_delay / timing_ref / FR).
    try:
        on_disk = json.loads((trial_root / "trial.json").read_text(encoding="utf-8"))
        trial_doc["artifacts"] = _union_artifacts(
            on_disk.get("artifacts"),
            trial_doc.get("artifacts"),
        )
    except (OSError, json.JSONDecodeError):
        pass
    try:
        metrics = json.loads((trial_root / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass

    return _finalize_trial(
        trial_root=trial_root,
        run_root=run_root,
        meta=meta,
        trial_doc=trial_doc,
        metrics=metrics,
        wav_artifacts=wav_artifacts,
        file_artifacts=file_artifacts,
        telemetry={
            "ts": _utc_now(),
            "host": cfg.host,
            "acoustic_profile": cfg.acoustic_profile,
            "runtime": cfg.runtime.runtime,
            "distance_m": cfg.geometry.distance_m,
            "azimuth_deg": cfg.geometry.azimuth_deg,
            "elevation_deg": cfg.geometry.elevation_deg,
            "campaign_kind": cfg.campaign_kind,
            "motion_direction": cfg.motion_direction,
            "fixed_source_m": cfg.fixed_source_m,
            "moving_source_m": (
                cfg.moving_source_m
                if cfg.moving_source_m is not None
                else cfg.geometry.distance_m
            ),
            "trial_path": str(trial_root),
            "stimulus_path": str(stim_src),
            "gains": cfg.runtime.gains.to_dict(),
            "tap_truncated": tap_truncated,
            "taps_missing": taps_missing,
            "runtime_exit_code": rc,
            "error": error,
            "suppression_fr": bool(fr_artifacts),
        },
    )


def config_from_session_dict(d: dict[str, Any]) -> MeasureConfig:
    """Build MeasureConfig from tech_testbench SessionSettings.to_trial_fields()."""
    gains_d = d.get("gains") or {}
    gains = Gains(
        input_gain_db=float(gains_d.get("input_gain_db", 0.0)),
        output_gain_db=float(gains_d.get("output_gain_db", 0.0)),
        mix_dry=float(gains_d.get("mix_dry", 0.0)),
        output_limiter=bool(gains_d.get("output_limiter", True)),
        limiter_release_ms=float(gains_d.get("limiter_release_ms", 25.0)),
        mic_gains=[float(v) for v in (gains_d.get("mic_gains") or [])],
    )
    host_target = str(d.get("host_target") or "local_windows")
    host = "pi_rig" if host_target == "pi_rig" else "windows_testbench"
    raw_bubble = d.get("bubble_radius_m")
    if raw_bubble is None or raw_bubble == "":
        bubble_radius_m: float | None = 1.5
    else:
        bubble_radius_m = float(raw_bubble)
    rt = RuntimeLaunch(
        runtime=str(d.get("runtime") or "cpp"),
        pipeline_in_device=d.get("pipeline_in_device"),
        pipeline_out_device=d.get("pipeline_out_device"),
        input_sr=int(d.get("input_sr") or 24000),
        output_sr=int(d.get("output_sr") or 24000),
        channel_map=str(d.get("channel_map") or ""),
        gains=gains,
        model_path=str(d.get("model_path") or ""),
        contract_path=str(d.get("contract_path") or ""),
        zoo_dir=str(d.get("zoo_dir") or ""),
        bubble_radius_m=bubble_radius_m,
        use_zoo=bool(d.get("use_zoo", bool(d.get("zoo_dir")))),
        soundbubble_rt_path=str(d.get("soundbubble_rt_path") or ""),
    )
    fixed_raw = d.get("fixed_source_m")
    moving_raw = d.get("moving_source_m")
    distance_m = float(d.get("distance_m") or 1.5)
    return MeasureConfig(
        acoustic_profile=str(d.get("acoustic_profile") or "windows_pico"),
        host=host,
        measure_root=str(d.get("measure_root") or (REPO_ROOT / "measure")),
        run_id=str(d.get("active_run_id") or ""),
        kind=str(d.get("kind") or "algorithm"),
        stimulus_path=str(d.get("stimulus_path") or ""),
        stimulus_out_device=d.get("stimulus_out_device"),
        hats_in_device=d.get("hats_in_device"),
        geometry=TrialGeometry(
            distance_m=distance_m,
            azimuth_deg=float(d.get("azimuth_deg") or 0.0),
            elevation_deg=float(d.get("elevation_deg") or 0.0),
        ),
        runtime=rt,
        notes=str(d.get("notes") or ""),
        campaign_kind=str(d.get("campaign_kind") or "controlled_ess"),
        motion_direction=str(d.get("motion_direction") or "static"),
        fixed_source_m=None if fixed_raw in (None, "") else float(fixed_raw),
        moving_source_m=(
            distance_m if moving_raw in (None, "") else float(moving_raw)
        ),
        stimulus_repeat=max(1, int(d.get("stimulus_repeat") or 1)),
        stimulus_play_channels=str(d.get("stimulus_play_channels") or "both"),
    )


def run_raw_polar_trial(cfg: MeasureConfig) -> dict[str, Any]:
    """Hardware-only: play stimulus while recording Pico mics (+ HATS when configured)."""
    import numpy as np

    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("sounddevice required for raw polar capture") from exc
    try:
        from scipy.io.wavfile import write as wavwrite
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy required to write raw polar WAV") from exc

    if cfg.stimulus_out_device is None:
        raise ValueError("stimulus_out_device is required")
    if cfg.runtime.pipeline_in_device is None:
        raise ValueError("pipeline_in_device (Pico mics) is required")
    if not _stimulus_ok(cfg):
        raise FileNotFoundError(f"stimulus WAV (or speech folder) not found: {cfg.stimulus_path}")

    cfg.kind = "raw_polar"
    trial_root, run_root, meta = _ensure_trial_dir(cfg)
    stim_src, stim_copy = _copy_stimulus(cfg, trial_root)
    if stim_copy is None:
        raise FileNotFoundError(f"stimulus WAV not found: {cfg.stimulus_path}")

    player = StimulusPlayer(
        str(stim_src),
        device=cfg.stimulus_out_device,
        repeat=int(cfg.stimulus_repeat or 1),
        play_channels=str(cfg.stimulus_play_channels or "both"),
    )
    hats = _make_hats_recorder(cfg)
    cap_sr = int(cfg.runtime.input_sr)
    ch_in = int(cfg.runtime.input_channels)
    chunks: list[np.ndarray] = []
    error: str | None = None
    sync = _new_sync()

    def _cb(indata, _frames, _time_info, status) -> None:  # noqa: ANN001
        _ = status
        chunks.append(np.asarray(indata, dtype=np.float32).copy())

    stream = sd.InputStream(
        samplerate=cap_sr,
        channels=ch_in,
        device=cfg.runtime.pipeline_in_device,
        dtype="float32",
        callback=_cb,
    )
    try:
        # Order: arm HATS → Pico mic stream → settle → stimulus → stop together.
        if hats is not None:
            sync["hats_armed_perf"] = time.perf_counter()
            hats.start()
        sync["pipeline_started_perf"] = time.perf_counter()
        stream.start()
        sleep_ms(cfg.settle_ms if cfg.settle_ms > 0 else 50.0)
        if hats is not None and not hats.is_started:
            raise RuntimeError("hats_in_device set but HATS stream failed to start before stimulus")
        sync["t0_perf"] = time.perf_counter()
        sync["stimulus_started_perf"] = sync["t0_perf"]
        player.play_blocking()
        sleep_ms(cfg.post_stimulus_ms)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        try:
            player.stop()
        except Exception:
            pass
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        sync["pipeline_stopped_perf"] = time.perf_counter()
        if hats is not None:
            hats_audio = hats.stop()
            sync["hats_stopped_perf"] = time.perf_counter()
            hats.write_wavs(trial_root, hats_audio)

    audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, ch_in), dtype=np.float32)
    out = trial_root / "raw_polar_8ch.wav"
    wavwrite(str(out), cap_sr, audio.astype(np.float32))
    elapsed_s = float(sync.get("pipeline_stopped_perf") or 0.0) - float(
        sync.get("pipeline_started_perf") or 0.0
    )
    wav_artifacts = ["stimulus_used.wav", "raw_polar_8ch.wav"]
    if hats is not None:
        for name in ("hats_stereo.wav", "hats_l.wav", "hats_r.wav"):
            if (trial_root / name).is_file() and name not in wav_artifacts:
                wav_artifacts.append(name)
        if not any((trial_root / n).is_file() for n in ("hats_l.wav", "hats_r.wav", "hats_stereo.wav")):
            if error is None:
                error = "hats_in_device set but missing hats_*.wav"

    trial_doc = _base_trial_doc(
        cfg,
        meta,
        stim_src=stim_src,
        wav_artifacts=wav_artifacts,
        sync=sync,
        player=player,
        elapsed_s=elapsed_s,
        error=error,
        extra={
            "pipeline_in_device": cfg.runtime.pipeline_in_device,
            "input_sr": cap_sr,
            "notes": cfg.notes or "hardware-only; no ONNX",
            "capture_frames": int(audio.shape[0]),
        },
    )
    if hats is not None:
        trial_doc["hats"] = hats.as_dict()
    metrics = {
        "elapsed_s": elapsed_s,
        "frames": int(audio.shape[0]),
        "channels": int(audio.shape[1]) if audio.ndim == 2 else 0,
        "sample_rate": cap_sr,
        "error": error,
        "azimuth_deg": cfg.geometry.azimuth_deg,
        "elevation_deg": cfg.geometry.elevation_deg,
        "sync": sync,
    }

    trial_doc = _finalize_trial(
        trial_root=trial_root,
        run_root=run_root,
        meta=meta,
        trial_doc=trial_doc,
        metrics=metrics,
        wav_artifacts=wav_artifacts,
        telemetry={
            "ts": _utc_now(),
            "kind": "raw_polar",
            "host": cfg.host,
            "acoustic_profile": cfg.acoustic_profile,
            "azimuth_deg": cfg.geometry.azimuth_deg,
            "elevation_deg": cfg.geometry.elevation_deg,
            "trial_path": str(trial_root),
            "error": error,
        },
    )

    store = meta.get("store")
    paths = meta.get("paths")
    if store is not None and paths is not None:
        series = {
            "type": "polar",
            "azimuth_deg": cfg.geometry.azimuth_deg,
            "elevation_deg": cfg.geometry.elevation_deg,
            "frames": metrics["frames"],
            "sample_rate": cap_sr,
            "note": "deconv/FR offline; capture WAV on disk",
        }
        csv_path = paths.plot_path("polar_capture.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(
            "azimuth_deg,elevation_deg,frames,sample_rate\n"
            f"{cfg.geometry.azimuth_deg},{cfg.geometry.elevation_deg},"
            f"{metrics['frames']},{cap_sr}\n",
            encoding="utf-8",
        )
        store.register_plot(
            paths,
            stem="polar_capture",
            series_csv=csv_path,
            series_json=series,
        )
    # Fullband levels / FR + relative mic delay (Part A / hardware polar).
    try:
        from tools.tech_testbench.app.analysis.metrics import write_trial_analysis

        merged = write_trial_analysis(trial_root, write_plots=True, merge_existing_metrics=True)
        metrics = {**metrics, **merged}
        try:
            trial_doc = json.loads((trial_root / "trial.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        if store is not None and paths is not None:
            # Preserve analysis-appended plots when refreshing store docs.
            on_disk = store.load_trial_json(paths)
            merged_doc = {**on_disk, **trial_doc}
            merged_doc["artifacts"] = _union_artifacts(
                on_disk.get("artifacts"),
                trial_doc.get("artifacts"),
            )
            store.save_trial_json(paths, merged_doc)
            store.save_metrics(paths, metrics)
            trial_doc = merged_doc
    except Exception:
        try:
            from tools.tech_testbench.app.analysis.mic_delay import analyze_mic_delay_trial

            delay = analyze_mic_delay_trial(trial_root, write_plots=True)
            if delay:
                metrics["relative_delay_ms"] = delay.get("relative_delay_ms")
                metrics["ref_channel"] = delay.get("ref_channel")
                metrics["mic_delay"] = delay
                trial_doc["relative_delay_ms"] = delay.get("relative_delay_ms")
                trial_doc["ref_channel"] = delay.get("ref_channel")
                (trial_root / "trial.json").write_text(
                    json.dumps(trial_doc, indent=2), encoding="utf-8"
                )
                (trial_root / "metrics.json").write_text(
                    json.dumps(metrics, indent=2), encoding="utf-8"
                )
                if store is not None and paths is not None:
                    on_disk = store.load_trial_json(paths)
                    merged_doc = {**on_disk, **trial_doc}
                    merged_doc["artifacts"] = _union_artifacts(
                        on_disk.get("artifacts"),
                        trial_doc.get("artifacts"),
                    )
                    store.save_trial_json(paths, merged_doc)
                    store.save_metrics(paths, metrics)
                    trial_doc = merged_doc
        except Exception:
            pass
    return trial_doc
