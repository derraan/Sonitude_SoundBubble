"""Persistent trial storage: WAVs, plots, metrics, trial.json — reloadable later.

Layout (under measure_root):
  <run_id>/
    run.json
    <trial_id>/
      trial.json
      metrics.json
      plots/
        *.png / *.svg / *.csv / *.json   # figures + series data for re-plot
      *.wav                             # taps, HATS, levels, etc.
      stimulus_used.wav                 # optional copy of stimulus
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def new_run_id() -> str:
    return f"run_{_utc_stamp()}_{uuid.uuid4().hex[:6]}"


def new_trial_id(kind: str = "trial") -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in kind)[:32] or "trial"
    return f"{safe}_{_utc_stamp()}_{uuid.uuid4().hex[:4]}"


@dataclass(frozen=True)
class TrialPaths:
    run_id: str
    trial_id: str
    run_root: Path
    root: Path
    trial_json: Path
    metrics_json: Path
    plots_dir: Path
    wav_dir: Path  # same as root; WAVs live beside trial.json

    def wav_path(self, name: str) -> Path:
        if not name.endswith(".wav"):
            name = f"{name}.wav"
        return self.root / name

    def plot_path(self, name: str) -> Path:
        return self.plots_dir / name


class TrialStore:
    """Creates and lists durable measure runs/trials for the tech testbench."""

    def __init__(self, measure_root: str | Path) -> None:
        self.measure_root = Path(measure_root)
        self.measure_root.mkdir(parents=True, exist_ok=True)

    def new_run(self, run_id: str | None = None, *, meta: dict[str, Any] | None = None) -> Path:
        run_id = run_id or new_run_id()
        run_root = self.measure_root / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **(meta or {}),
        }
        (run_root / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return run_root

    def new_trial(
        self,
        *,
        run_id: str | None = None,
        kind: str = "trial",
        trial_fields: dict[str, Any] | None = None,
    ) -> TrialPaths:
        if run_id is None:
            run_root = self.new_run()
            run_id = run_root.name
        else:
            run_root = self.measure_root / run_id
            if not run_root.is_dir():
                self.new_run(run_id=run_id)

        trial_id = new_trial_id(kind)
        root = run_root / trial_id
        plots = root / "plots"
        root.mkdir(parents=True, exist_ok=True)
        plots.mkdir(parents=True, exist_ok=True)

        paths = TrialPaths(
            run_id=run_id,
            trial_id=trial_id,
            run_root=run_root,
            root=root,
            trial_json=root / "trial.json",
            metrics_json=root / "metrics.json",
            plots_dir=plots,
            wav_dir=root,
        )
        doc = {
            "run_id": run_id,
            "trial_id": trial_id,
            "kind": kind,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": {"wavs": [], "plots": [], "files": []},
            **(trial_fields or {}),
        }
        # Ensure kind from constructor wins over a stale trial_fields copy
        doc["kind"] = kind
        self.save_trial_json(paths, doc)
        self.save_metrics(paths, {})
        return paths

    @staticmethod
    def save_trial_json(paths: TrialPaths, doc: dict[str, Any]) -> None:
        paths.trial_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    @staticmethod
    def load_trial_json(paths: TrialPaths | Path) -> dict[str, Any]:
        path = paths.trial_json if isinstance(paths, TrialPaths) else Path(paths)
        if path.is_dir():
            path = path / "trial.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def save_metrics(paths: TrialPaths, metrics: dict[str, Any]) -> None:
        paths.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    @staticmethod
    def load_metrics(paths: TrialPaths | Path) -> dict[str, Any]:
        path = paths.metrics_json if isinstance(paths, TrialPaths) else Path(paths)
        if path.is_dir():
            path = path / "metrics.json"
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def copy_wav(source: str | Path, dest: str | Path) -> Path:
        """Copy (never move) a capture into the trial folder."""
        source = Path(source).resolve()
        dest = Path(dest).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source == dest:
            return dest
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(str(source), str(dest))
        return dest

    def _append_artifact(self, paths: TrialPaths, bucket: str, rel: str) -> None:
        doc = self.load_trial_json(paths)
        arts = doc.setdefault("artifacts", {})
        items: list[str] = list(arts.get(bucket) or [])
        if rel not in items:
            items.append(rel)
        arts[bucket] = items
        self.save_trial_json(paths, doc)

    def register_wav(self, paths: TrialPaths, name: str, source: str | Path | None = None) -> Path:
        """Register a WAV already in the trial folder, or copy from ``source`` first."""
        dest = paths.wav_path(name)
        if source is not None:
            self.copy_wav(source, dest)
        elif not dest.is_file():
            raise FileNotFoundError(f"WAV not found and no source given: {dest}")
        self._append_artifact(paths, "wavs", dest.name)
        return dest

    def register_wavs(self, paths: TrialPaths, names: list[str] | tuple[str, ...]) -> list[Path]:
        """Batch-register WAVs already in the trial folder (one load-modify-save)."""
        written: list[Path] = []
        doc = self.load_trial_json(paths)
        arts = doc.setdefault("artifacts", {})
        items: list[str] = list(arts.get("wavs") or [])
        for name in names:
            dest = paths.wav_path(name)
            if not dest.is_file():
                continue
            if dest.name not in items:
                items.append(dest.name)
            written.append(dest)
        arts["wavs"] = items
        self.save_trial_json(paths, doc)
        return written

    @staticmethod
    def merge_artifacts(
        base: dict[str, Any] | None,
        overlay: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Union ``wavs`` / ``plots`` / ``files`` lists; overlay non-list keys win."""
        out: dict[str, Any] = dict(base or {})
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

    def register_file(
        self,
        paths: TrialPaths,
        name: str,
        source: str | Path | None = None,
    ) -> Path:
        """Copy/register a non-WAV artifact (NDJSON, notes, exports) beside trial.json."""
        dest = paths.root / name
        if source is not None:
            src = Path(source).resolve()
            dest = dest.resolve()
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src != dest:
                if not src.is_file():
                    raise FileNotFoundError(src)
                shutil.copy2(str(src), str(dest))
        elif not dest.is_file():
            raise FileNotFoundError(f"File not found and no source given: {dest}")
        self._append_artifact(paths, "files", dest.name)
        return dest

    def copy_stimulus(self, paths: TrialPaths, stimulus_path: str | Path | None) -> Path | None:
        """Copy Setup stimulus into the trial as stimulus_used.wav when the source exists."""
        if not stimulus_path:
            return None
        src = Path(stimulus_path)
        if not src.is_file():
            return None
        dest = self.register_wav(paths, "stimulus_used.wav", src)
        doc = self.load_trial_json(paths)
        doc["stimulus_used"] = dest.name
        self.save_trial_json(paths, doc)
        return dest

    def scan_wavs(self, paths: TrialPaths) -> list[Path]:
        """Index any *.wav already sitting in the trial folder (idempotent)."""
        found: list[Path] = []
        if not paths.root.is_dir():
            return found
        for wav in sorted(paths.root.glob("*.wav")):
            self.register_wav(paths, wav.name, None)
            found.append(wav)
        return found

    def register_plot(
        self,
        paths: TrialPaths,
        *,
        stem: str,
        image: str | Path | None = None,
        series_csv: str | Path | None = None,
        series_json: dict[str, Any] | None = None,
    ) -> list[Path]:
        """Save plot image and/or series data under plots/; update trial.json artifacts."""
        written: list[Path] = []
        plots_meta: list[str] = []
        if image is not None:
            src = Path(image)
            dest = paths.plot_path(src.name if src.suffix else f"{stem}.png")
            if src.is_file():
                resolved_src = src.resolve()
                resolved_dest = dest.resolve()
                if resolved_src != resolved_dest:
                    shutil.copy2(src, dest)
            written.append(dest)
            plots_meta.append(str(dest.relative_to(paths.root)).replace("\\", "/"))
        if series_csv is not None:
            src = Path(series_csv)
            dest = paths.plot_path(src.name if src.suffix else f"{stem}.csv")
            if src.is_file():
                resolved_src = src.resolve()
                resolved_dest = dest.resolve()
                if resolved_src != resolved_dest:
                    shutil.copy2(src, dest)
            written.append(dest)
            plots_meta.append(str(dest.relative_to(paths.root)).replace("\\", "/"))
        if series_json is not None:
            dest = paths.plot_path(f"{stem}.json")
            dest.write_text(json.dumps(series_json, indent=2), encoding="utf-8")
            written.append(dest)
            plots_meta.append(str(dest.relative_to(paths.root)).replace("\\", "/"))

        doc = self.load_trial_json(paths)
        arts = doc.setdefault("artifacts", {})
        existing: list[str] = list(arts.get("plots") or [])
        for rel in plots_meta:
            if rel not in existing:
                existing.append(rel)
        arts["plots"] = existing
        self.save_trial_json(paths, doc)
        return written

    def list_runs(self) -> list[str]:
        if not self.measure_root.is_dir():
            return []
        return sorted(
            (p.name for p in self.measure_root.iterdir() if p.is_dir() and not p.name.startswith(".")),
            reverse=True,
        )

    def list_trials(self, run_id: str) -> list[str]:
        run_root = self.measure_root / run_id
        if not run_root.is_dir():
            return []
        out: list[str] = []
        for p in sorted(run_root.iterdir(), reverse=True):
            if p.is_dir() and (p / "trial.json").is_file():
                out.append(p.name)
        return out

    def trial_paths(self, run_id: str, trial_id: str) -> TrialPaths:
        root = self.measure_root / run_id / trial_id
        return TrialPaths(
            run_id=run_id,
            trial_id=trial_id,
            run_root=self.measure_root / run_id,
            root=root,
            trial_json=root / "trial.json",
            metrics_json=root / "metrics.json",
            plots_dir=root / "plots",
            wav_dir=root,
        )

    def list_all_trials(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Newest-first (run_id, trial_id, trial.json summary)."""
        rows: list[tuple[str, str, dict[str, Any]]] = []
        for run_id in self.list_runs():
            for trial_id in self.list_trials(run_id):
                try:
                    doc = self.load_trial_json(self.measure_root / run_id / trial_id / "trial.json")
                except (OSError, json.JSONDecodeError):
                    doc = {}
                rows.append((run_id, trial_id, doc))
        rows.sort(key=lambda r: str(r[2].get("created_utc") or ""), reverse=True)
        return rows
