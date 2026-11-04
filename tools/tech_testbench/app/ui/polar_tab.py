"""Polar az/el characterization — optional ONNX; mics + dry/wet + HATS + stimulus."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.session import SessionSettings
from app.storage.trial_store import TrialStore
from app.ui.pose_queue import Pose, PoseQueuePanel

# Back-compat alias for imports that still name RawPolarTab.
RawPolarTab = None  # set after PolarTab defined


class _PolarWorker(QObject):
    logLine = Signal(str)
    finished = Signal(object)

    def __init__(
        self,
        session: SessionSettings,
        *,
        enable_onnx: bool,
        pipeline=None,
    ) -> None:
        super().__init__()
        self._session = session
        self._enable_onnx = enable_onnx
        self._pipeline = pipeline

    def run(self) -> None:
        try:
            from tools.measure.core import (
                config_from_session_dict,
                run_algorithm_trial,
                run_raw_polar_trial,
            )

            store = TrialStore(self._session.measure_root)
            run_id = self._session.ensure_run_id(store)
            fields = self._session.to_trial_fields()
            fields["active_run_id"] = run_id
            cfg = config_from_session_dict(fields)
            if self._enable_onnx:
                cfg.kind = "polar"
                cfg.runtime.tap_select = "raw,mapped,dry,wet"
                live = self._pipeline if self._pipeline is not None and self._pipeline.is_ready else None
                self.logLine.emit(
                    f"[polar/onnx] az={cfg.geometry.azimuth_deg} el={cfg.geometry.elevation_deg} "
                    f"hats={cfg.hats_in_device} run={run_id} taps={cfg.runtime.tap_select} "
                    f"live={'yes' if live else 'spawn'}"
                )
                doc = run_algorithm_trial(cfg, pipeline_session=live)
            else:
                self.logLine.emit(
                    f"[polar/raw] az={cfg.geometry.azimuth_deg} el={cfg.geometry.elevation_deg} "
                    f"hats={cfg.hats_in_device} run={run_id}"
                )
                doc = run_raw_polar_trial(cfg)
            self.finished.emit(doc)
        except Exception as exc:  # noqa: BLE001
            self.logLine.emit(traceback.format_exc())
            self.finished.emit(exc)


class PolarTab(QWidget):
    """Az/el pose queue: hardware or ONNX with full taps + optional ear-rig."""

    trialCreated = Signal(str, str)

    def __init__(
        self,
        session: SessionSettings,
        parent: QWidget | None = None,
        *,
        pipeline=None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.pipeline = pipeline
        self._thread: QThread | None = None
        self._worker: _PolarWorker | None = None

        root = QVBoxLayout(self)
        root.addWidget(
            QLabel(
                "Polar az/el: Setup stimulus + Pico mics; optional ONNX (raw/mapped/dry/wet taps); "
                "ear-rig stereo when hats_in_device is set. Start-of-take time sync across USB "
                "interfaces (not bit-clock locked). Same pose queues for az and el sweeps."
            )
        )

        opts = QGroupBox("Capture mode")
        opts_form = QFormLayout(opts)
        self.enable_onnx = QCheckBox("Enable ONNX pipeline (raw+mapped+dry+wet taps)")
        self.enable_onnx.setChecked(bool(session.use_zoo or session.model_path))
        self.enable_onnx.toggled.connect(self._refresh_expected)
        opts_form.addRow(self.enable_onnx)
        self.expected = QLabel()
        self.expected.setWordWrap(True)
        opts_form.addRow("Expected WAVs", self.expected)
        root.addWidget(opts)
        self._refresh_expected()

        box = QGroupBox("Geometry for next take")
        form = QFormLayout(box)
        self.az = QDoubleSpinBox()
        self.az.setRange(-180.0, 360.0)
        self.az.setDecimals(1)
        self.az.setSuffix(" °")
        self.az.setValue(session.azimuth_deg)
        self.el = QDoubleSpinBox()
        self.el.setRange(-90.0, 90.0)
        self.el.setDecimals(1)
        self.el.setSuffix(" °")
        self.el.setValue(session.elevation_deg)
        self.distance = QDoubleSpinBox()
        self.distance.setRange(0.1, 20.0)
        self.distance.setDecimals(2)
        self.distance.setSuffix(" m")
        self.distance.setValue(session.distance_m)
        form.addRow("Azimuth", self.az)
        form.addRow("Elevation", self.el)
        form.addRow("Distance (tag)", self.distance)
        root.addWidget(box)

        self.queue = PoseQueuePanel(apply_pose=self._apply_pose)
        root.addWidget(self.queue)
        self.auto_advance = QCheckBox("After successful capture, mark pose done → next")
        self.auto_advance.setChecked(True)
        root.addWidget(self.auto_advance)

        row = QHBoxLayout()
        self.prep_btn = QPushButton("Prepare trial folder only")
        self.capture_btn = QPushButton("Capture polar")
        self.prep_btn.clicked.connect(self._prepare_trial)
        self.capture_btn.clicked.connect(self._capture)
        row.addWidget(self.prep_btn)
        row.addWidget(self.capture_btn)
        row.addStretch(1)
        root.addLayout(row)

        self.status = QLabel("No polar trial yet.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, stretch=1)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._refresh_expected()

    def _expected_wavs(self) -> list[str]:
        wavs = ["stimulus_used.wav"]
        if self.enable_onnx.isChecked():
            wavs.extend(["raw_8ch.wav", "mapped_6ch.wav", "dry_ref.wav", "wet_out.wav"])
        else:
            wavs.append("raw_polar_8ch.wav")
        if self.session.hats_in_device is not None:
            wavs.extend(["hats_stereo.wav", "hats_l.wav", "hats_r.wav"])
        return wavs

    def _refresh_expected(self) -> None:
        self.expected.setText(", ".join(self._expected_wavs()))

    def _apply_pose(self, pose: Pose) -> None:
        self.az.setValue(pose.azimuth_deg)
        self.el.setValue(pose.elevation_deg)
        self.distance.setValue(pose.distance_m)
        self._sync_geo()

    def _sync_geo(self) -> None:
        self.session.azimuth_deg = float(self.az.value())
        self.session.elevation_deg = float(self.el.value())
        self.session.distance_m = float(self.distance.value())
        self.session.moving_source_m = float(self.distance.value())

    def _prepare_trial(self) -> None:
        """Write a run-level scratch note only — Capture creates the real trial.

        Avoids orphan ``pending_capture`` trial folders that previously appeared
        when Prepare called ``new_trial`` and Capture created a second trial.
        """
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(self, "Busy", "Polar capture already running.")
            return
        self._sync_geo()
        stim = self.session.stimulus_path.strip()
        if not stim:
            QMessageBox.warning(
                self,
                "Stimulus missing",
                "Set Setup → Stimulus WAV (shared REW ESS + timing-ref) before polar takes.",
            )
            return
        if not Path(stim).is_file():
            QMessageBox.warning(self, "Stimulus missing", f"Not found:\n{stim}")
            return

        store = TrialStore(self.session.measure_root)
        run_id = self.session.ensure_run_id(store)
        run_root = Path(self.session.measure_root) / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        expected = self._expected_wavs()
        pending = {
            "type": "polar_pending",
            "run_id": run_id,
            "enable_onnx": self.enable_onnx.isChecked(),
            "kind": "polar" if self.enable_onnx.isChecked() else "raw_polar",
            "azimuth_deg": self.session.azimuth_deg,
            "elevation_deg": self.session.elevation_deg,
            "distance_m": self.session.distance_m,
            "stimulus_path": stim,
            "expected_wavs": expected,
            "note": (
                "Scratch note only — Capture polar creates the durable trial. "
                "No orphan trial folder is created here."
            ),
        }
        pending_path = run_root / "polar_pending.json"
        pending_path.write_text(json.dumps(pending, indent=2), encoding="utf-8")
        self.status.setText(f"Ready → run {run_id} (pending note at {pending_path.name})")
        QMessageBox.information(
            self,
            "Polar ready",
            f"Run: {run_id}\n"
            f"Pending note: {pending_path.name}\n\n"
            f"expected: {', '.join(expected)}\n"
            "Click Capture polar to play+record into a new trial "
            "(Prepare no longer creates an empty trial folder).",
        )

    def _capture(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(self, "Busy", "Polar capture already running.")
            return
        self._sync_geo()
        self._refresh_expected()
        errs: list[str] = []
        stim = self.session.stimulus_path.strip()
        if not stim or not Path(stim).is_file():
            errs.append("Setup → Stimulus WAV is required and must exist.")
        if self.session.stimulus_out_device is None:
            errs.append("Setup → Stimulus out device is required.")
        if self.session.pipeline_in_device is None:
            errs.append("Setup → Pipeline in (Pico mics) is required.")
        if self.enable_onnx.isChecked() and self.session.pipeline_out_device is None:
            errs.append("Setup → Pipeline out is required when ONNX is enabled.")
        if self.enable_onnx.isChecked() and (
            self.pipeline is None or not self.pipeline.is_ready
        ):
            errs.append(
                "Global pipeline is not Ready — click Start Pipeline in the toolbar "
                "and wait until status shows Ready before ONNX polar takes."
            )
        if errs:
            QMessageBox.warning(self, "Setup incomplete", "\n".join(errs))
            return

        self.log.clear()
        mode = "ONNX" if self.enable_onnx.isChecked() else "hardware"
        self.status.setText(f"Capturing polar ({mode})…")
        self.capture_btn.setEnabled(False)
        self.prep_btn.setEnabled(False)

        self._thread = QThread(self)
        self._worker = _PolarWorker(
            self.session,
            enable_onnx=self.enable_onnx.isChecked(),
            pipeline=self.pipeline,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.logLine.connect(self.log.append)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _refresh_run_analysis(self, run_id: str, trial_id: str) -> None:
        """Analyze the new trial once, then rebuild aggregates without re-analyzing all."""
        try:
            from tools.tech_testbench.app.analysis.aggregate import analyze_run
            from tools.tech_testbench.app.analysis.metrics import write_trial_analysis

            trial_dir = Path(self.session.measure_root) / run_id / trial_id
            if trial_dir.is_dir():
                write_trial_analysis(trial_dir, write_plots=True, merge_existing_metrics=True)
                self.log.append(f"[analyze] trial={trial_id} refreshed")
            summary = analyze_run(
                self.session.measure_root,
                run_id=run_id,
                kind=None,
                write_per_trial=False,
                write_plots=True,
            )
            arts = (summary or {}).get("artifacts") or {}
            self.log.append(
                f"[analyze] run={run_id} artifacts={len(arts)} "
                f"dir={(summary or {}).get('analysis_dir')}"
            )
        except Exception as exc:  # noqa: BLE001
            self.log.append(f"[analyze] failed: {exc}\n{traceback.format_exc()}")

    def _on_finished(self, result) -> None:
        self.capture_btn.setEnabled(True)
        self.prep_btn.setEnabled(True)
        self._worker = None
        self._thread = None
        if isinstance(result, Exception):
            self.status.setText(f"Failed: {result}")
            QMessageBox.critical(self, "Polar failed", str(result))
            return
        run_id = str(result.get("run_id") or "")
        trial_id = str(result.get("trial_id") or "")
        err = result.get("error")
        self.status.setText(
            f"{'ERROR' if err else 'OK'} → measure/{run_id}/{trial_id}"
            + (f"\n{err}" if err else "")
        )
        self.log.append(json.dumps(result, indent=2, default=str))
        if run_id and trial_id:
            self.trialCreated.emit(run_id, trial_id)
        if err:
            QMessageBox.warning(self, "Polar error", str(err))
            return
        if run_id and trial_id:
            self.status.setText(self.status.text() + "\nRefreshing run analysis…")
            self._refresh_run_analysis(run_id, trial_id)
        if self.auto_advance.isChecked():
            self.queue.on_trial_captured()
        QMessageBox.information(self, "Polar captured", f"measure/{run_id}/{trial_id}")


# Back-compat for older imports.
RawPolarTab = PolarTab
