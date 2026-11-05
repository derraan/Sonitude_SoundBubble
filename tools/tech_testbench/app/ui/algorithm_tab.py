"""Algorithm trials — selectable ONNX RT + taps (+ HATS on pi_hats); durable trial folders."""

from __future__ import annotations

import json
import traceback

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.session import SessionSettings
from app.storage.trial_store import TrialStore
from app.ui.pose_queue import Pose, PoseQueuePanel


class _MeasureWorker(QObject):
    logLine = Signal(str)
    finished = Signal(object)  # trial.json dict | Exception

    def __init__(self, session: SessionSettings, pipeline=None) -> None:
        super().__init__()
        self._session = session
        self._pipeline = pipeline

    def run(self) -> None:
        try:
            from tools.measure.core import config_from_session_dict, run_algorithm_trial

            store = TrialStore(self._session.measure_root)
            run_id = self._session.ensure_run_id(store)
            fields = self._session.to_trial_fields()
            fields["active_run_id"] = run_id
            cfg = config_from_session_dict(fields)
            live = self._pipeline if self._pipeline is not None and self._pipeline.is_ready else None
            self.logLine.emit(
                f"[algorithm] profile={cfg.acoustic_profile} runtime={cfg.runtime.runtime} "
                f"kind={cfg.campaign_kind} dir={cfg.motion_direction} run={run_id} "
                f"live={'yes' if live else 'spawn'}"
            )
            doc = run_algorithm_trial(cfg, pipeline_session=live)
            self.finished.emit(doc)
        except Exception as exc:  # noqa: BLE001
            self.logLine.emit(traceback.format_exc())
            self.finished.emit(exc)


class AlgorithmTab(QWidget):
    """Starts tools.measure core (ONNX RT + stimulus + optional HATS) into TrialStore."""

    trialCreated = Signal(str, str)

    def __init__(
        self,
        session: SessionSettings,
        validate_setup,
        parent: QWidget | None = None,
        *,
        pipeline=None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.pipeline = pipeline
        self._validate_setup = validate_setup
        self._thread: QThread | None = None
        self._worker: _MeasureWorker | None = None
        self._auto_advance = True

        root = QVBoxLayout(self)
        root.addWidget(
            QLabel(
                "Starts selectable ONNX RT (cpp|python) with Setup gains + --tap-dir, "
                "plays Setup stimulus on stimulus_out, and on pi_hats records HATS in parallel "
                "via tools.measure. Library reloads the trial folder."
            )
        )

        ctx = QGroupBox("Campaign context")
        ctx_form = QFormLayout(ctx)
        self.campaign = QComboBox()
        self.campaign.addItem("controlled_ess — REW ESS characterization", "controlled_ess")
        self.campaign.addItem("in_context_speech — speech / product scenes", "in_context_speech")
        self.direction = QComboBox()
        self.direction.addItem("static", "static")
        self.direction.addItem("into_bubble (near → far reverse off)", "into_bubble")
        self.direction.addItem("out_of_bubble (far → near)", "out_of_bubble")
        self.fixed_src = QDoubleSpinBox()
        self.fixed_src.setRange(0.0, 20.0)
        self.fixed_src.setDecimals(2)
        self.fixed_src.setSuffix(" m")
        self.fixed_src.setSpecialValueText("(none — 1 speaker)")
        self.fixed_src.setMinimum(0.0)
        self.fixed_src.setValue(0.0)
        self.use_fixed = QCheckBox("Tag fixed/external 2nd source distance")
        self.use_fixed.setChecked(False)
        self.notes = QLineEdit()
        self.notes.setPlaceholderText(
            "e.g. BT speaker fixed @ 0.5 m playing utterance_loop.wav externally"
        )
        ctx_form.addRow("Campaign", self.campaign)
        ctx_form.addRow("Motion direction", self.direction)
        ctx_form.addRow(self.use_fixed)
        ctx_form.addRow("Fixed source", self.fixed_src)
        ctx_form.addRow("Notes", self.notes)
        root.addWidget(ctx)

        geo = QGroupBox("Trial geometry")
        form = QFormLayout(geo)
        self.distance = QDoubleSpinBox()
        self.distance.setRange(0.1, 20.0)
        self.distance.setSingleStep(0.1)
        self.distance.setDecimals(2)
        self.distance.setSuffix(" m")
        self.distance.setValue(session.distance_m)
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
        form.addRow("Distance / moving source", self.distance)
        form.addRow("Azimuth", self.az)
        form.addRow("Elevation", self.el)
        root.addWidget(geo)

        self.queue = PoseQueuePanel(apply_pose=self._apply_pose)
        root.addWidget(self.queue)
        self.auto_advance = QCheckBox("After successful trial, mark pose done → next")
        self.auto_advance.setChecked(True)
        root.addWidget(self.auto_advance)

        self.gains_lbl = QLabel()
        self.gains_lbl.setWordWrap(True)
        root.addWidget(self.gains_lbl)
        self._refresh_gains_label()

        row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview trial.json fields")
        self.folder_btn = QPushButton("Create trial folder only")
        self.start_btn = QPushButton("Start algorithm trial")
        self.preview_btn.clicked.connect(self._preview)
        self.folder_btn.clicked.connect(self._create_folder)
        self.start_btn.clicked.connect(self._start)
        row.addWidget(self.preview_btn)
        row.addWidget(self.folder_btn)
        row.addWidget(self.start_btn)
        row.addStretch(1)
        root.addLayout(row)

        self.status = QLabel("No algorithm trial yet.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        root.addWidget(self.preview, stretch=1)

        # Start from Setup's (restored) campaign; _on_context_changed below writes the
        # combo back into the session, so a default index would clobber it.
        idx = self.campaign.findData(session.campaign_kind)
        if idx >= 0:
            self.campaign.setCurrentIndex(idx)
        self.campaign.currentIndexChanged.connect(self._on_context_changed)
        self.direction.currentIndexChanged.connect(self._on_context_changed)
        self.use_fixed.toggled.connect(self._on_context_changed)
        self.fixed_src.valueChanged.connect(self._on_context_changed)
        self.notes.editingFinished.connect(self._sync_context)
        self._on_context_changed()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._refresh_gains_label()
        # Keep campaign combo aligned with Setup if session changed elsewhere.
        idx = self.campaign.findData(self.session.campaign_kind)
        if idx >= 0 and self.campaign.currentIndex() != idx:
            self.campaign.blockSignals(True)
            self.campaign.setCurrentIndex(idx)
            self.campaign.blockSignals(False)

    def _apply_pose(self, pose: Pose) -> None:
        self.distance.setValue(pose.distance_m)
        self.az.setValue(pose.azimuth_deg)
        self.el.setValue(pose.elevation_deg)
        self._sync_geo()

    def _on_context_changed(self) -> None:
        self._sync_context()
        reverse = self.session.motion_direction == "out_of_bubble"
        self.queue.set_reverse_distance(reverse)

    def _sync_context(self) -> None:
        self.session.campaign_kind = str(self.campaign.currentData())
        self.session.motion_direction = str(self.direction.currentData())
        if self.use_fixed.isChecked():
            self.session.fixed_source_m = float(self.fixed_src.value())
        else:
            self.session.fixed_source_m = None
        self.session.notes = self.notes.text().strip()
        self.session.moving_source_m = float(self.distance.value())

    def _refresh_gains_label(self) -> None:
        g = self.session.gains
        self.gains_lbl.setText(
            f"Setup gains → runtime={self.session.runtime}  "
            f"in={g.input_gain_db:.2f} dB  out={g.output_gain_db:.2f} dB  "
            f"mix_dry={g.mix_dry:.3f}  limiter={'on' if g.output_limiter else 'off'} "
            f"({g.limiter_release_ms:.0f} ms)  profile={self.session.acoustic_profile}  "
            f"repeat×{self.session.stimulus_repeat}"
        )

    def _sync_geo(self) -> None:
        self.session.distance_m = float(self.distance.value())
        self.session.azimuth_deg = float(self.az.value())
        self.session.elevation_deg = float(self.el.value())
        self.session.moving_source_m = float(self.distance.value())
        self._sync_context()
        self._refresh_gains_label()

    def _expected_wavs(self) -> list[str]:
        wavs = [
            "raw_8ch.wav",
            "mapped_6ch.wav",
            "dry_ref.wav",
            "wet_out.wav",
            "stimulus_used.wav",
        ]
        if self.session.acoustic_profile == "pi_hats":
            wavs.extend(["hats_l.wav", "hats_r.wav", "hats_stereo.wav"])
        return wavs

    def _preview(self) -> None:
        self._sync_geo()
        payload = {
            **self.session.to_trial_fields(),
            "kind": "algorithm",
            "expected_wavs": self._expected_wavs(),
        }
        self.preview.setPlainText(json.dumps(payload, indent=2))

    def _create_folder(self) -> None:
        self._sync_geo()
        store = TrialStore(self.session.measure_root)
        run_id = self.session.ensure_run_id(store)
        paths = store.new_trial(
            run_id=run_id,
            kind="algorithm",
            trial_fields={**self.session.to_trial_fields()},
        )
        store.copy_stimulus(paths, self.session.stimulus_path)
        store.save_metrics(paths, {"pending": True, "expected_wavs": self._expected_wavs()})
        doc = store.load_trial_json(paths)
        self.preview.setPlainText(json.dumps({"created": str(paths.root), **doc}, indent=2))
        self.status.setText(f"Ready → {paths.root}")
        self.trialCreated.emit(paths.run_id, paths.trial_id)

    def _set_busy(self, busy: bool) -> None:
        self.start_btn.setEnabled(not busy)
        self.folder_btn.setEnabled(not busy)
        self.preview_btn.setEnabled(not busy)

    def _start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(self, "Busy", "Algorithm trial already running.")
            return
        self._sync_geo()
        errs = list(self._validate_setup() or [])
        if self.session.pipeline_out_device is None:
            errs.append("Pipeline output device (Pico DAC / earcups) is required.")
        if self.session.stimulus_out_device is None:
            errs.append("Stimulus out device is required.")
        if (
            self.session.stimulus_out_device is not None
            and self.session.pipeline_out_device is not None
            and int(self.session.stimulus_out_device) == int(self.session.pipeline_out_device)
        ):
            errs.append(
                "stimulus_out == pipeline_out will conflict (exclusive PortAudio). "
                "Use a separate stimulus speaker (windows_pico) or speakers vs Pico DAC (pi_hats)."
            )
        if self.pipeline is None or not self.pipeline.is_ready:
            errs.append(
                "Global pipeline is not Ready — click Start Pipeline in the toolbar "
                "and wait until status shows Ready."
            )
        if errs:
            QMessageBox.warning(self, "Setup incomplete", "\n".join(errs))
            return

        self.preview.clear()
        self.status.setText("Starting algorithm trial…")
        self._set_busy(True)

        self._thread = QThread(self)
        self._worker = _MeasureWorker(self.session, pipeline=self.pipeline)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.logLine.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_log(self, line: str) -> None:
        self.preview.append(line)

    def _on_finished(self, result) -> None:
        self._set_busy(False)
        self._worker = None
        self._thread = None
        if isinstance(result, Exception):
            self.status.setText(f"Failed: {result}")
            QMessageBox.critical(self, "Algorithm trial failed", str(result))
            return
        run_id = str(result.get("run_id") or "")
        trial_id = str(result.get("trial_id") or "")
        err = result.get("error")
        self.status.setText(
            f"{'ERROR' if err else 'OK'} → measure/{run_id}/{trial_id}"
            + (f"\n{err}" if err else "")
        )
        self.preview.append("\n" + json.dumps(result, indent=2, default=str))
        if run_id and trial_id:
            self.trialCreated.emit(run_id, trial_id)
        if err:
            QMessageBox.warning(self, "Trial finished with errors", str(err))
        else:
            if self.auto_advance.isChecked():
                self.queue.on_trial_captured()
            wavs = (result.get("artifacts") or {}).get("wavs") or []
            plots = (result.get("artifacts") or {}).get("plots") or []
            fr_note = ""
            if any("suppression_fr" in str(p) for p in plots):
                fr_note = "\n\nWet/suppression FR → plots/suppression_fr.*"
                self.preview.append("[analysis] suppression_fr written for Library / polar FR.")
            QMessageBox.information(
                self,
                "Trial complete",
                f"measure/{run_id}/{trial_id}\n\nWAVs: {', '.join(wavs) or '(none)'}"
                f"{fr_note}",
            )
