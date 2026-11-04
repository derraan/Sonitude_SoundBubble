"""E2E latency — run stimulus through the rig with taps; measure ms across stages."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.session import SessionSettings
from app.storage.trial_store import TrialStore


class _E2ELatencyWorker(QObject):
    logLine = Signal(str)
    finished = Signal(object)  # (trial_doc, latency_result) | Exception

    def __init__(
        self,
        session: SessionSettings,
        *,
        max_tau_s: float,
        raw_channel: int | None,
        pipeline=None,
    ) -> None:
        super().__init__()
        self._session = session
        self._max_tau_s = max_tau_s
        self._raw_channel = raw_channel
        self._pipeline = pipeline

    def run(self) -> None:
        try:
            from tools.measure.core import config_from_session_dict, run_algorithm_trial
            from tools.tech_testbench.app.analysis.e2e_latency import (
                analyze_e2e_latency_trial,
            )

            store = TrialStore(self._session.measure_root)
            run_id = self._session.ensure_run_id(store)
            fields = self._session.to_trial_fields()
            fields["active_run_id"] = run_id
            fields["kind"] = "e2e_latency"
            cfg = config_from_session_dict(fields)
            cfg.kind = "e2e_latency"
            # Ensure all stages needed for the chain are tapped.
            cfg.runtime.tap_select = "raw,mapped,dry,wet"
            live = self._pipeline if self._pipeline is not None and self._pipeline.is_ready else None

            self.logLine.emit(
                f"[e2e_latency] profile={cfg.acoustic_profile} runtime={cfg.runtime.runtime} "
                f"run={run_id} max_tau_s={self._max_tau_s} live={'yes' if live else 'spawn'}"
            )
            doc = run_algorithm_trial(cfg, pipeline_session=live)
            trial_id = str(doc.get("trial_id") or "")
            trial_root = Path(self._session.measure_root) / run_id / trial_id
            self.logLine.emit(f"[e2e_latency] capture done → {trial_root}")

            if doc.get("error"):
                self.logLine.emit(f"[e2e_latency] trial error: {doc['error']}")

            latency = None
            if trial_root.is_dir():
                latency = analyze_e2e_latency_trial(
                    trial_root,
                    max_tau_s=float(self._max_tau_s),
                    raw_channel=self._raw_channel,
                    write_plots=True,
                )
                if latency is None:
                    self.logLine.emit(
                        "[e2e_latency] analysis skipped — need ≥2 of "
                        "stimulus_used / raw_8ch / dry_ref / wet_out"
                    )
                else:
                    sm = latency.get("summary_ms") or {}
                    self.logLine.emit(
                        "[e2e_latency] "
                        f"stim→raw={sm.get('stimulus_to_raw_ms')} ms  "
                        f"raw→dry={sm.get('raw_to_dry_ms')} ms  "
                        f"dry→wet={sm.get('dry_to_wet_ms')} ms  "
                        f"stim→wet={sm.get('stimulus_to_wet_ms')} ms"
                    )
                    # Refresh trial doc after analysis wrote e2e fields.
                    tj = trial_root / "trial.json"
                    if tj.is_file():
                        try:
                            doc = json.loads(tj.read_text(encoding="utf-8"))
                        except json.JSONDecodeError:
                            pass

            self.finished.emit({"trial": doc, "latency": latency})
        except Exception as exc:  # noqa: BLE001
            self.logLine.emit(traceback.format_exc())
            self.finished.emit(exc)


class E2ELatencyTab(QWidget):
    """Play Setup stimulus through ONNX RT + taps; report stage latencies in ms."""

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
        self._worker: _E2ELatencyWorker | None = None

        root = QVBoxLayout(self)
        root.addWidget(
            QLabel(
                "Runs the same stimulus → rig → ONNX path as Algorithm, with taps "
                "raw / mapped / dry / wet. After capture, GCC-PHAT measures latency "
                "(ms) between stimulus, raw mic, dry_ref, and wet_out."
            )
        )

        opts = QGroupBox("Measurement")
        form = QFormLayout(opts)
        self.max_tau = QDoubleSpinBox()
        self.max_tau.setRange(0.05, 5.0)
        self.max_tau.setDecimals(2)
        self.max_tau.setSingleStep(0.05)
        self.max_tau.setValue(1.0)
        self.max_tau.setSuffix(" s")
        self.max_tau.setToolTip(
            "Max |lag| searched by GCC-PHAT. Raise if settle + acoustic path is long."
        )
        self.raw_ch = QSpinBox()
        self.raw_ch.setRange(-1, 7)
        self.raw_ch.setValue(-1)
        self.raw_ch.setSpecialValueText("mean (all ch)")
        self.raw_ch.setToolTip(
            "Which raw_8ch channel to correlate (-1 = mean of all channels)."
        )
        form.addRow("Max lag search", self.max_tau)
        form.addRow("Raw channel", self.raw_ch)
        root.addWidget(opts)

        self.gains_lbl = QLabel()
        self.gains_lbl.setWordWrap(True)
        root.addWidget(self.gains_lbl)
        self._refresh_gains_label()

        row = QHBoxLayout()
        self.start_btn = QPushButton("Run e2e latency")
        self.start_btn.clicked.connect(self._start)
        row.addWidget(self.start_btn)
        row.addStretch(1)
        root.addLayout(row)

        self.status = QLabel("No e2e latency trial yet.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, stretch=1)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._refresh_gains_label()

    def _refresh_gains_label(self) -> None:
        g = self.session.gains
        self.gains_lbl.setText(
            f"Setup → runtime={self.session.runtime}  "
            f"in={g.input_gain_db:.2f} dB  out={g.output_gain_db:.2f} dB  "
            f"mix_dry={g.mix_dry:.3f}  profile={self.session.acoustic_profile}  "
            f"repeat×{self.session.stimulus_repeat}"
        )

    def _set_busy(self, busy: bool) -> None:
        self.start_btn.setEnabled(not busy)

    def _start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(self, "Busy", "E2E latency trial already running.")
            return

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

        raw_ch_val = int(self.raw_ch.value())
        raw_channel = None if raw_ch_val < 0 else raw_ch_val

        self.log.clear()
        self.status.setText("Starting e2e latency trial…")
        self._set_busy(True)
        self._refresh_gains_label()

        self._thread = QThread(self)
        self._worker = _E2ELatencyWorker(
            self.session,
            max_tau_s=float(self.max_tau.value()),
            raw_channel=raw_channel,
            pipeline=self.pipeline,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.logLine.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_log(self, line: str) -> None:
        self.log.append(line)

    def _on_finished(self, result) -> None:
        self._set_busy(False)
        self._worker = None
        self._thread = None
        if isinstance(result, Exception):
            self.status.setText(f"Failed: {result}")
            QMessageBox.critical(self, "E2E latency failed", str(result))
            return

        doc = result.get("trial") or {}
        latency = result.get("latency")
        run_id = str(doc.get("run_id") or "")
        trial_id = str(doc.get("trial_id") or "")
        err = doc.get("error")

        sm = (latency or {}).get("summary_ms") or {}
        summary_line = (
            f"stim→raw {sm.get('stimulus_to_raw_ms')} ms · "
            f"raw→dry {sm.get('raw_to_dry_ms')} ms · "
            f"dry→wet {sm.get('dry_to_wet_ms')} ms · "
            f"stim→wet {sm.get('stimulus_to_wet_ms')} ms"
            if latency
            else "(no latency result — check taps)"
        )
        self.status.setText(
            f"{'ERROR' if err else 'OK'} → measure/{run_id}/{trial_id}\n{summary_line}"
            + (f"\n{err}" if err else "")
        )
        self.log.append("\n" + json.dumps({"latency_ms": sm, "pairs": (latency or {}).get("pairs")}, indent=2, default=str))

        if run_id and trial_id:
            self.trialCreated.emit(run_id, trial_id)

        if err:
            QMessageBox.warning(self, "Trial finished with errors", str(err))
        else:
            QMessageBox.information(
                self,
                "E2E latency complete",
                f"measure/{run_id}/{trial_id}\n\n{summary_line}\n\n"
                "Details → plots/e2e_latency.* and metrics.json",
            )
