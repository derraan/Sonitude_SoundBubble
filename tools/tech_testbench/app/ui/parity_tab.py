"""C++ vs Python offline WAV parity — creates TrialStore trial and runs xcheck."""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
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


class _ParityWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, kwargs: dict) -> None:
        super().__init__()
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            from tools.measure.parity import run_parity
            from tools.measure.runtime import Gains

            kw = dict(self.kwargs)
            gains_d = kw.pop("gains")
            kw["gains"] = Gains(**gains_d)
            result = run_parity(**kw)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class ParityTab(QWidget):
    trialCreated = Signal(str, str)

    def __init__(self, session: SessionSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self._thread: QThread | None = None
        self._worker: _ParityWorker | None = None

        root = QVBoxLayout(self)
        root.addWidget(
            QLabel(
                "Offline: feed one raw_8ch.wav through cpp and python with identical Setup "
                "gains; diff wet. Writes wet_cpp / wet_python / wet_diff + metrics into a "
                "parity TrialStore trial."
            )
        )
        form = QFormLayout()
        self.raw = QLineEdit()
        browse = QPushButton("…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.raw)
        row.addWidget(browse)
        form.addRow("raw_8ch.wav", row)
        root.addLayout(form)

        btn_row = QHBoxLayout()
        folder_btn = QPushButton("Create parity trial folder")
        folder_btn.clicked.connect(self._create_folder)
        self.run_btn = QPushButton("Run parity cross-check")
        self.run_btn.clicked.connect(self._run_parity)
        btn_row.addWidget(folder_btn)
        btn_row.addWidget(self.run_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.status = QLabel("No parity trial yet.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        root.addWidget(self.log)
        root.addStretch(1)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "raw_8ch WAV", "", "WAV (*.wav)")
        if path:
            self.raw.setText(path)

    def _create_folder(self) -> None:
        raw = self.raw.text().strip()
        store = TrialStore(self.session.measure_root)
        run_id = self.session.ensure_run_id(store)
        fields = {**self.session.to_trial_fields(), "parity_raw_8ch": raw}
        paths = store.new_trial(run_id=run_id, kind="parity", trial_fields=fields)
        if raw and Path(raw).is_file():
            store.register_wav(paths, "raw_8ch.wav", raw)
        store.copy_stimulus(paths, self.session.stimulus_path)
        store.save_metrics(
            paths,
            {
                "pending": True,
                "expected_wavs": [
                    "raw_8ch.wav",
                    "wet_cpp.wav",
                    "wet_python.wav",
                    "wet_diff.wav",
                ],
                "gains": self.session.gains.to_dict(),
            },
        )
        csv_path = paths.plot_path("parity_pending.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            f.write("metric,value\n")
            f.write("pending,1\n")
            f.write(f"raw_8ch_set,{1 if raw else 0}\n")
        store.register_plot(
            paths,
            stem="parity_pending",
            series_csv=csv_path,
            series_json={
                "type": "parity_pending",
                "note": "Use Run parity cross-check to fill wet diffs",
                "raw_8ch": raw or None,
            },
        )
        self.status.setText(f"Ready → {paths.root}")
        self.trialCreated.emit(paths.run_id, paths.trial_id)
        QMessageBox.information(
            self,
            "Parity trial folder",
            f"{paths.root}\n\n"
            "Click “Run parity cross-check” to replay through soundbubble-rt and "
            "inference_pipeline with locked Setup gains.",
        )

    def _run_parity(self) -> None:
        raw = self.raw.text().strip()
        if not raw or not Path(raw).is_file():
            QMessageBox.warning(self, "Parity", "Select an existing raw_8ch.wav first.")
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "Parity", "Parity run already in progress.")
            return

        store = TrialStore(self.session.measure_root)
        run_id = self.session.ensure_run_id(store)
        s = self.session

        if s.use_zoo:
            zoo_dir = s.zoo_dir or ""
            model_path = ""
            contract_path = ""
        else:
            zoo_dir = ""
            model_path = s.model_path or ""
            contract_path = s.contract_path or ""
            if not model_path:
                QMessageBox.warning(
                    self,
                    "Parity",
                    "Setup needs a zoo dir + bubble radius, or an explicit model path.",
                )
                return

        kwargs = {
            "raw_8ch": raw,
            "gains": s.gains.to_dict(),
            "measure_root": s.measure_root,
            "run_id": run_id,
            "model_path": model_path,
            "contract_path": contract_path,
            "zoo_dir": zoo_dir,
            "bubble_radius_m": s.bubble_radius_m,
            "soundbubble_rt_path": s.soundbubble_rt_path or "",
            "channel_map": s.channel_map or "0,1,2,3,4,5",
            "input_sr": s.input_sr,
            "output_sr": s.output_sr,
            "trial_fields": s.to_trial_fields(),
        }

        self.run_btn.setEnabled(False)
        self.status.setText("Running offline cpp + python replay…")
        self.log.setPlainText("Starting parity…\n")

        self._thread = QThread(self)
        self._worker = _ParityWorker(kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _on_finished(self, result: object) -> None:
        self.run_btn.setEnabled(True)
        wet = getattr(result, "metrics", {}).get("wet_diff") or {}
        msg = (
            f"Done → {result.trial_dir}\n"
            f"max_abs_diff={wet.get('max_abs_diff')}  "
            f"rms_diff={wet.get('rms_diff')}  corr={wet.get('corr')}"
        )
        self.status.setText(msg)
        self.log.append(msg)
        self.trialCreated.emit(result.run_id, result.trial_id)
        QMessageBox.information(self, "Parity complete", msg)

    def _on_failed(self, err: str) -> None:
        self.run_btn.setEnabled(True)
        self.status.setText("Parity failed.")
        self.log.append(err)
        QMessageBox.critical(self, "Parity failed", err[:2000])

    def _cleanup_thread(self) -> None:
        self._worker = None
        self._thread = None
