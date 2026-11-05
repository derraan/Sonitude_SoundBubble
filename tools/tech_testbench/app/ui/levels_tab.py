"""Mic element levels — wraps tools/mic_calibration_probe.py; saves under TrialStore."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.session import REPO_ROOT, SessionSettings
from app.storage.trial_store import TrialStore


class LevelsTab(QWidget):
    """Hardware-only levels; TrialStore is the sole persistence writer."""

    trialCreated = Signal(str, str)  # run_id, trial_id

    def __init__(self, session: SessionSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self._proc: QProcess | None = None
        self._log_path: Path | None = None
        self._cal_path: Path | None = None
        self._wav_path: Path | None = None
        self._paths = None  # TrialPaths | None
        self._mode: str = "clap"

        root = QVBoxLayout(self)
        root.addWidget(
            QLabel(
                "Uses the Setup ESS/stimulus WAV (same file as Polar / Algorithm / REW).\n"
                "No jig: (1) per-capsule — one mic at a time on USB slot 0; "
                "(2) headphone array — all capsules seated, arc coplanar to baffle @ 1 m.\n"
                "Primary: play stimulus + calibrate. Secondary: clap-only probe. No ONNX."
            )
        )

        char = QGroupBox("Hardware characterization tag")
        char_form = QFormLayout(char)
        self.hw_mode = QComboBox()
        self.hw_mode.addItem(
            "per_capsule — one mic on USB slot 0",
            "per_capsule",
        )
        self.hw_mode.addItem(
            "headphone_array — all seats, arc coplanar @ 1 m",
            "headphone_array",
        )
        self.capsule_id = QLineEdit()
        self.capsule_id.setPlaceholderText("e.g. left_ear / mic00 / serial or paint mark")
        self.usb_slot = QSpinBox()
        self.usb_slot.setRange(0, 7)
        self.usb_slot.setValue(0)
        self.usb_slot.setToolTip("Wiring seat used for this take. Per-capsule: keep 0.")
        char_form.addRow("Mode", self.hw_mode)
        char_form.addRow("Capsule ID", self.capsule_id)
        char_form.addRow("USB slot", self.usb_slot)
        root.addWidget(char)
        self.hw_mode.currentIndexChanged.connect(self._sync_char_tags)
        self.capsule_id.editingFinished.connect(self._sync_char_tags)
        self.usb_slot.valueChanged.connect(self._sync_char_tags)

        stim_box = QGroupBox("Stimulus")
        stim_form = QFormLayout(stim_box)
        self.play_ch = QComboBox()
        self.play_ch.addItem("Left channel only", "left")
        self.play_ch.addItem("Right channel only", "right")
        self.play_ch.addItem("Both", "both")
        pref = (session.stimulus_play_channels or "left").strip().lower()
        idx = self.play_ch.findData(pref if pref in ("left", "right", "both") else "left")
        self.play_ch.setCurrentIndex(idx if idx >= 0 else 0)
        self.play_ch.setToolTip(
            "Which stereo DAC legs get the ESS. "
            "Files named *_refL need Left (or Both) for the timing reference."
        )
        self.play_ch.currentIndexChanged.connect(self._sync_play_ch)
        self.sweep_btn = QPushButton("Play stimulus + calibrate")
        self.sweep_btn.clicked.connect(lambda: self._run("stimulus_sweep"))
        stim_form.addRow("Play channel", self.play_ch)
        stim_form.addRow(self.sweep_btn)
        root.addWidget(stim_box)

        row = QHBoxLayout()
        self.secs = QSpinBox()
        self.secs.setRange(1, 60)
        self.secs.setValue(5)
        self.secs.setSuffix(" s")
        self.secs.setToolTip("Capture length for clap-only mode (ignored for stimulus duration)")
        self.clap_btn = QPushButton("Clap probe (no play)")
        self.clap_btn.clicked.connect(lambda: self._run("clap"))
        row.addWidget(QLabel("Clap capture"))
        row.addWidget(self.secs)
        row.addWidget(self.clap_btn)
        row.addStretch(1)
        root.addLayout(row)
        self.status = QLabel("No levels trial yet.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, stretch=1)

    def _set_busy(self, busy: bool) -> None:
        self.sweep_btn.setEnabled(not busy)
        self.clap_btn.setEnabled(not busy)

    def _sync_char_tags(self) -> None:
        self.session.hw_char_mode = str(self.hw_mode.currentData())
        self.session.capsule_id = self.capsule_id.text().strip()
        self.session.usb_slot = int(self.usb_slot.value())
        if self.session.hw_char_mode == "per_capsule" and self.usb_slot.value() != 0:
            # Soft preference: per-capsule protocol uses seat 0.
            pass

    def _sync_play_ch(self) -> None:
        self.session.stimulus_play_channels = str(self.play_ch.currentData() or "both")

    def showEvent(self, event) -> None:  # noqa: N802 — Qt
        super().showEvent(event)
        # Pick up Setup → Play channel edits without restarting the app.
        pref = (self.session.stimulus_play_channels or "left").strip().lower()
        idx = self.play_ch.findData(pref if pref in ("left", "right", "both") else "left")
        if idx >= 0 and idx != self.play_ch.currentIndex():
            self.play_ch.blockSignals(True)
            self.play_ch.setCurrentIndex(idx)
            self.play_ch.blockSignals(False)

    def _run(self, mode: str) -> None:
        if self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Busy", "Probe already running.")
            return
        script = REPO_ROOT / "tools" / "mic_calibration_probe.py"
        if not script.is_file():
            QMessageBox.critical(self, "Missing", f"Not found: {script}")
            return

        self._sync_char_tags()
        self._sync_play_ch()
        if mode == "stimulus_sweep":
            errs: list[str] = []
            stim = self.session.stimulus_path.strip()
            if not stim or not Path(stim).is_file():
                errs.append("Setup → Stimulus WAV is required and must exist.")
            if self.session.stimulus_out_device is None:
                errs.append("Setup → Stimulus out device is required.")
            if self.session.pipeline_in_device is None:
                errs.append("Setup → Pipeline in (Pico mics) is required.")
            if self.session.hw_char_mode == "per_capsule" and not self.session.capsule_id:
                errs.append("Per-capsule mode needs a Capsule ID tag before capture.")
            if errs:
                QMessageBox.warning(self, "Setup incomplete", "\n".join(errs))
                return

        store = TrialStore(self.session.measure_root)
        run_id = self.session.ensure_run_id(store)
        paths = store.new_trial(
            run_id=run_id,
            kind="levels",
            trial_fields={**self.session.to_trial_fields()},
        )
        used = store.copy_stimulus(paths, self.session.stimulus_path)
        self._paths = paths
        self._mode = mode
        self._log_path = paths.root / "mic_calibration_probe.ndjson"
        self._cal_path = paths.root / "calibration.json"
        self._wav_path = paths.root / "levels_8ch.wav"
        self.status.setText(f"Trial → {paths.root}")

        label = paths.trial_id
        play_ch = self.session.stimulus_play_channels or "both"
        argv = [
            str(script),
            label,
            "--secs",
            str(self.secs.value()),
            "--log",
            str(self._log_path),
            "--rate",
            str(int(self.session.input_sr or 24000)),
            "--channels",
            "8",
            "--calibration-json",
            str(self._cal_path),
            "--wav-out",
            str(self._wav_path),
        ]
        if mode == "stimulus_sweep":
            argv.extend(
                [
                    "--stimulus",
                    self.session.stimulus_path.strip(),
                    "--out-device",
                    str(int(self.session.stimulus_out_device)),
                    "--in-device",
                    str(int(self.session.pipeline_in_device)),
                    "--stimulus-play-channels",
                    play_ch,
                ]
            )

        self.log.clear()
        self.log.append(f"trial → {paths.root}\n")
        self.log.append(f"mode={mode}  stimulus_copied={bool(used)}\n")
        self.log.append(f"$ {sys.executable} {' '.join(argv)}\n")
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.setWorkingDirectory(str(REPO_ROOT))
        env = QProcessEnvironment.systemEnvironment()
        # Ensure ``import tools.measure`` works even if cwd alone is not enough.
        existing = env.value("PYTHONPATH")
        env.insert(
            "PYTHONPATH",
            str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}",
        )
        self._proc.setProcessEnvironment(env)
        self._proc.readyReadStandardOutput.connect(self._on_out)
        self._proc.finished.connect(self._on_finished)
        self._set_busy(True)
        self._proc.start(sys.executable, argv)

    def _on_out(self) -> None:
        if self._proc is None:
            return
        text = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.log.append(text.rstrip())

    def _on_finished(self, code: int, _status) -> None:
        self._set_busy(False)
        self.log.append(f"\n[exit {code}]")
        if self._paths is None:
            return
        store = TrialStore(self.session.measure_root)
        paths = self._paths
        for name in (
            "mic_calibration_probe.ndjson",
            "calibration.json",
            "levels_8ch.wav",
            "stimulus_used.wav",
        ):
            p = paths.root / name
            if p.is_file():
                store.register_file(paths, name, None)
        store.scan_wavs(paths)
        doc = store.load_trial_json(paths)
        doc["probe_exit_code"] = code
        doc["probe_secs"] = int(self.secs.value())
        doc["levels_mode"] = self._mode
        cal_summary = None
        if self._cal_path and self._cal_path.is_file():
            try:
                cal_summary = json.loads(self._cal_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cal_summary = None
            doc["calibration"] = cal_summary
            self.log.append(
                "\nHand agent calibration.json for firmware k_mic_cal_gain[6] "
                f"(spread_db={cal_summary.get('spread_db_before') if cal_summary else '?'})."
            )
        metrics: dict = {
            "probe_exit_code": code,
            "probe_secs": int(self.secs.value()),
            "levels_mode": self._mode,
            "log": self._log_path.name if self._log_path else None,
            "calibration_json": "calibration.json" if self._cal_path and self._cal_path.is_file() else None,
            "k_mic_cal_gain": (cal_summary or {}).get("k_mic_cal_gain"),
            "spread_db_before": (cal_summary or {}).get("spread_db_before"),
        }
        # Part A relative delay from ESS multi-ch capture (GCC-PHAT).
        if self._mode == "stimulus_sweep" and self._wav_path and self._wav_path.is_file():
            try:
                from tools.tech_testbench.app.analysis.mic_delay import analyze_mic_delay_trial

                delay = analyze_mic_delay_trial(paths.root, write_plots=True)
                if delay:
                    metrics["relative_delay_ms"] = delay.get("relative_delay_ms")
                    metrics["ref_channel"] = delay.get("ref_channel")
                    metrics["mic_delay"] = delay
                    doc["relative_delay_ms"] = delay.get("relative_delay_ms")
                    doc["ref_channel"] = delay.get("ref_channel")
                    self.log.append(
                        f"\nrelative_delay_ms={delay.get('relative_delay_ms')} "
                        f"(ref_ch={delay.get('ref_channel')})"
                    )
            except Exception as exc:  # noqa: BLE001
                self.log.append(f"\n[mic_delay skipped] {exc}")
            # Absolute stim→mic via ESS timing-ref (controlled_ess only).
            try:
                from tools.tech_testbench.app.analysis.timing_ref import analyze_timing_ref_trial

                timing = analyze_timing_ref_trial(
                    paths.root,
                    campaign_kind=str(
                        doc.get("campaign_kind") or self.session.campaign_kind or ""
                    ),
                    play_channels=str(
                        doc.get("stimulus_play_channels")
                        or self.session.stimulus_play_channels
                        or "left"
                    ),
                    write_plots=True,
                )
                if timing is not None:
                    metrics["timing_ref"] = timing
                    doc["timing_ref_enabled"] = bool(timing.get("enabled"))
                    if timing.get("enabled"):
                        doc["stim_to_raw_ms"] = timing.get("stim_to_raw_ms")
                        self.log.append(
                            f"\ntiming_ref stim→raw={timing.get('stim_to_raw_ms')} ms "
                            f"(ref_leg={timing.get('ref_leg')})"
                        )
                    else:
                        self.log.append(
                            f"\n[timing_ref skipped] {timing.get('reason')}"
                        )
            except Exception as exc:  # noqa: BLE001
                self.log.append(f"\n[timing_ref skipped] {exc}")
        store.save_trial_json(paths, doc)
        store.save_metrics(paths, metrics)
        self.log.append(f"Saved under Library → {paths.run_id} / {paths.trial_id}")
        self.status.setText(f"Saved → {paths.run_id} / {paths.trial_id}  (exit {code})")
        self.trialCreated.emit(paths.run_id, paths.trial_id)
