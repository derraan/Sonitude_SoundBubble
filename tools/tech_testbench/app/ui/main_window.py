"""Top-level window — same tab pattern as the Sonitude algorithm testbench."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QToolBar,
)

from app.pipeline_session import PipelineSession, PipelineState, launch_from_session
from app.session import SessionSettings
from app.storage.trial_store import TrialStore
from app.ui.algorithm_tab import AlgorithmTab
from app.ui.e2e_latency_tab import E2ELatencyTab
from app.ui.levels_tab import LevelsTab
from app.ui.library_tab import LibraryTab
from app.ui.parity_tab import ParityTab
from app.ui.polar_tab import PolarTab
from app.ui.rew_tab import RewTab
from app.ui.setup_tab import SetupTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sound Bubble Tech Testbench")
        self.resize(1100, 780)
        self.session = SessionSettings()
        self.pipeline = PipelineSession()
        self.pipeline.add_listener(self._on_pipeline_changed)

        tb = QToolBar("Run")
        tb.addAction("New run", self._new_run)
        self._start_pipe_act = tb.addAction("Start Pipeline", self._start_pipeline)
        self._stop_pipe_act = tb.addAction("Stop Pipeline", self._stop_pipeline)
        self._stop_pipe_act.setEnabled(False)
        self.addToolBar(tb)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._run_lbl = QLabel("Active run: (none — first trial creates one)")
        self._pipe_lbl = QLabel("Pipeline: Stopped")
        self._status.addWidget(self._run_lbl, stretch=1)
        self._status.addPermanentWidget(self._pipe_lbl)

        tabs = QTabWidget()
        self.setup_tab = SetupTab(self.session)
        self.levels_tab = LevelsTab(self.session)
        self.polar_tab = PolarTab(self.session, pipeline=self.pipeline)
        self.algorithm_tab = AlgorithmTab(
            self.session, self.setup_tab.validate_basic, pipeline=self.pipeline
        )
        self.e2e_latency_tab = E2ELatencyTab(
            self.session, self.setup_tab.validate_basic, pipeline=self.pipeline
        )
        self.parity_tab = ParityTab(self.session)
        self.rew_tab = RewTab(self.session)
        self.library_tab = LibraryTab(self.session)

        tabs.addTab(self.setup_tab, "Setup")
        tabs.addTab(self.levels_tab, "Levels")
        tabs.addTab(self.polar_tab, "Polar")
        tabs.addTab(self.algorithm_tab, "Algorithm")
        tabs.addTab(self.e2e_latency_tab, "E2E latency")
        tabs.addTab(self.parity_tab, "Parity")
        tabs.addTab(self.rew_tab, "REW FR")
        tabs.addTab(self.library_tab, "Library")
        self.setCentralWidget(tabs)
        self._tabs = tabs

        self.setup_tab.settingsChanged.connect(self._on_settings_changed)
        # Setup's campaign combo is the single source of truth (it owns the stimulus
        # auto-fill: ESS file <-> speech folder). Algorithm-tab changes drive it.
        self.algorithm_tab.campaign.currentIndexChanged.connect(self._on_algorithm_campaign_changed)
        for tab in (
            self.levels_tab,
            self.polar_tab,
            self.algorithm_tab,
            self.e2e_latency_tab,
            self.parity_tab,
        ):
            tab.trialCreated.connect(self._on_trial_created)

        self._pipe_timer = QTimer(self)
        self._pipe_timer.setInterval(1000)
        self._pipe_timer.timeout.connect(self._tick_pipeline)
        self._pipe_timer.start()

        self._refresh_run_label()
        self._on_pipeline_changed()

    def _refresh_run_label(self) -> None:
        rid = self.session.active_run_id or "(none — first trial creates one)"
        self._run_lbl.setText(
            f"Active run: {rid}  |  measure root: {self.session.measure_root}  |  "
            f"profile: {self.session.acoustic_profile}"
        )

    def _on_pipeline_changed(self) -> None:
        st = self.pipeline.state
        self._pipe_lbl.setText(f"Pipeline: {self.pipeline.status_text}")
        running = st not in (PipelineState.STOPPED, PipelineState.FAULT) or self.pipeline.is_alive()
        starting = st in (PipelineState.STARTING, PipelineState.WARMING, PipelineState.READY)
        self._start_pipe_act.setEnabled(st in (PipelineState.STOPPED, PipelineState.FAULT))
        self._stop_pipe_act.setEnabled(starting or running)

    def _tick_pipeline(self) -> None:
        if self.pipeline.state in (PipelineState.READY, PipelineState.WARMING):
            self.pipeline.refresh_status_from_profile()
            self._on_pipeline_changed()

    def _start_pipeline(self) -> None:
        self.setup_tab.apply_to_session()
        errs = self.setup_tab.validate_basic()
        if self.session.pipeline_out_device is None:
            errs.append("Pipeline out (Pico DAC / earcups) is required to Start Pipeline.")
        if self.session.runtime == "cpp" and not self.session.soundbubble_rt_path:
            errs.append("Setup → soundbubble-rt path is empty.")
        errs.extend(self.setup_tab.validate_pipeline_audio())
        if errs:
            QMessageBox.warning(self, "Cannot start pipeline", "\n".join(errs))
            return
        try:
            launch = launch_from_session(self.session)
            self.pipeline.start(launch)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Start Pipeline failed", str(exc))
            return
        self._on_pipeline_changed()
        # Non-blocking: a modal here used to sit on screen while warm already Faulted.
        self.statusBar().showMessage(
            f"Pipeline warming (in={self.session.pipeline_in_device}@"
            f"{self.session.input_sr} Hz, out={self.session.pipeline_out_device}@"
            f"{self.session.output_sr} Hz). Wait for Ready before takes.",
            8000,
        )

    def _stop_pipeline(self) -> None:
        try:
            rc = self.pipeline.stop(grace_s=4.0)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Stop Pipeline failed", str(exc))
            return
        self._on_pipeline_changed()
        QMessageBox.information(self, "Pipeline stopped", f"Exit code: {rc}")

    def _on_settings_changed(self) -> None:
        self._refresh_run_label()
        self.library_tab.refresh()
        self.algorithm_tab._refresh_gains_label()  # noqa: SLF001 — keep Setup gains visible
        self.e2e_latency_tab._refresh_gains_label()  # noqa: SLF001
        # Align Algorithm campaign combo with Setup campaign_kind.
        idx = self.algorithm_tab.campaign.findData(self.session.campaign_kind)
        if idx >= 0 and self.algorithm_tab.campaign.currentIndex() != idx:
            self.algorithm_tab.campaign.blockSignals(True)
            self.algorithm_tab.campaign.setCurrentIndex(idx)
            self.algorithm_tab.campaign.blockSignals(False)

    def _on_algorithm_campaign_changed(self) -> None:
        kind = self.algorithm_tab.campaign.currentData()
        idx = self.setup_tab.campaign.findData(kind)
        if idx >= 0 and self.setup_tab.campaign.currentIndex() != idx:
            # Triggers SetupTab._on_campaign_changed -> stimulus auto-fill -> apply_to_session.
            self.setup_tab.campaign.setCurrentIndex(idx)

    def _on_trial_created(self, run_id: str, trial_id: str) -> None:
        self._refresh_run_label()
        self.library_tab.refresh()
        self.library_tab.select_trial(run_id, trial_id)

    def _new_run(self) -> None:
        store = TrialStore(self.session.measure_root)
        run_id = self.session.start_new_run(store)
        self._refresh_run_label()
        self.library_tab.refresh()
        QMessageBox.information(
            self,
            "New run",
            f"New run folder:\n{store.measure_root / run_id}\n\n"
            "Following trials append here until you click New run again.\n\n"
            "Live Pi→Windows PCM is out of scope; copy measure trees with scp.",
        )
        self._tabs.setCurrentWidget(self.library_tab)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.pipeline.is_alive():
            try:
                self.pipeline.stop(grace_s=3.0)
            except Exception:
                pass
        super().closeEvent(event)
