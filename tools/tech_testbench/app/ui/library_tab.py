"""Browse and reopen saved runs/trials (WAVs + plots + metrics)."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.session import SessionSettings
from app.storage.trial_store import TrialStore


class LibraryTab(QWidget):
    """Reload past captures: open folder, show trial.json / metrics, preview plot images."""

    trialSelected = Signal(str, str)  # run_id, trial_id

    def __init__(self, session: SessionSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session

        root = QVBoxLayout(self)
        root.addWidget(
            QLabel(
                "All captured WAVs and plot data live under Measure root via TrialStore. "
                "Select a trial to inspect or reopen the folder. UI never deletes trials."
            )
        )

        bar = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.open_btn = QPushButton("Open trial folder")
        self.open_root_btn = QPushButton("Open measure root")
        self.refresh_btn.clicked.connect(self.refresh)
        self.open_btn.clicked.connect(self._open_trial_folder)
        self.open_root_btn.clicked.connect(self._open_measure_root)
        bar.addWidget(self.refresh_btn)
        bar.addWidget(self.open_btn)
        bar.addWidget(self.open_root_btn)
        bar.addStretch(1)
        root.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_select)
        split.addWidget(self.list)

        right = QWidget()
        right_l = QVBoxLayout(right)
        self.meta = QTextEdit()
        self.meta.setReadOnly(True)
        self.plot_preview = QLabel("Plot preview")
        self.plot_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_preview.setMinimumHeight(200)
        self.plot_preview.setStyleSheet("QLabel { background: #1e1e1e; color: #aaa; }")
        right_l.addWidget(QLabel("trial.json + metrics + artifacts"))
        right_l.addWidget(self.meta, stretch=2)
        right_l.addWidget(QLabel("First plot image (if any)"))
        right_l.addWidget(self.plot_preview, stretch=1)
        split.addWidget(right)
        split.setSizes([360, 640])
        root.addWidget(split, stretch=1)

        self._selected: tuple[str, str] | None = None
        self.refresh()

    def _store(self) -> TrialStore:
        return TrialStore(self.session.measure_root)

    def refresh(self) -> None:
        prev = self._selected
        self.list.clear()
        store = self._store()
        active = self.session.active_run_id
        for run_id, trial_id, doc in store.list_all_trials():
            kind = doc.get("kind", "trial")
            profile = doc.get("acoustic_profile", "?")
            created = str(doc.get("created_utc", ""))[:19]
            marker = "*" if active and run_id == active else " "
            label = f"{marker} {run_id} / {trial_id}  [{kind}]  {profile}  {created}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, (run_id, trial_id))
            self.list.addItem(item)
        if self.list.count() == 0:
            self.meta.setPlainText(f"No trials yet under:\n{self.session.measure_root}")
            self.plot_preview.setText("No plots")
            self.plot_preview.setPixmap(QPixmap())
            self._selected = None
            return
        if prev:
            self.select_trial(*prev)

    def select_trial(self, run_id: str, trial_id: str) -> None:
        """Select a trial in the list after a capture (if still present)."""
        for i in range(self.list.count()):
            item = self.list.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            if data == (run_id, trial_id):
                self.list.setCurrentItem(item)
                return

    def _on_select(self, cur: QListWidgetItem | None, _prev: QListWidgetItem | None) -> None:
        if cur is None:
            self._selected = None
            return
        data = cur.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        run_id, trial_id = data
        self._selected = (run_id, trial_id)
        self.trialSelected.emit(run_id, trial_id)
        store = self._store()
        paths = store.trial_paths(run_id, trial_id)
        try:
            trial = store.load_trial_json(paths)
            metrics = store.load_metrics(paths)
        except OSError as exc:
            self.meta.setPlainText(str(exc))
            return
        # Re-index any WAVs dropped into the folder outside the GUI
        try:
            store.scan_wavs(paths)
            trial = store.load_trial_json(paths)
        except OSError:
            pass
        wavs = sorted(p.name for p in paths.root.glob("*.wav"))
        extras = sorted(
            p.name
            for p in paths.root.iterdir()
            if p.is_file() and p.suffix.lower() not in {".wav", ".json"}
        )
        plots = []
        if paths.plots_dir.is_dir():
            plots = sorted(
                str(p.relative_to(paths.root)).replace("\\", "/")
                for p in paths.plots_dir.iterdir()
                if p.is_file()
            )
        view = {
            "path": str(paths.root),
            "trial": trial,
            "metrics": metrics,
            "wav_files_on_disk": wavs,
            "other_files_on_disk": extras,
            "plot_files_on_disk": plots,
        }
        self.meta.setPlainText(json.dumps(view, indent=2))

        preview = None
        for p in paths.plots_dir.glob("*") if paths.plots_dir.is_dir() else []:
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                preview = p
                break
        if preview and preview.is_file():
            pix = QPixmap(str(preview))
            if not pix.isNull():
                self.plot_preview.setPixmap(
                    pix.scaled(
                        self.plot_preview.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                self.plot_preview.setText(str(preview))
        else:
            self.plot_preview.setPixmap(QPixmap())
            self.plot_preview.setText("No plot image in plots/")

    def _open_trial_folder(self) -> None:
        if not self._selected:
            QMessageBox.information(self, "Library", "Select a trial first.")
            return
        run_id, trial_id = self._selected
        folder = self._store().trial_paths(run_id, trial_id).root
        if not folder.is_dir():
            QMessageBox.warning(self, "Missing", str(folder))
            return
        QDesktopServices.openUrl(folder.as_uri())

    def _open_measure_root(self) -> None:
        root = Path(self.session.measure_root)
        root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(root.as_uri())
