"""REW absolute FR checklist — external tool, paths only."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from app.session import REPO_ROOT, SessionSettings


CHECKLIST = """REW absolute mic/DAC FR (windows_pico hardware day)

1. Use the **same** ESS + timing-ref stimulus WAV selected on Setup (shown below).
   Do not regenerate a new sweep mid-campaign — Levels / Polar / Algorithm share it.
2. Open REW; measure dual-channel TF with timing reference.
3. Measure DAC / per-mic FR at fixed geometry (on-axis, fixed distance).
4. Export IR/FR if you want overlays in offline Python plots.
5. Do not use REW for dry_ref vs wet_out suppression — that stays in Algorithm taps.
6. On pi_hats, product FR prefers HATS capture from the Algorithm tab, not this checklist.

Full procedure: tools/tech_testbench/docs/rew_fr_day.md
Live Pi→Windows PCM is out of scope — REW runs locally with the Pico on Windows.
"""


class RewTab(QWidget):
    def __init__(self, session: SessionSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        root = QVBoxLayout(self)
        root.addWidget(QLabel("REW stays external. This tab is procedure + path reminder only."))

        self.stim_lbl = QLabel()
        self.stim_lbl.setWordWrap(True)
        root.addWidget(self.stim_lbl)

        row = QHBoxLayout()
        open_doc = QPushButton("Open rew_fr_day.md")
        open_doc.clicked.connect(self._open_doc)
        refresh = QPushButton("Refresh stimulus path")
        refresh.clicked.connect(self._refresh_paths)
        row.addWidget(open_doc)
        row.addWidget(refresh)
        row.addStretch(1)
        root.addLayout(row)

        body = QTextEdit()
        body.setReadOnly(True)
        body.setPlainText(CHECKLIST)
        root.addWidget(body)
        self._refresh_paths()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._refresh_paths()

    def _refresh_paths(self) -> None:
        stim = self.session.stimulus_path.strip() or "(not set — pick on Setup)"
        exists = Path(self.session.stimulus_path).is_file() if self.session.stimulus_path else False
        self.stim_lbl.setText(
            f"Setup stimulus WAV:\n{stim}\n"
            f"({'found' if exists else 'missing / unset'})  "
            f"profile={self.session.acoustic_profile}"
        )

    def _open_doc(self) -> None:
        doc = REPO_ROOT / "tools" / "tech_testbench" / "docs" / "rew_fr_day.md"
        if doc.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(doc)))
        else:
            self.stim_lbl.setText(f"Doc not found: {doc}")
