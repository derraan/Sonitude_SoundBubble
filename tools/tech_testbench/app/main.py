"""Entrypoint for the Sound Bubble tech testbench GUI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow `python -m app.main` from tools/tech_testbench/
_ROOT = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[3]  # Sound_Bubble/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from app.ui.layout_persist import APP, ORG
from app.ui.main_window import MainWindow


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    QCoreApplication.setOrganizationName(ORG)
    QCoreApplication.setApplicationName(APP)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
