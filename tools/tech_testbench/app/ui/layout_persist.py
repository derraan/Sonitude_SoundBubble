"""QSettings helpers (mirrors Sonitude testbench layout_persist)."""

from __future__ import annotations

from PySide6.QtCore import QSettings

ORG = "Sonitude"
APP = "SoundBubbleTechTestbench"


def settings() -> QSettings:
    return QSettings(ORG, APP)
