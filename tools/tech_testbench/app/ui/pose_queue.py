"""Discrete pose queues for Algorithm / Polar campaign grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class Pose:
    distance_m: float
    azimuth_deg: float
    elevation_deg: float
    label: str = ""

    def display(self) -> str:
        if self.label:
            return self.label
        return (
            f"d={self.distance_m:.2f} m  az={self.azimuth_deg:.0f}°  "
            f"el={self.elevation_deg:.0f}°"
        )


def _az_steps() -> list[float]:
    return [float(a) for a in range(0, 360, 30)]  # 0..330


def _el_steps() -> list[float]:
    return [60.0, 30.0, 0.0, -30.0, -60.0]


def _part_b_distances() -> list[float]:
    return [0.5, 1.0, 2.0]


def _in_context_distances() -> list[float]:
    return [0.5, 1.0, 1.5, 2.0]


def _in_context_elev_dists() -> list[float]:
    return [0.5, 1.0, 1.5, 2.0]


PRESET_IDS = (
    ("part_b_az", "Part B az 0…330 @ 30° (d=1.0 m)"),
    ("part_b_el", "Part B el ±60 @ 30° (az=0, d=1.0 m)"),
    ("part_b_dist", "Part B distances 0.5 / 1.0 / 2.0 m"),
    ("ctx_dist", "In-context distances 0.5 / 1.0 / 1.5 / 2.0 m"),
    ("ctx_elev_dist", "In-context elev × dist"),
)


def build_preset(preset_id: str, *, reverse_distance: bool = False) -> list[Pose]:
    """Build pose list. ``reverse_distance`` reverses distance-ordered presets (out_of_bubble)."""
    if preset_id == "part_b_az":
        return [
            Pose(1.0, az, 0.0, label=f"Part B az {az:.0f}° @ 1.0 m")
            for az in _az_steps()
        ]
    if preset_id == "part_b_el":
        return [
            Pose(1.0, 0.0, el, label=f"Part B el {el:+.0f}° @ 1.0 m")
            for el in _el_steps()
        ]
    if preset_id == "part_b_dist":
        dists = list(_part_b_distances())
        if reverse_distance:
            dists.reverse()
        return [
            Pose(d, 0.0, 0.0, label=f"Part B d={d:.1f} m")
            for d in dists
        ]
    if preset_id == "ctx_dist":
        dists = list(_in_context_distances())
        if reverse_distance:
            dists.reverse()
        return [
            Pose(d, 0.0, 0.0, label=f"In-context d={d:.1f} m")
            for d in dists
        ]
    if preset_id == "ctx_elev_dist":
        poses: list[Pose] = []
        dists = list(_in_context_elev_dists())
        if reverse_distance:
            dists.reverse()
        for d in dists:
            for el in _el_steps():
                poses.append(
                    Pose(
                        d,
                        0.0,
                        el,
                        label=f"In-context d={d:.1f} m el={el:+.0f}°",
                    )
                )
        return poses
    raise ValueError(f"unknown pose preset: {preset_id}")


class PoseQueuePanel(QWidget):
    """Operator checklist: load preset → apply current → mark done after capture."""

    def __init__(
        self,
        *,
        apply_pose: Callable[[Pose], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._apply_pose = apply_pose
        self._poses: list[Pose] = []
        self._index: int = 0
        self._done: set[int] = set()

        box = QGroupBox("Pose queue (discrete grid)")
        layout = QVBoxLayout(box)
        row = QHBoxLayout()
        self.preset = QComboBox()
        for pid, label in PRESET_IDS:
            self.preset.addItem(label, pid)
        self.load_btn = QPushButton("Load preset")
        self.load_btn.clicked.connect(self._load)
        self.apply_btn = QPushButton("Apply current pose")
        self.apply_btn.clicked.connect(self._apply_current)
        self.done_btn = QPushButton("Mark done → next")
        self.done_btn.clicked.connect(self._mark_done_advance)
        row.addWidget(self.preset, stretch=1)
        row.addWidget(self.load_btn)
        row.addWidget(self.apply_btn)
        row.addWidget(self.done_btn)
        layout.addLayout(row)

        self.status = QLabel("No queue loaded.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.list = QListWidget()
        self.list.setMaximumHeight(140)
        self.list.currentRowChanged.connect(self._on_row)
        layout.addWidget(self.list)

        hint = QLabel(
            "into_bubble / out_of_bubble on Algorithm reverses distance-ordered presets. "
            "After a successful capture, Mark done → next (or auto-advance if wired)."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)

        self._reverse_distance = False

    def set_reverse_distance(self, reverse: bool) -> None:
        self._reverse_distance = bool(reverse)

    def _load(self) -> None:
        pid = str(self.preset.currentData())
        self._poses = build_preset(pid, reverse_distance=self._reverse_distance)
        self._index = 0
        self._done.clear()
        self.list.clear()
        for i, pose in enumerate(self._poses):
            self.list.addItem(QListWidgetItem(f"[ ] {i + 1}. {pose.display()}"))
        self._refresh_status()
        if self._poses:
            self.list.setCurrentRow(0)
            self._apply_current()

    def _on_row(self, row: int) -> None:
        if 0 <= row < len(self._poses):
            self._index = row
            self._refresh_status()

    def _refresh_status(self) -> None:
        n = len(self._poses)
        if n == 0:
            self.status.setText("No queue loaded.")
            return
        pose = self._poses[self._index]
        done_n = len(self._done)
        self.status.setText(
            f"{self._index + 1}/{n}  ({done_n} done) — move to: {pose.display()}"
        )
        for i in range(self.list.count()):
            item = self.list.item(i)
            mark = "x" if i in self._done else " "
            item.setText(f"[{mark}] {i + 1}. {self._poses[i].display()}")

    def _apply_current(self) -> None:
        if not self._poses:
            return
        pose = self._poses[self._index]
        self._apply_pose(pose)
        self._refresh_status()

    def _mark_done_advance(self) -> None:
        if not self._poses:
            return
        self._done.add(self._index)
        nxt = self._index + 1
        if nxt < len(self._poses):
            self._index = nxt
            self.list.setCurrentRow(self._index)
            self._apply_current()
        else:
            self._refresh_status()
            self.status.setText(self.status.text() + "  — queue complete.")

    def on_trial_captured(self) -> None:
        """Call after a successful capture to mark current pose done and advance."""
        if self._poses:
            self._mark_done_advance()

    def current_pose(self) -> Pose | None:
        if not self._poses:
            return None
        return self._poses[self._index]
