"""Setup: acoustic profile, devices, gains, runtime, model, stimulus."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.devices import (
    DeviceInfo,
    device_by_index,
    format_device,
    list_input_devices,
    list_output_devices,
    suggested_pipeline_sr,
    validate_pipeline_rates,
)
from app.session import (
    MIC_CAL_LEVELS_DBFS,
    MODEL_LEVEL_DBFS,
    REPO_ROOT,
    Gains,
    SessionSettings,
    normalized_mic_gains,
)

DEFAULT_SPEECH_DIR = REPO_ROOT / "datasets" / "LibriSpeech" / "dev-clean"
MIC_GAIN_MIN = 0.01
MIC_GAIN_MAX = 20.0
MIC_SLIDER_SCALE = 100  # slider ticks per x1.0
from app.ui.layout_persist import settings as qsettings


class SetupTab(QWidget):
    settingsChanged = Signal()

    def __init__(self, session: SessionSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self._inputs: list[DeviceInfo] = []
        self._outputs: list[DeviceInfo] = []

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()
        root.addLayout(left, 1)
        root.addLayout(right, 1)

        profile_box = QGroupBox("Acoustic profile")
        profile_form = QFormLayout(profile_box)
        self.profile = QComboBox()
        self.profile.addItem("windows_pico — Pico DAC stimulus / direct array", "windows_pico")
        self.profile.addItem("pi_hats — speakers → earcups → sim-head → interface", "pi_hats")
        self.host = QComboBox()
        self.host.addItem("local_windows", "local_windows")
        self.host.addItem("pi_rig", "pi_rig")
        self.campaign = QComboBox()
        self.campaign.addItem("controlled_ess — shared REW ESS", "controlled_ess")
        self.campaign.addItem("in_context_speech — speech loop WAV", "in_context_speech")
        profile_form.addRow("Profile", self.profile)
        profile_form.addRow("Host target", self.host)
        profile_form.addRow("Campaign", self.campaign)
        left.addWidget(profile_box)

        dev_box = QGroupBox("Devices")
        dev_form = QFormLayout(dev_box)
        self.stimulus_out = QComboBox()
        self.pipeline_in = QComboBox()
        self.pipeline_out = QComboBox()
        self.hats_in = QComboBox()
        refresh = QPushButton("Refresh devices")
        refresh.clicked.connect(self.refresh_devices)
        for combo in (self.stimulus_out, self.pipeline_in, self.pipeline_out, self.hats_in):
            combo.setMinimumContentsLength(40)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        dev_form.addRow("Stimulus out", self.stimulus_out)
        dev_form.addRow("Pipeline in (Pico mics)", self.pipeline_in)
        dev_form.addRow("Pipeline out (Pico DAC / earcups)", self.pipeline_out)
        dev_form.addRow("HATS in (audio interface)", self.hats_in)
        dev_form.addRow("", refresh)
        self.hats_hint = QLabel(
            "Ear-rig / HATS stereo input (always opened as 2ch L/R; "
            "optional on windows_pico; required on pi_hats)."
        )
        self.hats_hint.setWordWrap(True)
        dev_form.addRow(self.hats_hint)
        left.addWidget(dev_box)

        io_box = QGroupBox("Class B I/O")
        io_form = QFormLayout(io_box)
        self.input_sr = QSpinBox()
        self.input_sr.setRange(8000, 96000)
        self.input_sr.setSingleStep(1000)
        self.input_sr.setValue(session.input_sr)
        self.output_sr = QSpinBox()
        self.output_sr.setRange(8000, 96000)
        self.output_sr.setSingleStep(1000)
        self.output_sr.setValue(session.output_sr)
        self.channel_map = QLineEdit(session.channel_map)
        self.channel_map.setPlaceholderText("5,4,3,2,1,0")
        self.sr_hint = QLabel(
            "Pipeline SR must match the Pico device native rate (INMP441 WDM-KS = 24000 Hz only). "
            "Wrong SR often fails Start Pipeline as PortAudio 'Invalid device'."
        )
        self.sr_hint.setWordWrap(True)
        self.channel_map_hint = QLabel(
            "Default is edge/MIC_INDEX_CANON.md §4 (5,4,3,2,1,0). "
            "Override only if the headset wiring changes."
        )
        self.channel_map_hint.setWordWrap(True)
        io_form.addRow("Input SR", self.input_sr)
        io_form.addRow("Output SR", self.output_sr)
        io_form.addRow("", self.sr_hint)
        io_form.addRow("Channel map", self.channel_map)
        io_form.addRow("", self.channel_map_hint)
        self._snap_sr_btn = QPushButton("Snap SR to selected Pico devices")
        self._snap_sr_btn.clicked.connect(self.snap_pipeline_sr_to_devices)
        io_form.addRow("", self._snap_sr_btn)
        left.addWidget(io_box)
        left.addStretch(1)

        gain_box = QGroupBox("Gains (pipeline)")
        gain_form = QFormLayout(gain_box)
        g = session.gains
        self.input_gain = QDoubleSpinBox()
        self.input_gain.setRange(-60.0, 60.0)
        self.input_gain.setDecimals(2)
        self.input_gain.setSuffix(" dB")
        self.input_gain.setValue(float(g.input_gain_db))
        self.output_gain = QDoubleSpinBox()
        self.output_gain.setRange(-60.0, 60.0)
        self.output_gain.setDecimals(2)
        self.output_gain.setSuffix(" dB")
        self.output_gain.setValue(float(g.output_gain_db))
        self.mix_dry = QDoubleSpinBox()
        self.mix_dry.setRange(0.0, 1.0)
        self.mix_dry.setSingleStep(0.05)
        self.mix_dry.setDecimals(3)
        self.mix_dry.setValue(float(g.mix_dry))
        self.limiter = QCheckBox("Output limiter enabled")
        self.limiter.setChecked(bool(g.output_limiter))
        self.limiter_release = QDoubleSpinBox()
        self.limiter_release.setRange(1.0, 500.0)
        self.limiter_release.setSuffix(" ms")
        self.limiter_release.setValue(float(g.limiter_release_ms))
        gain_form.addRow("Input gain", self.input_gain)
        gain_form.addRow("Output gain", self.output_gain)
        gain_form.addRow("Mix dry", self.mix_dry)
        gain_form.addRow(self.limiter)
        gain_form.addRow("Limiter release", self.limiter_release)
        right.addWidget(gain_box)

        # Per-mic linear gains by DEVICE channel (before --channel-map, cpp runtime).
        # Fader + box per mic; values are linear multipliers (x1.0 = unity, always > 0).
        mic_box = QGroupBox("Mic gains (linear, device channel)")
        mic_v = QVBoxLayout(mic_box)
        faders = QGridLayout()
        self.mic_gain_sliders: list[QSlider] = []
        self.mic_gain_boxes: list[QDoubleSpinBox] = []
        self.mic_gain_db_labels: list[QLabel] = []
        cur = [float(v) for v in (getattr(g, "mic_gains", None) or normalized_mic_gains())]
        cur = (cur + [1.0] * 6)[:6]
        for ch in range(6):
            head = QLabel(f"ch{ch}")
            head.setAlignment(Qt.AlignHCenter)
            sl = QSlider(Qt.Vertical)
            sl.setRange(int(MIC_GAIN_MIN * MIC_SLIDER_SCALE), int(MIC_GAIN_MAX * MIC_SLIDER_SCALE))
            sl.setTickPosition(QSlider.TicksBothSides)
            sl.setTickInterval(int(1.0 * MIC_SLIDER_SCALE))
            sl.setMinimumHeight(140)
            sl.setToolTip(f"Device input channel {ch} linear gain")
            sb = QDoubleSpinBox()
            sb.setRange(MIC_GAIN_MIN, MIC_GAIN_MAX)
            sb.setDecimals(3)
            sb.setSingleStep(0.01)
            sb.setPrefix("x")
            sb.setToolTip(f"Device input channel {ch} linear gain (x1.000 = unity)")
            db = QLabel()
            db.setAlignment(Qt.AlignHCenter)
            sl.valueChanged.connect(lambda v, box=sb: box.setValue(v / MIC_SLIDER_SCALE))
            sb.valueChanged.connect(lambda v, slider=sl, lab=db: self._sync_mic_fader(slider, lab, v))
            sb.setValue(cur[ch])
            self._sync_mic_fader(sl, db, cur[ch])
            faders.addWidget(head, 0, ch, Qt.AlignHCenter)
            faders.addWidget(sl, 1, ch, Qt.AlignHCenter)
            faders.addWidget(sb, 2, ch)
            faders.addWidget(db, 3, ch)
            self.mic_gain_sliders.append(sl)
            self.mic_gain_boxes.append(sb)
            self.mic_gain_db_labels.append(db)
        mic_v.addLayout(faders)
        norm_row = QHBoxLayout()
        norm_row.addWidget(QLabel("Model level"))
        self.model_level = QDoubleSpinBox()
        self.model_level.setRange(-60.0, 0.0)
        self.model_level.setDecimals(1)
        self.model_level.setSuffix(" dBFS")
        self.model_level.setValue(float(getattr(g, "model_level_dbfs", MODEL_LEVEL_DBFS)))
        self.model_level.setToolTip("Level each mic is normalized to (pre input gain). Bench default -19 dBFS + 15 dB input gain")
        norm_row.addWidget(self.model_level)
        self._mic_norm_btn = QPushButton("Normalize to model")
        self._mic_norm_btn.setToolTip(
            "Set each mic gain so its calibration level lands at the model level (input gain is applied on top). "
            "Cal levels (dBFS, dev ch0..5): "
            + ", ".join(f"{v:g}" for v in MIC_CAL_LEVELS_DBFS)
        )
        self._mic_norm_btn.clicked.connect(self._normalize_mic_gains)
        norm_row.addWidget(self._mic_norm_btn)
        self._mic_unity_btn = QPushButton("Unity")
        self._mic_unity_btn.clicked.connect(lambda: self._set_mic_gains([1.0] * 6))
        norm_row.addWidget(self._mic_unity_btn)
        norm_row.addStretch(1)
        mic_v.addLayout(norm_row)
        right.addWidget(mic_box)

        rt_box = QGroupBox("ONNX realtime pipeline")
        rt_form = QFormLayout(rt_box)
        self.runtime = QComboBox()
        self.runtime.addItem("cpp — soundbubble-rt", "cpp")
        self.runtime.addItem("python — inference_pipeline", "python")
        self.rt_path = QLineEdit(session.soundbubble_rt_path)
        self.rt_path.setPlaceholderText("path to soundbubble-rt.exe (cpp)")
        browse_rt = QPushButton("…")
        browse_rt.clicked.connect(self._browse_rt)
        rt_row = QHBoxLayout()
        rt_row.addWidget(self.rt_path)
        rt_row.addWidget(browse_rt)
        self.use_zoo = QCheckBox("Use zoo dir + bubble radius")
        self.use_zoo.setChecked(bool(session.use_zoo))
        self.zoo_dir = QLineEdit(session.zoo_dir)
        self.model_path = QLineEdit(session.model_path)
        self.contract_path = QLineEdit(session.contract_path)
        self.bubble = QDoubleSpinBox()
        self.bubble.setRange(0.1, 10.0)
        self.bubble.setSingleStep(0.1)
        self.bubble.setValue(session.bubble_radius_m)
        self.bubble.setSuffix(" m")
        rt_form.addRow("Runtime", self.runtime)
        rt_form.addRow("soundbubble-rt", rt_row)
        rt_form.addRow(self.use_zoo)
        rt_form.addRow("Zoo dir", self.zoo_dir)
        rt_form.addRow("Bubble radius", self.bubble)
        rt_form.addRow("Model path", self.model_path)
        rt_form.addRow("Contract path", self.contract_path)
        right.addWidget(rt_box)

        paths_box = QGroupBox("Stimulus & output")
        paths_form = QFormLayout(paths_box)
        self.stimulus = QLineEdit(session.stimulus_path)
        self.stimulus.setPlaceholderText("One shared REW ESS + timing-ref WAV (all tabs)")
        browse_stim = QPushButton("…")
        browse_stim.clicked.connect(self._browse_stimulus)
        stim_row = QHBoxLayout()
        stim_row.addWidget(self.stimulus)
        stim_row.addWidget(browse_stim)
        self.stimulus_meta = QLabel("Stimulus: (not set)")
        self.stimulus_meta.setWordWrap(True)
        self.stimulus_repeat = QSpinBox()
        self.stimulus_repeat.setRange(1, 100)
        self.stimulus_repeat.setValue(max(1, int(session.stimulus_repeat)))
        self.stimulus_repeat.setToolTip(
            "Play the WAV N times per take (speech loops). Keep 1 for ESS characterization."
        )
        self.stimulus_play_ch = QComboBox()
        self.stimulus_play_ch.addItem("Left channel only", "left")
        self.stimulus_play_ch.addItem("Right channel only", "right")
        self.stimulus_play_ch.addItem("Both", "both")
        # Prefer Left for shared ESS (*_refL); fall back to session / Both.
        pref = (session.stimulus_play_channels or "left").strip().lower()
        idx = self.stimulus_play_ch.findData(pref if pref in ("left", "right", "both") else "left")
        self.stimulus_play_ch.setCurrentIndex(idx if idx >= 0 else 0)
        self.stimulus_play_ch.setToolTip(
            "Which stereo output legs get the stimulus. "
            "ESS files named *_refL need Left (or Both) for the timing reference."
        )
        self.take_len_lbl = QLabel("Take length: —")
        self.take_len_lbl.setWordWrap(True)
        self.measure_root = QLineEdit(session.measure_root)
        browse_out = QPushButton("…")
        browse_out.clicked.connect(self._browse_measure_root)
        out_row = QHBoxLayout()
        out_row.addWidget(self.measure_root)
        out_row.addWidget(browse_out)
        paths_form.addRow("Stimulus WAV", stim_row)
        paths_form.addRow("", self.stimulus_meta)
        paths_form.addRow("Stimulus repeat", self.stimulus_repeat)
        paths_form.addRow("Play channel", self.stimulus_play_ch)
        paths_form.addRow("", self.take_len_lbl)
        paths_form.addRow("Measure root", out_row)
        right.addWidget(paths_box)
        right.addStretch(1)

        self.profile.currentIndexChanged.connect(self._on_profile_changed)
        self.campaign.currentIndexChanged.connect(self._on_campaign_changed)
        for w in (
            self.profile,
            self.host,
            self.campaign,
            self.stimulus_out,
            self.pipeline_in,
            self.pipeline_out,
            self.hats_in,
            self.runtime,
            self.stimulus_play_ch,
            self.use_zoo,
            self.limiter,
        ):
            if isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self.apply_to_session)
            else:
                w.toggled.connect(self.apply_to_session)
        for w in (
            self.input_sr,
            self.output_sr,
            self.input_gain,
            self.output_gain,
            self.mix_dry,
            self.limiter_release,
            self.bubble,
            self.stimulus_repeat,
            *self.mic_gain_boxes,
            self.model_level,
        ):
            w.valueChanged.connect(self.apply_to_session)
        for w in (
            self.channel_map,
            self.rt_path,
            self.zoo_dir,
            self.model_path,
            self.contract_path,
            self.stimulus,
            self.measure_root,
        ):
            w.editingFinished.connect(self.apply_to_session)

        self._load_persisted_paths()
        self.refresh_devices()
        self._on_profile_changed()
        self.apply_to_session()

    def _load_persisted_paths(self) -> None:
        s = qsettings()
        stim = s.value("paths/stimulus", "")
        root = s.value("paths/measure_root", "")
        rt = s.value("paths/soundbubble_rt", "")
        cmap = s.value("paths/channel_map", "")
        if isinstance(stim, str) and stim:
            self.stimulus.setText(stim)
        if isinstance(root, str) and root:
            self.measure_root.setText(root)
        if isinstance(rt, str) and rt:
            self.rt_path.setText(rt)
        if isinstance(cmap, str) and cmap:
            self.channel_map.setText(cmap)
        # Device indices: prefer last session, else SessionSettings defaults.
        for key, attr in (
            ("devices/stimulus_out", "stimulus_out_device"),
            ("devices/pipeline_in", "pipeline_in_device"),
            ("devices/pipeline_out", "pipeline_out_device"),
            ("devices/hats_in", "hats_in_device"),
        ):
            raw = s.value(key, None)
            if raw is None or raw == "":
                continue
            try:
                setattr(self.session, attr, int(raw))
            except (TypeError, ValueError):
                pass
        for key, widget in (
            ("io/input_sr", self.input_sr),
            ("io/output_sr", self.output_sr),
        ):
            raw = s.value(key, None)
            if raw is None or raw == "":
                continue
            try:
                widget.setValue(int(raw))
            except (TypeError, ValueError):
                pass

    def _persist_paths(self) -> None:
        s = qsettings()
        s.setValue("paths/stimulus", self.stimulus.text().strip())
        s.setValue("paths/measure_root", self.measure_root.text().strip())
        s.setValue("paths/soundbubble_rt", self.rt_path.text().strip())
        s.setValue("paths/channel_map", self.channel_map.text().strip())
        s.setValue("io/input_sr", int(self.input_sr.value()))
        s.setValue("io/output_sr", int(self.output_sr.value()))
        for key, val in (
            ("devices/stimulus_out", self.stimulus_out.currentData()),
            ("devices/pipeline_in", self.pipeline_in.currentData()),
            ("devices/pipeline_out", self.pipeline_out.currentData()),
            ("devices/hats_in", self.hats_in.currentData()),
        ):
            if val is None:
                s.remove(key)
            else:
                s.setValue(key, int(val))

    def refresh_devices(self) -> None:
        self._inputs = list_input_devices()
        self._outputs = list_output_devices()
        self._fill_combo(
            self.stimulus_out,
            self._outputs,
            allow_none=True,
            preferred=self.session.stimulus_out_device,
            name_needles=("1-2", "QUAD-CAPTURE"),
            prefer_hostapi="WDM-KS",
        )
        self._fill_combo(
            self.pipeline_in,
            self._inputs,
            allow_none=True,
            preferred=self.session.pipeline_in_device,
            name_needles=("INMP441",),
            prefer_hostapi="WDM-KS",
            min_in_ch=8,
        )
        self._fill_combo(
            self.pipeline_out,
            self._outputs,
            allow_none=True,
            preferred=self.session.pipeline_out_device,
            name_needles=("INMP441",),
            prefer_hostapi="WDM-KS",
        )
        self._fill_combo(
            self.hats_in,
            self._inputs,
            allow_none=True,
            preferred=self.session.hats_in_device,
            name_needles=("1-2", "QUAD-CAPTURE"),
            # DirectSound delivers real stereo L/R on QUAD 1-2; WDM-KS/WASAPI
            # often open as 2ch with a permanently silent left channel.
            prefer_hostapi="DirectSound",
        )
        self.snap_pipeline_sr_to_devices(silent=True)
        self.apply_to_session()

    def snap_pipeline_sr_to_devices(self, silent: bool = False) -> None:
        """Force pipeline SR to the selected Pico devices' native rates."""
        in_dev = device_by_index(self.pipeline_in.currentData())
        out_dev = device_by_index(self.pipeline_out.currentData())
        in_sr = suggested_pipeline_sr(in_dev, int(self.input_sr.value()))
        out_sr = suggested_pipeline_sr(out_dev, int(self.output_sr.value()))
        # Keep matched rates on this bench when both ends are Pico.
        if in_dev is not None and out_dev is not None and in_sr != out_sr:
            # Prefer input native if both are INMP441; else input wins.
            out_sr = in_sr
        changed = in_sr != int(self.input_sr.value()) or out_sr != int(self.output_sr.value())
        self.input_sr.setValue(int(in_sr))
        self.output_sr.setValue(int(out_sr))
        if in_dev is not None or out_dev is not None:
            self.sr_hint.setText(
                f"Pipeline SR → in={in_sr} Hz"
                + (f" ({in_dev.name})" if in_dev else "")
                + f", out={out_sr} Hz"
                + (f" ({out_dev.name})" if out_dev else "")
                + ". Pico WDM-KS is 24000-only; wrong SR → 'Invalid device'."
            )
        if changed and not silent:
            self.apply_to_session()

    @staticmethod
    def _device_matches(d: DeviceInfo, needles: tuple[str, ...] | None) -> bool:
        if not needles:
            return True
        name = d.name.lower()
        return all(n.lower() in name for n in needles)

    @classmethod
    def _resolve_device_index(
        cls,
        devices: list[DeviceInfo],
        preferred: int | None,
        *,
        name_needles: tuple[str, ...] | None = None,
        prefer_hostapi: str | None = None,
        min_in_ch: int | None = None,
    ) -> int | None:
        """Keep preferred index only if it still names the expected hardware.

        When ``prefer_hostapi`` is set, re-home onto that host API even if the
        saved index still exists (HATS: DirectSound stereo vs WDM-KS dead-L).
        """
        by_idx = {d.index: d for d in devices}
        preferred_ok = False
        if preferred is not None and preferred in by_idx:
            d = by_idx[preferred]
            ok_ch = min_in_ch is None or d.max_input_channels >= min_in_ch
            preferred_ok = cls._device_matches(d, name_needles) and ok_ch
            if preferred_ok and not prefer_hostapi:
                return preferred
            if preferred_ok and prefer_hostapi.lower() in d.hostapi.lower():
                return preferred
        if not name_needles:
            return preferred if preferred_ok else preferred
        matches = [
            d
            for d in devices
            if cls._device_matches(d, name_needles)
            and (min_in_ch is None or d.max_input_channels >= min_in_ch)
        ]
        if prefer_hostapi:
            host = [d for d in matches if prefer_hostapi.lower() in d.hostapi.lower()]
            if host:
                return host[0].index
        if preferred_ok:
            return preferred
        return matches[0].index if matches else preferred

    @classmethod
    def _fill_combo(
        cls,
        combo: QComboBox,
        devices: list[DeviceInfo],
        *,
        allow_none: bool,
        preferred: int | None = None,
        name_needles: tuple[str, ...] | None = None,
        prefer_hostapi: str | None = None,
        min_in_ch: int | None = None,
    ) -> None:
        prev = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if allow_none:
            combo.addItem("(none)", None)
        for d in devices:
            combo.addItem(format_device(d), d.index)
        seed = prev if prev is not None else preferred
        pick = cls._resolve_device_index(
            devices,
            seed if isinstance(seed, int) else preferred,
            name_needles=name_needles,
            prefer_hostapi=prefer_hostapi,
            min_in_ch=min_in_ch,
        )
        if pick is not None:
            idx = combo.findData(pick)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _on_profile_changed(self) -> None:
        pi = self.profile.currentData() == "pi_hats"
        self.hats_in.setEnabled(True)
        self.hats_hint.setText(
            "HATS input required for pi_hats (sim-head / ear-rig → audio interface)."
            if pi
            else (
                "HATS / ear-rig optional on windows_pico — set a device to record "
                "hats_*.wav; leave (none) for array-only Part A takes."
            )
        )
        self.apply_to_session()

    def _on_campaign_changed(self) -> None:
        kind = str(self.campaign.currentData())
        cur = self.stimulus.text().strip()
        if kind == "in_context_speech":
            self.stimulus.setPlaceholderText(
                "Speech folder (random utterance loop per take) or one speech WAV"
            )
            # Remember the ESS and switch to the default speech pool, unless a speech
            # source is already set.
            if cur and Path(cur).is_file() and "MeasSweep" in Path(cur).name:
                self._last_ess_path = cur
            if not cur or (Path(cur).is_file() and "MeasSweep" in Path(cur).name):
                if DEFAULT_SPEECH_DIR.is_dir():
                    self.stimulus.setText(str(DEFAULT_SPEECH_DIR))
        else:
            self.stimulus.setPlaceholderText(
                "One shared REW ESS + timing-ref WAV (all tabs)"
            )
            if (not cur or Path(cur).is_dir()) and getattr(self, "_last_ess_path", ""):
                self.stimulus.setText(self._last_ess_path)
        self.apply_to_session()

    def _browse_stimulus(self) -> None:
        if str(self.campaign.currentData()) == "in_context_speech":
            path = QFileDialog.getExistingDirectory(
                self, "Speech folder (random utterance loop per take)", str(DEFAULT_SPEECH_DIR)
            )
            if path:
                self.stimulus.setText(path)
                self.apply_to_session()
            return
        path, _ = QFileDialog.getOpenFileName(self, "Stimulus WAV", "", "WAV (*.wav);;All (*.*)")
        if path:
            self.stimulus.setText(path)
            self.apply_to_session()

    def _browse_measure_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Measure root")
        if path:
            self.measure_root.setText(path)
            self.apply_to_session()

    def _browse_rt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "soundbubble-rt", "", "Executable (*.exe);;All (*.*)"
        )
        if path:
            self.rt_path.setText(path)
            self.apply_to_session()

    def apply_to_session(self) -> None:
        s = self.session
        s.acoustic_profile = str(self.profile.currentData())
        s.host_target = str(self.host.currentData())
        s.campaign_kind = str(self.campaign.currentData())
        s.stimulus_out_device = self.stimulus_out.currentData()
        s.pipeline_in_device = self.pipeline_in.currentData()
        s.pipeline_out_device = self.pipeline_out.currentData()
        s.hats_in_device = self.hats_in.currentData()
        s.input_sr = int(self.input_sr.value())
        s.output_sr = int(self.output_sr.value())
        s.channel_map = self.channel_map.text().strip()
        s.runtime = str(self.runtime.currentData())
        s.soundbubble_rt_path = self.rt_path.text().strip()
        s.use_zoo = self.use_zoo.isChecked()
        s.zoo_dir = self.zoo_dir.text().strip()
        s.model_path = self.model_path.text().strip()
        s.contract_path = self.contract_path.text().strip()
        s.bubble_radius_m = float(self.bubble.value())
        s.stimulus_path = self.stimulus.text().strip()
        s.stimulus_repeat = max(1, int(self.stimulus_repeat.value()))
        s.stimulus_play_channels = str(self.stimulus_play_ch.currentData() or "both")
        s.measure_root = self.measure_root.text().strip() or s.measure_root
        s.gains = Gains(
            input_gain_db=float(self.input_gain.value()),
            output_gain_db=float(self.output_gain.value()),
            mix_dry=float(self.mix_dry.value()),
            output_limiter=self.limiter.isChecked(),
            limiter_release_ms=float(self.limiter_release.value()),
            mic_gains=[float(sb.value()) for sb in self.mic_gain_boxes],
            model_level_dbfs=float(self.model_level.value()),
        )
        self._refresh_stimulus_meta()
        self._persist_paths()
        self.settingsChanged.emit()

    def _sync_mic_fader(self, slider: QSlider, label: QLabel, value: float) -> None:
        slider.blockSignals(True)
        slider.setValue(int(round(value * MIC_SLIDER_SCALE)))
        slider.blockSignals(False)
        label.setText(f"{20.0 * math.log10(max(value, 1e-6)):+.1f} dB")

    def _set_mic_gains(self, gains: list[float]) -> None:
        for sb, v in zip(self.mic_gain_boxes, gains):
            sb.setValue(float(v))

    def _normalize_mic_gains(self) -> None:
        self._set_mic_gains(normalized_mic_gains(MIC_CAL_LEVELS_DBFS, float(self.model_level.value())))

    def _refresh_stimulus_meta(self) -> None:
        path = self.stimulus.text().strip()
        repeat = max(1, int(self.stimulus_repeat.value()))
        kind = str(self.campaign.currentData())
        if not path:
            msg = (
                "Speech loop WAV required for in-context — not set."
                if kind == "in_context_speech"
                else "Shared ESS required for Levels / Polar / Algorithm / REW — not set."
            )
            self.stimulus_meta.setText(msg)
            self.take_len_lbl.setText("Take length: —")
            return
        p = Path(path)
        if kind == "in_context_speech" and p.is_dir():
            from tools.measure.stimulus import SPEECH_LOOP_SECONDS, list_speech_files

            n = len(list_speech_files(p))
            self.stimulus_meta.setText(
                f"Speech folder · {n} utterances · a new random loop (~{SPEECH_LOOP_SECONDS:.0f} s, "
                f"levelled) is generated per take and saved as stimulus_used.wav"
                if n else f"No *.flac / *.wav speech files under {path}"
            )
            self.take_len_lbl.setText(
                f"Take length: ~{SPEECH_LOOP_SECONDS:.0f} s × {repeat} = ~{SPEECH_LOOP_SECONDS * repeat:.0f} s"
            )
            return
        if not p.is_file():
            self.stimulus_meta.setText(f"Missing file: {path}")
            self.take_len_lbl.setText("Take length: —")
            return
        try:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            sha = h.hexdigest()[:12]
            size_kb = p.stat().st_size / 1024.0
            keep = (
                "(speech / in-context — may change between campaigns)"
                if kind == "in_context_speech"
                else "(keep this ESS file for the whole campaign)"
            )
            self.stimulus_meta.setText(
                f"{p.name}  ·  {size_kb:.0f} KiB  ·  sha256 {sha}…  {keep}"
            )
            dur_s = self._stimulus_duration_s(p)
            if dur_s is not None:
                total = dur_s * repeat
                self.take_len_lbl.setText(
                    f"Take length: {dur_s:.2f} s × {repeat} = {total:.2f} s"
                )
            else:
                self.take_len_lbl.setText(f"Take length: (unknown) × {repeat}")
        except OSError as exc:
            self.stimulus_meta.setText(f"Cannot read stimulus: {exc}")
            self.take_len_lbl.setText("Take length: —")

    @staticmethod
    def _stimulus_duration_s(path: Path) -> float | None:
        try:
            from scipy.io.wavfile import read as wavread

            sr, audio = wavread(str(path))
            n = int(getattr(audio, "shape", [0])[0])
            return float(n) / float(sr) if sr else None
        except Exception:
            return None

    def validate_basic(self) -> list[str]:
        """Shared Setup gate used by Algorithm (and callable by other tabs)."""
        errs: list[str] = []
        s = self.session
        if not s.stimulus_path:
            errs.append(
                "Stimulus WAV path is empty — pick the shared REW ESS + timing-ref on Setup."
            )
        elif not Path(s.stimulus_path).is_file() and not (
            s.campaign_kind == "in_context_speech" and Path(s.stimulus_path).is_dir()
        ):
            errs.append(f"Stimulus WAV (or speech folder) not found: {s.stimulus_path}")
        if s.stimulus_out_device is None:
            errs.append("Stimulus out device is required (Setup).")
        if s.acoustic_profile == "pi_hats" and s.hats_in_device is None:
            errs.append("pi_hats requires a HATS input device.")
        if s.pipeline_in_device is None:
            errs.append("Pipeline input device (Pico mics) is required.")
        return errs

    def validate_pipeline_audio(self) -> list[str]:
        """Gate Start Pipeline: devices present + SR accepted by PortAudio."""
        self.apply_to_session()
        self.snap_pipeline_sr_to_devices(silent=True)
        self.apply_to_session()
        s = self.session
        errs, in_sr, out_sr = validate_pipeline_rates(
            pipeline_in=s.pipeline_in_device,
            pipeline_out=s.pipeline_out_device,
            input_sr=int(s.input_sr),
            output_sr=int(s.output_sr),
        )
        if in_sr != int(self.input_sr.value()) or out_sr != int(self.output_sr.value()):
            self.input_sr.setValue(int(in_sr))
            self.output_sr.setValue(int(out_sr))
            self.apply_to_session()
        return errs
