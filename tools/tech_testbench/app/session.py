"""Shared session settings for the tech testbench GUI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


# Per-mic calibration, indexed by DEVICE channel 0..5 (before --channel-map).
# WIDEBAND-ONLY rule (2026-11-04): a mic is corrected only for a wideband
# (flat within 3 dB over 250 Hz-8 kHz, >= 6 dB) hardware offset, measured vs the
# loudest mics in front-facing (az 0, el 0) takes; natural inter-mic differences
# are kept (all mics share one base gain). See docs/polar_sweep_1m_ep35_20261104.md.
#   MIC_REF_LEVEL_DBFS: active-sweep level of a normal-state mic at the cal pose
#                       (dev 0/1/3 mean, polar_20261104_041315_fa49, ESS 1 m az 0).
#   MIC_WIDEBAND_OFFSET_DB: measured flat hardware offset per mic (0 = normal).
#     dev2 -12.6 and dev5 -12.9 dB: mean of 5 direct-capture runs 2026-11-04 06:55
#     UTC (sd <= 0.04 dB; debug/mic_state_check_series_20261104.json), stable all day.
#     dev4 TOGGLES: normal since 05:37 (0.0, used here); low state -12.1 dB -> fader ~x6.1.
#     dev1 (-0.8) / dev3 (-0.8) / dev0 (+0.6) are within capsule tolerance -> not corrected.
MIC_REF_LEVEL_DBFS: float = -22.6
MIC_WIDEBAND_OFFSET_DB: tuple[float, ...] = (0.0, 0.0, -12.6, 0.0, 0.0, -12.9)
MIC_CAL_LEVELS_DBFS: tuple[float, ...] = tuple(
    round(MIC_REF_LEVEL_DBFS + o, 2) for o in MIC_WIDEBAND_OFFSET_DB
)
# Previous (superseded) per-pose calibration from polar_20261104_034603_4326,
# dev4 in its low state: (-22.1, -23.1, -35.4, -22.8, -34.9, -35.4).
# Per-mic normalization target (dBFS, pre input gain). -19 + DEFAULT_INPUT_GAIN_DB
# 15 dB was the best-sounding live setting on the bench with EP35 (2026-11-04,
# polar_20261104_041315_fa49: voice band keep -18..-29 dB vs -39..-57 at -18/+10).
MODEL_LEVEL_DBFS: float = -19.0
DEFAULT_INPUT_GAIN_DB: float = 15.0


def normalized_mic_gains(
    levels_dbfs: tuple[float, ...] | list[float] = MIC_CAL_LEVELS_DBFS,
    model_level_dbfs: float = MODEL_LEVEL_DBFS,
) -> list[float]:
    """Linear (always > 0) per-mic gains that put each mic at the model level.

    Input gain is applied on top (default DEFAULT_INPUT_GAIN_DB).
    """
    return [round(10.0 ** ((model_level_dbfs - lv) / 20.0), 4) for lv in levels_dbfs]


def _default_mic_gains() -> list[float]:
    return normalized_mic_gains()


@dataclass
class Gains:
    input_gain_db: float = DEFAULT_INPUT_GAIN_DB
    output_gain_db: float = 0.0
    mix_dry: float = 0.0
    output_limiter: bool = True
    limiter_release_ms: float = 25.0
    # Linear per-mic gains by device channel (x1.0 = unity). Default: normalized
    # so every mic sits at MODEL_LEVEL_DBFS for the calibration take.
    mic_gains: list[float] = field(default_factory=_default_mic_gains)
    model_level_dbfs: float = MODEL_LEVEL_DBFS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionSettings:
    acoustic_profile: str = "windows_pico"  # windows_pico | pi_hats
    host_target: str = "local_windows"  # local_windows | pi_rig

    # Shared REW ESS + timing-ref (*_refL); Windows Pico Part A / controlled_ess default.
    stimulus_path: str = (
        r"C:\Users\darre\Documents\Obsidian Notes\CDE3301\3301-ANCHRTF\AY 2627 SEM1"
        r"\Software and Firmware\6ChMic-DataCollection"
        r"\256kMeasSweep_0_to_20000_-20_dBFS_44k_Float_LR_refL.wav"
    )
    measure_root: str = str(REPO_ROOT / "measure")
    active_run_id: str = ""  # current run folder; trials append here until cleared

    # Local Windows Pico bench (QUAD-CAPTURE stim/HATS + Class B INMP441). Indices
    # drift after replug/driver churn; Setup matches by name (INMP441 / QUAD 1-2).
    stimulus_out_device: int | None = 41  # QUAD-CAPTURE 1-2 out @ 44.1 kHz WDM-KS
    pipeline_in_device: int | None = 37  # INMP441 8ch mic @ 24 kHz WDM-KS
    pipeline_out_device: int | None = 38  # INMP441 DAC / earcups @ 24 kHz WDM-KS
    hats_in_device: int | None = 17  # QUAD-CAPTURE 1-2 in @ 44.1 kHz DirectSound (stereo)

    input_sr: int = 24000
    output_sr: int = 24000
    channel_map: str = "5,4,3,2,1,0"  # MIC_INDEX_CANON.md §4

    runtime: str = "cpp"  # cpp | python
    soundbubble_rt_path: str = str(
        REPO_ROOT / "edge" / "cpp" / "build-local" / "soundbubble-rt.exe"
    )
    # EP35 default (2026-11-04). EP24 mic0m25 stays available: optim_sonitude_dist_count_1_5m_ep24_mic0m25.*
    model_path: str = str(
        REPO_ROOT / "edge" / "zoo" / "optim_sonitude_dist_count_1_5m_ep35.onnx"
    )
    contract_path: str = str(
        REPO_ROOT
        / "edge"
        / "zoo"
        / "optim_sonitude_dist_count_1_5m_ep35.runtime.json"
    )
    zoo_dir: str = str(REPO_ROOT / "edge" / "zoo")
    use_zoo: bool = False
    bubble_radius_m: float = 1.5

    gains: Gains = field(default_factory=Gains)

    distance_m: float = 1.5
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0

    # Campaign / in-context labeling (1A discrete grids, 2A external 2nd source)
    campaign_kind: str = "controlled_ess"  # controlled_ess | in_context_speech
    motion_direction: str = "static"  # static | into_bubble | out_of_bubble
    fixed_source_m: float | None = None  # second/external speaker distance; None = 1-speaker
    moving_source_m: float | None = None  # defaults to distance_m when writing trials
    notes: str = ""
    stimulus_repeat: int = 1  # play stimulus WAV N times (speech loop takes)
    # Stereo DAC routing for the shared ESS/speech WAV: both | left | right
    # ESS files named *_refL need left (or both) so the timing reference is audible.
    stimulus_play_channels: str = "left"

    # Hardware characterization (no jig): per-capsule on USB slot 0, then full array
    hw_char_mode: str = "headphone_array"  # per_capsule | headphone_array
    capsule_id: str = ""  # operator label for the capsule under test (per_capsule)
    usb_slot: int = 0  # physical wiring slot used for this take (per_capsule → always 0)

    def ensure_run_id(self, store: Any) -> str:
        """Keep appending trials to one run until the operator starts a new run."""
        if self.active_run_id:
            run_dir = Path(self.measure_root) / self.active_run_id
            if run_dir.is_dir():
                return self.active_run_id
        run_root = store.new_run(
            meta={
                "acoustic_profile": self.acoustic_profile,
                "host_target": self.host_target,
                "stimulus_path": self.stimulus_path,
                "campaign_kind": self.campaign_kind,
            }
        )
        self.active_run_id = run_root.name
        return self.active_run_id

    def start_new_run(self, store: Any) -> str:
        self.active_run_id = ""
        return self.ensure_run_id(store)

    def effective_moving_source_m(self) -> float:
        if self.moving_source_m is not None:
            return float(self.moving_source_m)
        return float(self.distance_m)

    def to_trial_fields(self) -> dict[str, Any]:
        return {
            "acoustic_profile": self.acoustic_profile,
            "host_target": self.host_target,
            "stimulus_path": self.stimulus_path,
            "stimulus_out_device": self.stimulus_out_device,
            "pipeline_in_device": self.pipeline_in_device,
            "pipeline_out_device": self.pipeline_out_device,
            "hats_in_device": self.hats_in_device,
            "input_sr": self.input_sr,
            "output_sr": self.output_sr,
            "channel_map": self.channel_map,
            "runtime": self.runtime,
            "soundbubble_rt_path": self.soundbubble_rt_path,
            "model_path": self.model_path,
            "contract_path": self.contract_path,
            "zoo_dir": self.zoo_dir if self.use_zoo else "",
            "use_zoo": self.use_zoo,
            "bubble_radius_m": self.bubble_radius_m,
            "gains": self.gains.to_dict(),
            "distance_m": self.distance_m,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "measure_root": self.measure_root,
            "active_run_id": self.active_run_id,
            "campaign_kind": self.campaign_kind,
            "motion_direction": self.motion_direction,
            "fixed_source_m": self.fixed_source_m,
            "moving_source_m": self.effective_moving_source_m(),
            "notes": self.notes,
            "stimulus_repeat": max(1, int(self.stimulus_repeat)),
            "stimulus_play_channels": self.stimulus_play_channels,
            "hw_char_mode": self.hw_char_mode,
            "capsule_id": self.capsule_id,
            "usb_slot": int(self.usb_slot),
        }
