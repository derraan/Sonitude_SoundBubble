"""Smoke check for mostly-work testbench mods. Run from repo root."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write as wavwrite

from tools.measure.core import _union_artifacts, config_from_session_dict
from tools.measure.stimulus import StimulusPlayer
from tools.tech_testbench.app.analysis.mic_delay import (
    analyze_mic_delay_trial,
    relative_delays_from_multich,
)
from tools.tech_testbench.app.analysis.timing_ref import analyze_timing_ref_trial
from tools.tech_testbench.app.session import SessionSettings
from tools.tech_testbench.app.storage.trial_store import TrialStore
from tools.tech_testbench.app.ui.pose_queue import build_preset


def main() -> None:
    assert len(build_preset("part_b_az")) == 12
    assert [p.distance_m for p in build_preset("ctx_dist", reverse_distance=True)] == [
        2.0,
        1.7,
        1.5,
        1.3,
        1.0,
        0.5,
    ]
    assert len(build_preset("ctx_elev_dist")) == 20

    sr = 24000
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(sr).astype(np.float64)
    delay = 12
    x1 = np.concatenate([np.zeros(delay), x0[:-delay]])
    t = np.arange(sr)
    tone = np.sin(2 * np.pi * 800 * t / sr)
    tone_d = np.concatenate([np.zeros(delay), tone[:-delay]])
    for name, a0, a1 in (("noise", x0, x1), ("tone", tone, tone_d)):
        audio = np.stack([a0, a1, a0], axis=1)
        r = relative_delays_from_multich(audio, sr, ref_channel=0, max_channels=3)
        print(name, r["relative_delay_samples"])
        assert abs(r["relative_delay_samples"][1] - delay) < 1.5, (name, r)

    td = Path(tempfile.mkdtemp())
    (td / "trial.json").write_text(json.dumps({"kind": "levels"}), encoding="utf-8")
    wavwrite(
        str(td / "levels_8ch.wav"),
        sr,
        np.stack([x0, x1], axis=1).astype(np.float32),
    )
    out = analyze_mic_delay_trial(td, write_plots=False)
    assert out and abs(out["relative_delay_samples"][1] - 12) < 1.5
    assert (td / "plots" / "mic_delay.csv").is_file()
    m = json.loads((td / "metrics.json").read_text(encoding="utf-8"))
    assert "relative_delay_ms" in m

    # Speech campaign must not enable timing_ref.
    skip = analyze_timing_ref_trial(
        td, campaign_kind="in_context_speech", write_plots=False
    )
    assert skip is not None and skip.get("enabled") is False

    s = SessionSettings(
        campaign_kind="in_context_speech",
        fixed_source_m=0.5,
        stimulus_repeat=3,
        distance_m=1.3,
    )
    cfg = config_from_session_dict(s.to_trial_fields())
    assert cfg.stimulus_repeat == 3 and cfg.fixed_source_m == 0.5

    # bubble_radius 0.0 must not be coerced to 1.5
    cfg0 = config_from_session_dict(
        {"bubble_radius_m": 0.0, "zoo_dir": "edge/zoo", "use_zoo": True}
    )
    assert cfg0.runtime.bubble_radius_m == 0.0

    # Artifact deep-merge keeps analysis plots when capture overlays FR-only subset.
    merged = _union_artifacts(
        {"wavs": ["a.wav"], "plots": ["plots/mic_delay.csv", "plots/timing_ref.json"]},
        {"wavs": ["b.wav"], "plots": ["plots/suppression_fr.png"]},
    )
    assert "plots/mic_delay.csv" in merged["plots"]
    assert "plots/timing_ref.json" in merged["plots"]
    assert "plots/suppression_fr.png" in merged["plots"]
    assert TrialStore.merge_artifacts(
        {"plots": ["plots/mic_delay.csv"]},
        {"plots": ["plots/timing_ref.json"]},
    )["plots"] == ["plots/mic_delay.csv", "plots/timing_ref.json"]

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    wavwrite(path, sr, (x0[: sr // 10] * 0.1).astype(np.float32))
    player = StimulusPlayer(path, device=None, repeat=4)
    assert abs(player.info.duration_s - 0.4) < 1e-3
    os.unlink(path)
    print("ok")


if __name__ == "__main__":
    main()
