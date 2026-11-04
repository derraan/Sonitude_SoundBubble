"""Unit checks for ESS timing-ref analysis. Run from repo root:

  python tools/tech_testbench/scripts/test_timing_ref.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write as wavwrite

from tools.tech_testbench.app.analysis.e2e_latency import (
    _apply_timing_ref_absolute_legs,
    measure_e2e_latency,
)
from tools.tech_testbench.app.analysis.timing_ref import (
    analyze_timing_ref_trial,
    extract_timing_template,
    locate_timing_ref,
)


def _click(sr: int, n: int, at: int, amp: float = 0.9) -> np.ndarray:
    x = np.zeros(n, dtype=np.float64)
    # Short bipolar impulse (REW-like timing click energy).
    if 0 <= at < n:
        x[at] = amp
    if 0 <= at + 1 < n:
        x[at + 1] = -0.6 * amp
    if 0 <= at + 2 < n:
        x[at + 2] = 0.25 * amp
    return x


def test_extract_and_locate() -> None:
    sr = 24000
    n = sr  # 1 s
    t0 = 1200
    delay = 48  # 2 ms at 24 kHz
    leg = _click(sr, n, t0)
    # Quiet sweep body after click so envelope peak is the click.
    leg[t0 + 100 :] += 0.01 * np.sin(2 * np.pi * 1000 * np.arange(n - t0 - 100) / sr)
    stim = np.stack([leg, np.zeros(n)], axis=1)

    extracted = extract_timing_template(stim, sr, ref_leg="left")
    assert extracted is not None, "expected timing template"
    template, stim_t0 = extracted
    assert abs(stim_t0 - t0) <= 2, (stim_t0, t0)

    stream = np.zeros(n + delay, dtype=np.float64)
    stream[delay : delay + n] = leg
    loc = locate_timing_ref(
        stream,
        sr,
        template,
        max_tau_s=0.05,
        stimulus_t0_sample=stim_t0,
        stimulus_ref_leg=leg,
        stimulus_sr=sr,
    )
    assert abs(loc["lag_samples"] - delay) < 1.0, loc


def test_trial_controlled_ess() -> None:
    sr = 24000
    n = sr
    t0 = 800
    delay = 36
    leg = _click(sr, n, t0)
    stim = np.stack([leg, 0.05 * np.random.default_rng(0).standard_normal(n)], axis=1).astype(
        np.float32
    )
    buf = n + delay + 8
    raw0 = np.zeros(buf, dtype=np.float32)
    raw0[delay : delay + n] = leg.astype(np.float32)
    raw1 = np.zeros(buf, dtype=np.float32)
    raw1[delay + 3 : delay + 3 + n] = leg.astype(np.float32)

    td = Path(tempfile.mkdtemp(prefix="timing_ref_ess_"))
    (td / "trial.json").write_text(
        json.dumps(
            {
                "kind": "levels",
                "campaign_kind": "controlled_ess",
                "stimulus_play_channels": "left",
            }
        ),
        encoding="utf-8",
    )
    wavwrite(str(td / "stimulus_used.wav"), sr, stim)
    wavwrite(str(td / "levels_8ch.wav"), sr, np.stack([raw0, raw1], axis=1))

    out = analyze_timing_ref_trial(td, write_plots=False)
    assert out is not None and out.get("enabled") is True, out
    assert abs(float(out["streams"]["raw_ch0"]["lag_samples"]) - delay) < 1.5, out
    assert (td / "plots" / "timing_ref.json").is_file()
    m = json.loads((td / "metrics.json").read_text(encoding="utf-8"))
    assert m.get("timing_ref", {}).get("enabled") is True


def test_speech_campaign_skips() -> None:
    sr = 24000
    n = sr // 4
    leg = _click(sr, n, 100)
    stim = np.stack([leg, leg], axis=1).astype(np.float32)
    td = Path(tempfile.mkdtemp(prefix="timing_ref_speech_"))
    (td / "trial.json").write_text(
        json.dumps(
            {
                "kind": "algorithm",
                "campaign_kind": "in_context_speech",
                "stimulus_play_channels": "both",
            }
        ),
        encoding="utf-8",
    )
    wavwrite(str(td / "stimulus_used.wav"), sr, stim)
    wavwrite(str(td / "raw_8ch.wav"), sr, np.stack([leg, leg], axis=1).astype(np.float32))

    out = analyze_timing_ref_trial(td, write_plots=False)
    assert out is not None
    assert out.get("enabled") is False
    assert "controlled_ess" in str(out.get("reason", ""))


def test_e2e_prefers_timing_ref_absolute() -> None:
    sr = 24000
    n = 4000
    stim = np.random.default_rng(1).standard_normal(n)
    raw = np.concatenate([np.zeros(20), stim[:-20]])
    dry = np.concatenate([np.zeros(5), raw[:-5]])
    result = measure_e2e_latency(
        {"stimulus": stim, "raw": raw, "dry": dry},
        {"stimulus": sr, "raw": sr, "dry": sr},
        max_tau_s=0.05,
    )
    timing = {
        "enabled": True,
        "stim_to_raw_ms": 12.345,
        "stim_to_dry_ms": 13.0,
        "stim_to_wet_ms": None,
    }
    patched = _apply_timing_ref_absolute_legs(result, timing)
    assert patched["summary_ms"]["stimulus_to_raw_ms"] == 12.345
    assert patched["summary_ms"]["stimulus_to_dry_ms"] == 13.0
    assert "stimulus_to_raw" in patched["timing_ref_absolute_legs"]
    # Adjacent tap↔tap still present from GCC.
    assert patched["summary_ms"]["raw_to_dry_ms"] is not None


def main() -> None:
    test_extract_and_locate()
    test_trial_controlled_ess()
    test_speech_campaign_skips()
    test_e2e_prefers_timing_ref_absolute()
    print("ok")


if __name__ == "__main__":
    main()
