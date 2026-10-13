from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import edge.pipeline.processing as processing
from edge.controller.config import PipelineConfig, build_pipeline_argv
from edge.pipeline.runner import compute_io_blocks, sustainable_steps

ZOO_MODEL = REPO_ROOT / "edge" / "zoo" / "optim_pretrain_1_5m.onnx"


def test_48k_capture_blocks_divide_model_hop_exactly():
    blocks = compute_io_blocks(192, 1, 24000, 48000, 48000)
    assert (blocks.input_block, blocks.output_block) == (384, 384)
    assert blocks.input_surplus_per_block == 0.0


def test_44k1_capture_blocks_gain_model_samples_every_block():
    blocks = compute_io_blocks(192, 2, 24000, 44100, 48000)
    assert blocks.input_block == 706
    assert blocks.input_surplus_per_block == pytest.approx(0.218, abs=1e-3)


def test_one_hop_budget_never_plans_zero_steps():
    # 48 kHz, 1 hop: 8 ms block, ~1.1 ms non-inference; the old 8.5 ms seed gave 0 steps forever.
    assert sustainable_steps(8.0, 1.1, 8.5) == 1
    assert sustainable_steps(8.0, 1.1, 2.8) == 2


@pytest.mark.skipif(not ZOO_MODEL.is_file(), reason="zoo model not present")
def test_warmup_returns_timing_and_leaves_zero_state():
    from edge.pipeline.onnx_streamer import ONNXStreamer

    streamer = ONNXStreamer(str(ZOO_MODEL), intra_threads=1)
    ms = streamer.warmup((1, 6, 288), runs=3)
    assert ms > 0.0
    assert all(not np.any(v) for v in streamer.state_arrays.values())


def _block_vs_oneshot_snr_db(monkeypatch, src: int, dst: int, block: int) -> float:
    monkeypatch.setattr(processing, "soxr", None)
    t = np.arange(src // 2) / src
    x = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)[None, :]
    rs = processing.StreamingResampler(src_sr=src, dst_sr=dst, num_channels=1)
    assert rs.backend == "polyphase-fallback"
    y = np.concatenate([rs.process(x[:, i : i + block]) for i in range(0, x.shape[1] - block + 1, block)], axis=1)
    g = np.gcd(src, dst)
    ref = resample_poly(x, dst // g, src // g, axis=-1, window=("kaiser", 8.6))
    n = min(y.shape[1], ref.shape[1]) - 200
    err = y[0, 100:n] - ref[0, 100:n]
    return float(10 * np.log10(np.sum(ref[0, 100:n] ** 2) / np.sum(err**2)))


@pytest.mark.parametrize("src,dst,block", [(48000, 24000, 384), (24000, 48000, 192)])
def test_halfband_emits_exact_block_lengths_and_matches_oneshot(src, dst, block):
    x = np.random.default_rng(0).standard_normal((2, block * 20)).astype(np.float32)
    blocked = processing.StreamingResampler(src_sr=src, dst_sr=dst, num_channels=2)
    assert blocked.backend == "halfband-fir"
    outs = [blocked.process(x[:, i : i + block]) for i in range(0, x.shape[1], block)]
    assert {o.shape[1] for o in outs} == {block * dst // src}
    oneshot = processing.StreamingResampler(src_sr=src, dst_sr=dst, num_channels=2).process(x)
    np.testing.assert_allclose(np.concatenate(outs, axis=1), oneshot, atol=1e-6)


@pytest.mark.parametrize("src,dst", [(48000, 24000), (24000, 48000)])
def test_halfband_passes_speech_band_and_rejects_aliases(src, dst):
    t = np.arange(src) / src

    def gain_db(freq: float) -> float:
        rs = processing.StreamingResampler(src_sr=src, dst_sr=dst, num_channels=1)
        y = rs.process(np.sin(2 * np.pi * freq * t)[None, :].astype(np.float32))[0, dst // 10 :]
        return float(20 * np.log10(np.sqrt(2 * np.mean(y**2)) + 1e-12))

    assert abs(gain_db(1000.0)) < 0.1
    # Down: 20 kHz would alias to 4 kHz. Up: the 4 kHz input's image lands at 20 kHz (checked via down path).
    if src > dst:
        assert gain_db(20000.0) < -60.0


def test_asrc_halfband_44k1_to_24k_is_smooth_and_rate_exact():
    rs = processing.StreamingResampler(src_sr=44100, dst_sr=24000, num_channels=2)
    assert rs.backend == "asrc-halfband"
    block = 706
    x = np.random.default_rng(1).standard_normal((2, block * 50)).astype(np.float32)
    outs = [rs.process(x[:, i : i + block]) for i in range(0, x.shape[1], block)]
    lengths = [o.shape[1] for o in outs[5:]]
    assert max(lengths) - min(lengths) <= 2  # SoXR bursts were hundreds of samples
    total_out = sum(o.shape[1] for o in outs)
    assert abs(total_out / x.shape[1] - 24000 / 44100) < 0.002


def test_drift_monitor_returns_finite_ppm():
    mon = processing.DriftMonitor(block_duration_sec=0.016, target_fill=3, window_sec=1.0, update_every=1, max_ppm=500.0)
    ppm, corr = mon.update(5)
    assert np.isfinite(ppm) and 0.98 <= corr <= 1.02


def test_fallback_resampler_nonexact_ratio_blocks_track_oneshot(monkeypatch):
    # 44.1→24 now uses asrc-halfband; this keeps coverage of the polyphase fallback path.
    assert _block_vs_oneshot_snr_db(monkeypatch, 32000, 24000, 512) > 40.0


def test_controller_argv_passes_queue_and_overload_policy():
    argv = build_pipeline_argv(PipelineConfig(queue_policy="latest", overload_policy="silence_fill"))
    flags = dict(zip(argv[1::2], argv[2::2]))
    assert flags["--queue-policy"] == "latest"
    assert flags["--overload-policy"] == "silence_fill"
    for flag in ("--max-capture-age-ms", "--recover-steps-per-block", "--recover-ahead-multiplier", "--warmup-runs"):
        assert flag in argv
    assert "--zoo-dir" in argv and "--model" not in argv


def test_pipeline_imports_as_module_without_torch():
    code = "import sys, edge.inference_pipeline; sys.exit(1 if 'torch' in sys.modules else 0)"
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


ZOO_DIR = REPO_ROOT / "edge" / "zoo"
EP36_ONNX = ZOO_DIR / "optim_pretrain_sonitude_dist_count_1_5m_ep36.onnx"
FIXTURE_CONTRACT = ZOO_DIR / "optim_count_distance_fixture_1_5m.runtime.json"


@pytest.mark.skipif(not EP36_ONNX.is_file(), reason="ep36 zoo model not present")
def test_zoo_resolve_prefers_ep36_over_fixture():
    from edge.inference_pipeline import _resolve_zoo_model

    onnx_path, contract_path, available = _resolve_zoo_model(str(ZOO_DIR), 1.5)
    assert 1.5 in available
    assert Path(onnx_path).name == "optim_pretrain_sonitude_dist_count_1_5m_ep36.onnx"
    assert Path(contract_path).name == "optim_pretrain_sonitude_dist_count_1_5m_ep36.runtime.json"


@pytest.mark.skipif(not FIXTURE_CONTRACT.is_file(), reason="fixture contract not present")
def test_untrained_contract_refused_without_allow_flag():
    # Explicit --model + untrained contract must exit unless --allow-untrained.
    cmd = [
        sys.executable,
        "-m",
        "edge.inference_pipeline",
        "--model",
        str(ZOO_DIR / "optim_count_distance_fixture_1_5m.onnx"),
        "--contract-path",
        str(FIXTURE_CONTRACT),
        "--list-devices",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode != 0
    assert "refusing untrained" in (res.stderr + res.stdout).lower() or "trained_epochs=0" in (res.stderr + res.stdout)


def test_controller_silence_dbfs_default_is_minus_100():
    cfg = PipelineConfig()
    assert cfg.silence_dbfs == -100.0
    argv = build_pipeline_argv(cfg)
    flags = dict(zip(argv[1::2], argv[2::2]))
    assert float(flags["--silence-dbfs"]) == -100.0


def test_controller_and_pipeline_default_input_sr_is_48000():
    from edge import inference_pipeline as ip

    assert ip.DEFAULT_INPUT_SR == 48000
    cfg = PipelineConfig()
    assert cfg.input_sr == 48000
    argv = build_pipeline_argv(cfg)
    flags = dict(zip(argv[1::2], argv[2::2]))
    assert int(flags["--input-sr"]) == 48000


def test_assert_stream_samplerate_rejects_mismatch():
    from edge.realtime.audio_io import assert_stream_samplerate

    class _Stream:
        samplerate = 44100.0

    try:
        assert_stream_samplerate(_Stream(), expected_sr=48000, label="input")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "44100" in str(exc) and "48000" in str(exc)


def test_channel_level_validator_silence_threshold_b8():
    # -60 dBFS is below the old -50 default (muted) but above the new -100 default (kept).
    amp = 10 ** (-60.0 / 20.0)
    x = np.full((1, 256), amp, dtype=np.float32)

    mute_old = processing.ChannelLevelValidator(
        num_channels=1,
        silence_dbfs=-50.0,
        fault_min_dbfs=-1.0,
        fault_max_dbfs=0.0,
        silence_window_frames=1,
    )
    y_old, _, statuses_old = mute_old.validate(x, frame_idx=0)
    assert statuses_old[0] == "silence"
    assert float(np.max(np.abs(y_old))) == 0.0

    keep_new = processing.ChannelLevelValidator(
        num_channels=1,
        silence_dbfs=-100.0,
        fault_min_dbfs=-1.0,
        fault_max_dbfs=0.0,
        silence_window_frames=1,
    )
    y_new, _, statuses_new = keep_new.validate(x, frame_idx=0)
    assert statuses_new[0] == "ok"
    assert float(np.max(np.abs(y_new))) > 0.0


def test_channel_probe_xcorr_recovers_known_lag():
    from edge.tools.channel_probe import _xcorr_lag_samples

    rng = np.random.default_rng(0)
    ref = rng.standard_normal(8000).astype(np.float64)
    delay = 7
    sig = np.concatenate([np.zeros(delay), ref[:-delay]])
    lag, peak = _xcorr_lag_samples(ref, sig, max_lag=32)
    assert abs(lag - delay) < 0.25
    assert peak > 0.9
