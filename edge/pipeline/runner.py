from __future__ import annotations

from dataclasses import dataclass


def snap_effective_model_sr(model_sr: int, correction_ratio: float, grid_hz: int = 100) -> int:
    """
    Snap drift-corrected model rate to a grid to keep resample_poly gcd large.
    """
    sr = float(model_sr) * float(correction_ratio)
    g = max(1, int(grid_hz))
    return max(g, int(round(sr / g)) * g)


@dataclass(frozen=True)
class IoBlocks:
    model_io_chunk: int
    input_block: int
    output_block: int
    # Model-rate samples gained (+) or lost (-) per capture block from rounding input_block.
    input_surplus_per_block: float


def compute_io_blocks(model_chunk: int, hops_per_io: int, model_sr: int, input_sr: int, output_sr: int) -> IoBlocks:
    model_io_chunk = int(model_chunk) * max(1, int(hops_per_io))
    input_block = int(round(model_io_chunk * input_sr / model_sr))
    output_block = int(round(model_io_chunk * output_sr / model_sr))
    surplus = input_block * float(model_sr) / float(input_sr) - model_io_chunk
    return IoBlocks(model_io_chunk, input_block, output_block, surplus)


def sustainable_steps(block_budget_ms: float, noninfer_ms: float, infer_ema_ms: float, reserve_ms: float = 1.0) -> int:
    # Floor at 1: the inference EMA only updates when a step runs, so 0 would latch forever.
    safe_budget_ms = max(0.0, block_budget_ms - noninfer_ms - reserve_ms)
    return max(1, int(safe_budget_ms // max(0.001, infer_ema_ms)))


@dataclass(frozen=True)
class OutputRingParams:
    ring_n_frames: int
    ring_max_ahead_frames: int
    ring_target_ahead_frames: int


def compute_output_ring_params(output_sr: int, output_block: int, *, max_ahead_sec: float, target_ahead_sec: float) -> OutputRingParams:
    sr = int(output_sr)
    block = int(output_block)
    ring_n_frames = max(block * 64, int(4.0 * sr))
    ring_max_ahead = max(block * 3, int(max_ahead_sec * sr))
    ring_target_ahead = max(block, int(target_ahead_sec * sr))
    if ring_target_ahead >= ring_max_ahead:
        ring_target_ahead = max(block, ring_max_ahead // 2)
    if ring_n_frames < ring_max_ahead * 2:
        ring_n_frames = ring_max_ahead * 2
    return OutputRingParams(
        ring_n_frames=ring_n_frames,
        ring_max_ahead_frames=ring_max_ahead,
        ring_target_ahead_frames=ring_target_ahead,
    )

