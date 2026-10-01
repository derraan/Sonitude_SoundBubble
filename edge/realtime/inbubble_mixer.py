"""In-bubble controllable mixer (additive on promoted bubble output).

Mix rule (default g_k=1 recovers b_hat exactly)::

    y = b_hat + sum_k (g_k - 1) * s_hat_k

Outside-bubble energy must already be absent from b_hat and s_hat_k;
this module does not resurrect suppressed sources.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MixerResult:
    y: np.ndarray
    b_hat: np.ndarray
    gains: tuple[float, ...]
    residual_rms: float


def mix_inbubble(
    b_hat: np.ndarray,
    s_hats: Sequence[np.ndarray],
    gains: Sequence[float] | None = None,
) -> MixerResult:
    """Apply the in-bubble mix rule.

    Parameters
    ----------
    b_hat:
        Promoted bubble mixture, shape ``(T,)`` or ``(1, T)``.
    s_hats:
        In-bubble speaker streams, each same length as ``b_hat``.
    gains:
        Per-stream gains. ``None`` or all ones → ``y == b_hat``.
    """
    b = np.asarray(b_hat, dtype=np.float32).reshape(-1)
    streams = [np.asarray(s, dtype=np.float32).reshape(-1) for s in s_hats]
    for i, s in enumerate(streams):
        if s.shape[0] != b.shape[0]:
            raise ValueError(f"s_hat[{i}] length {s.shape[0]} != b_hat {b.shape[0]}")

    if gains is None:
        g = tuple(1.0 for _ in streams)
    else:
        if len(gains) != len(streams):
            raise ValueError(f"gains ({len(gains)}) != streams ({len(streams)})")
        g = tuple(float(x) for x in gains)

    y = b.copy()
    for gk, sk in zip(g, streams):
        if gk != 1.0:
            y = y + (gk - 1.0) * sk

    if streams:
        residual = b - np.sum(np.stack(streams, axis=0), axis=0)
        residual_rms = float(np.sqrt(np.mean(residual * residual) + 1e-12))
    else:
        residual_rms = 0.0

    return MixerResult(y=y.astype(np.float32, copy=False), b_hat=b, gains=g, residual_rms=residual_rms)


def apply_leakage(
    s_hats: Sequence[np.ndarray],
    leak: float,
) -> list[np.ndarray]:
    """Cross-fade streams to simulate separation leakage.

    ``leak=0`` keeps oracle streams; ``leak=1`` fully swaps / averages neighbors.
    For K=2: ``s0' = (1-a)*s0 + a*s1``, ``s1' = (1-a)*s1 + a*s0`` with ``a=leak``.
    For K>2: each stream mixes toward the mean of the others.
    """
    a = float(np.clip(leak, 0.0, 1.0))
    streams = [np.asarray(s, dtype=np.float32).reshape(-1) for s in s_hats]
    if len(streams) < 2 or a == 0.0:
        return [s.copy() for s in streams]

    out: list[np.ndarray] = []
    for i, s in enumerate(streams):
        others = [streams[j] for j in range(len(streams)) if j != i]
        mean_others = np.mean(np.stack(others, axis=0), axis=0)
        out.append(((1.0 - a) * s + a * mean_others).astype(np.float32))
    return out


def si_sdr_np(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SDR in dB (numpy)."""
    e = np.asarray(est, dtype=np.float64).reshape(-1)
    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    n = min(e.shape[0], r.shape[0])
    e = e[:n]
    r = r[:n]
    if n == 0:
        return float("nan")
    alpha = np.dot(e, r) / (np.dot(r, r) + eps)
    s_target = alpha * r
    e_noise = e - s_target
    return float(10.0 * np.log10((np.dot(s_target, s_target) + eps) / (np.dot(e_noise, e_noise) + eps)))
