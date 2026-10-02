"""DSENet-style soft angular gains for Sound Bubble steered targets.

Math (Kovalyov / Patel / Panahi, IEEE Access 2023; Sonitude port)::

    beta(theta) = exp(-0.5 * (theta / sigma) ** rho)
    w_i         = 1[d_i <= R] * beta(theta_i - theta0; sigma, rho)
    z           = sum_i w_i * x_ref,i

Hard bubble GT for b_hat is unchanged: sum of inside stems (w_i with beta=1).
Outside-bubble sources stay suppressed (beta never resurrects them).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


# Frozen Stage-G defaults (match Sonitude / DSENet paper priors).
DEFAULT_SIGMA = 0.2
DEFAULT_RHO = 8.0
# Coarse vertical-arc look directions (Stage C L/R evidence; not UCA fine beams).
DEFAULT_THETA0_GRID_DEG = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)


@dataclass(frozen=True)
class BetaConfig:
    sigma: float = DEFAULT_SIGMA
    rho: float = DEFAULT_RHO
    theta0_grid_deg: tuple[float, ...] = DEFAULT_THETA0_GRID_DEG


def wrap_angle_rad(theta: np.ndarray | float) -> np.ndarray:
    """Wrap to (-pi, pi]."""
    t = np.asarray(theta, dtype=np.float64)
    return (t + np.pi) % (2.0 * np.pi) - np.pi


def beta_from_azimuth_rad(
    theta_rad: np.ndarray | float,
    sigma: float = DEFAULT_SIGMA,
    rho: float = DEFAULT_RHO,
) -> np.ndarray:
    """Soft angular gain β(θ) = exp(-0.5 (θ/σ)^ρ)."""
    t = wrap_angle_rad(theta_rad)
    sig = max(float(sigma), 1e-8)
    return np.exp(-0.5 * np.power(np.abs(t) / sig, float(rho))).astype(np.float32)


def beta_from_azimuth_rad_torch(
    theta_rad: torch.Tensor,
    sigma: float = DEFAULT_SIGMA,
    rho: float = DEFAULT_RHO,
) -> torch.Tensor:
    t = (theta_rad + np.pi) % (2.0 * np.pi) - np.pi
    sig = max(float(sigma), 1e-8)
    return torch.exp(-0.5 * torch.pow(t.abs() / sig, float(rho)))


def product_weights(
    distances_m: Sequence[float],
    angles_deg: Sequence[float],
    *,
    theta0_deg: float,
    dis_threshold: float,
    sigma: float = DEFAULT_SIGMA,
    rho: float = DEFAULT_RHO,
) -> np.ndarray:
    """w_i = 1[d_i <= R] * β(θ_i - θ0). Outside-bubble → 0."""
    d = np.asarray(distances_m, dtype=np.float64)
    ang = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    delta = wrap_angle_rad(ang - np.deg2rad(float(theta0_deg)))
    beta = beta_from_azimuth_rad(delta, sigma=sigma, rho=rho)
    inside = (d <= float(dis_threshold)).astype(np.float32)
    return (inside * beta).astype(np.float32)


def steered_target_from_stems(
    stems: Sequence[np.ndarray],
    distances_m: Sequence[float],
    angles_deg: Sequence[float],
    *,
    theta0_deg: float,
    dis_threshold: float,
    sigma: float = DEFAULT_SIGMA,
    rho: float = DEFAULT_RHO,
) -> tuple[np.ndarray, np.ndarray]:
    """Build z = sum_i w_i x_i. Returns (z, weights)."""
    if not stems:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    w = product_weights(
        distances_m,
        angles_deg,
        theta0_deg=theta0_deg,
        dis_threshold=dis_threshold,
        sigma=sigma,
        rho=rho,
    )
    stacked = np.stack([np.asarray(s, dtype=np.float32).reshape(-1) for s in stems], axis=0)
    z = np.sum(stacked * w[:, None], axis=0).astype(np.float32)
    return z, w


def sample_theta0_deg(
    rng: np.random.Generator | None = None,
    grid_deg: Sequence[float] = DEFAULT_THETA0_GRID_DEG,
) -> float:
    g = rng or np.random.default_rng()
    return float(g.choice(np.asarray(list(grid_deg), dtype=np.float64)))


def gains_from_beta(
    angles_deg: Sequence[float],
    theta0_deg: float,
    *,
    sigma: float = DEFAULT_SIGMA,
    rho: float = DEFAULT_RHO,
    g_min: float = 0.25,
    g_max: float = 2.0,
) -> list[float]:
    """Map β(θ_k - θ0) ∈ [0,1] → mixer gains in [g_min, g_max]."""
    ang = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    delta = wrap_angle_rad(ang - np.deg2rad(float(theta0_deg)))
    beta = beta_from_azimuth_rad(delta, sigma=sigma, rho=rho)
    return [float(g_min + (g_max - g_min) * b) for b in beta]
