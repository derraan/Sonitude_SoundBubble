"""DSENet-style soft angular gains for Sound Bubble steered targets.

Math (Kovalyov / Patel / Panahi, IEEE Access 2023; Sonitude port)::

    beta(theta) = exp(-0.5 * (theta / sigma) ** rho)
    w_i         = 1[d_i <= R] * beta(theta_i - theta0; sigma, rho)
    z           = sum_i w_i * x_ref,i

Hard bubble GT for b_hat is unchanged: sum of inside stems (w_i with beta=1).
Outside-bubble sources stay suppressed (beta never resurrects them).

Look / no-look SI-SDR (training metric)::
    look     = steered mix with θ0 nearest the speaker
    no-look  = hard bubble (β=1 for all in-bubble sources)
    delta_db = min(cap, SI-SDR(look, spk) - SI-SDR(no-look, spk))   # cap default 6 dB

Beta modulation searches σ so the oracle look delta approaches ``cap`` without exceeding it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


# Frozen Stage-G defaults (match Sonitude / DSENet paper priors).
DEFAULT_SIGMA = 0.2
DEFAULT_RHO = 8.0
DEFAULT_LOOK_DELTA_CAP_DB = 6.0
# Candidate σ values for β modulation (radians). Smaller → sharper beam.
DEFAULT_SIGMA_CANDIDATES = (
    0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.0,
)


def make_theta0_grid_deg(step_deg: float = 30.0, start_deg: float = 0.0) -> tuple[float, ...]:
    """Full-circle look directions at ``step_deg`` spacing (default 30° → 12 looks).

    Returns angles in ``[start, start+360)`` normalized to ``[0, 360)``.
    Example: ``make_theta0_grid_deg(30) == (0, 30, …, 330)``.
    """
    step = float(step_deg)
    if step <= 0.0 or step > 360.0:
        raise ValueError(f"step_deg must be in (0, 360], got {step_deg}")
    start = float(start_deg) % 360.0
    n = int(round(360.0 / step))
    if abs(n * step - 360.0) > 1e-6:
        raise ValueError(f"step_deg={step_deg} does not divide 360 evenly")
    return tuple((start + i * step) % 360.0 for i in range(n))


# Full 360° at 30° steps (not an arc-limited ±90° subset).
DEFAULT_THETA0_GRID_DEG = make_theta0_grid_deg(30.0)


@dataclass(frozen=True)
class BetaConfig:
    sigma: float = DEFAULT_SIGMA
    rho: float = DEFAULT_RHO
    theta0_grid_deg: tuple[float, ...] = DEFAULT_THETA0_GRID_DEG
    look_delta_cap_db: float = DEFAULT_LOOK_DELTA_CAP_DB


def wrap_angle_rad(theta: np.ndarray | float) -> np.ndarray:
    """Wrap to (-pi, pi]."""
    t = np.asarray(theta, dtype=np.float64)
    return (t + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angle_deg(theta: np.ndarray | float) -> np.ndarray:
    t = np.asarray(theta, dtype=np.float64)
    return (t + 180.0) % 360.0 - 180.0


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


def nearest_theta0_deg(
    angle_deg: float,
    grid_deg: Sequence[float] = DEFAULT_THETA0_GRID_DEG,
) -> float:
    """θ0 on the 30° grid closest to ``angle_deg`` (circular)."""
    grid = np.asarray(list(grid_deg), dtype=np.float64)
    d = np.abs(wrap_angle_deg(grid - float(angle_deg)))
    return float(grid[int(np.argmin(d))])


def sample_look_theta0_deg(
    angles_deg: Sequence[float],
    *,
    grid_deg: Sequence[float] = DEFAULT_THETA0_GRID_DEG,
    rng: np.random.Generator | None = None,
) -> tuple[float, int]:
    """Pick a look direction: nearest grid θ0 to a random in-bubble speaker angle.

    Returns (theta0_deg, looked_speaker_index).
    """
    g = rng or np.random.default_rng()
    angs = [float(a) for a in angles_deg]
    if not angs:
        return sample_theta0_deg(g, grid_deg), -1
    spk_idx = int(g.integers(0, len(angs)))
    return nearest_theta0_deg(angs[spk_idx], grid_deg), spk_idx


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


def si_sdr_db(
    est: np.ndarray | torch.Tensor,
    ref: np.ndarray | torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Scale-invariant SDR in dB (numpy path for dataset / metrics)."""
    if torch.is_tensor(est):
        e = est.detach().float().reshape(-1).cpu().numpy().astype(np.float64)
    else:
        e = np.asarray(est, dtype=np.float64).reshape(-1)
    if torch.is_tensor(ref):
        r = ref.detach().float().reshape(-1).cpu().numpy().astype(np.float64)
    else:
        r = np.asarray(ref, dtype=np.float64).reshape(-1)
    n = min(e.shape[0], r.shape[0])
    if n == 0:
        return float("nan")
    e = e[:n]
    r = r[:n]
    alpha = np.dot(e, r) / (np.dot(r, r) + eps)
    s_target = alpha * r
    e_noise = e - s_target
    return float(10.0 * np.log10((np.dot(s_target, s_target) + eps) / (np.dot(e_noise, e_noise) + eps)))


def si_sdr_db_torch(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable SI-SDR in dB. ``est``/``ref`` shape [T] or [B,T] or [B,1,T]."""
    e = est.reshape(est.shape[0], -1) if est.dim() >= 2 else est.reshape(1, -1)
    r = ref.reshape(ref.shape[0], -1) if ref.dim() >= 2 else ref.reshape(1, -1)
    t = min(e.shape[-1], r.shape[-1])
    e = e[..., :t]
    r = r[..., :t]
    # Broadcast batch if one side is B=1
    if e.shape[0] != r.shape[0]:
        if e.shape[0] == 1:
            e = e.expand(r.shape[0], -1)
        elif r.shape[0] == 1:
            r = r.expand(e.shape[0], -1)
    dot_er = (e * r).sum(dim=-1)
    dot_rr = (r * r).sum(dim=-1).clamp_min(eps)
    alpha = dot_er / dot_rr
    s_target = alpha.unsqueeze(-1) * r
    e_noise = e - s_target
    num = (s_target * s_target).sum(dim=-1).clamp_min(eps)
    den = (e_noise * e_noise).sum(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(num / den)


def look_delta_db(
    look_sig: np.ndarray | torch.Tensor,
    no_look_sig: np.ndarray | torch.Tensor,
    speaker_ref: np.ndarray | torch.Tensor,
    *,
    cap_db: float = DEFAULT_LOOK_DELTA_CAP_DB,
) -> float:
    """SI-SDR(look) − SI-SDR(no-look) vs speaker, capped at ``cap_db``."""
    d = si_sdr_db(look_sig, speaker_ref) - si_sdr_db(no_look_sig, speaker_ref)
    if not np.isfinite(d):
        return float("nan")
    return float(min(float(cap_db), d))


def modulate_beta_for_look_delta(
    stems: Sequence[np.ndarray],
    distances_m: Sequence[float],
    angles_deg: Sequence[float],
    *,
    theta0_deg: float,
    looked_spk_idx: int,
    dis_threshold: float,
    sigma: float = DEFAULT_SIGMA,
    rho: float = DEFAULT_RHO,
    target_delta_db: float = DEFAULT_LOOK_DELTA_CAP_DB,
    sigma_candidates: Sequence[float] = DEFAULT_SIGMA_CANDIDATES,
) -> dict:
    """Search σ so oracle look−no-look SI-SDR for the looked speaker approaches ``target_delta_db``.

    No-look = hard bubble (sum of inside stems). Look = β-weighted mix at ``theta0_deg``.
    Prefers the largest delta ≤ target; if none ≤ target, the smallest overshoot.
    """
    if not stems or looked_spk_idx < 0 or looked_spk_idx >= len(stems):
        z, w = steered_target_from_stems(
            stems, distances_m, angles_deg,
            theta0_deg=theta0_deg, dis_threshold=dis_threshold, sigma=sigma, rho=rho,
        )
        return {
            "sigma": float(sigma),
            "rho": float(rho),
            "z": z,
            "weights": w,
            "oracle_delta_db": float("nan"),
            "capped_delta_db": float("nan"),
        }

    stacked = np.stack([np.asarray(s, dtype=np.float32).reshape(-1) for s in stems], axis=0)
    bubble = np.sum(stacked, axis=0).astype(np.float32)
    spk = stacked[looked_spk_idx]
    sisdr_noolook = si_sdr_db(bubble, spk)

    best = None  # (score, sigma, z, w, raw_delta)
    # score: prefer delta in [0, target], maximize delta; else minimize overshoot
    for sig in sigma_candidates:
        z, w = steered_target_from_stems(
            stems, distances_m, angles_deg,
            theta0_deg=theta0_deg, dis_threshold=dis_threshold, sigma=float(sig), rho=rho,
        )
        raw = si_sdr_db(z, spk) - sisdr_noolook
        if not np.isfinite(raw):
            continue
        if raw <= target_delta_db:
            # Feasible: maximize delta (minimize shortfall to target).
            score = (0, target_delta_db - raw, float(sig))
        else:
            # Overshoot: minimize overshoot; break ties toward wider beam (larger σ).
            score = (1, raw - target_delta_db, -float(sig))
        if best is None or score < best[0]:
            best = (score, float(sig), z, w, float(raw))

    if best is None:
        z, w = steered_target_from_stems(
            stems, distances_m, angles_deg,
            theta0_deg=theta0_deg, dis_threshold=dis_threshold, sigma=sigma, rho=rho,
        )
        raw = si_sdr_db(z, spk) - sisdr_noolook
        return {
            "sigma": float(sigma),
            "rho": float(rho),
            "z": z,
            "weights": w,
            "oracle_delta_db": float(raw) if np.isfinite(raw) else float("nan"),
            "capped_delta_db": float(min(target_delta_db, raw)) if np.isfinite(raw) else float("nan"),
        }

    _, sig_used, z, w, raw = best
    return {
        "sigma": sig_used,
        "rho": float(rho),
        "z": z,
        "weights": w,
        "oracle_delta_db": raw,
        "capped_delta_db": float(min(target_delta_db, raw)),
    }
