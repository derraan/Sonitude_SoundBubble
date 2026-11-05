"""Reusable LibriSpeech distance→dBFS sweep for training validation and CLI eval."""
from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pyroomacoustics as pra
import torch

SR = 24000
DUR_S = 2.0
TARGET_IN_DBFS = -25.0
DEFAULT_DISTANCES = np.round(np.arange(3.0, 0.49, -0.1), 2)
DEFAULT_RT60_LIST = [0.2, 0.8, 1.0, 1.6]
DEFAULT_SPEECH_PATH = r"C:\Users\darre\Sound_Bubble\datasets\_speech_cache\5694-64038-0000.flac"

# Train edge-aux operating region (mild RT60, multi-level).
DEFAULT_EDGE_LEVELS_DBFS = [-10.0, -20.0, -30.0, -40.0]
DEFAULT_EDGE_RT60_LIST = [0.2, 0.35, 0.5]
DEFAULT_EDGE_DISTANCES = [0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 2.0]
DEFAULT_PASSBAND_KEEP_DISTS = [0.8, 1.0, 1.2]
DEFAULT_EDGE_NEAR_M = 1.4
DEFAULT_EDGE_FAR_M = 2.0
DEFAULT_SUPPRESSION_DROP_DB = 6.0

# Elevation-aux (flatten vertical notch, esp. ≈−20°): in-bubble keep across elev.
DEFAULT_ELEV_AUX_ELEVATIONS_DEG = [-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0]
DEFAULT_ELEV_AUX_DISTANCES = [0.8, 1.0, 1.2]
DEFAULT_ELEV_NOTCH_DEG = [-20.0]
DEFAULT_ELEV_REF_DEG = 0.0
DEFAULT_ELEV_MAX_NOTCH_DROP_DB = 3.0
# Tall room so ±60° at ~1.2 m fits; dry RT60 may need absorption fallback.
ELEV_ROOM_DIM = [8.0, 8.0, 6.0]
ELEV_MIC_CENTER = np.array([4.0, 4.0, 3.0], dtype=np.float64)

MIC_OFFSETS_M = np.array(
    [
        [-0.128, -0.015, 0.0],
        [-0.102, 0.0, 0.113],
        [-0.038, 0.0, 0.169],
        [0.038, 0.0, 0.169],
        [0.106, 0.0, 0.117],
        [0.131, -0.015, 0.007],
    ],
    dtype=np.float64,
)


def db_fs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    return 20.0 * math.log10(max(rms, 1e-12))


def load_libri_segment(path: str, sr: int = SR, dur_s: float = DUR_S) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    n = int(sr * dur_s)
    if len(y) < n:
        reps = int(np.ceil(n / max(len(y), 1)))
        y = np.tile(y, reps)
    y = y[:n].astype(np.float32)
    peak = float(np.max(np.abs(y)) + 1e-12)
    return (y / peak * 0.3).astype(np.float32)


def simulate_at_distance(distance_m: float, src: np.ndarray, sr: int, rt60_s: float) -> np.ndarray:
    room_dim = [8.0, 8.0, 3.0]
    mic_center = np.array([4.0, 4.0, 1.5])
    mics = MIC_OFFSETS_M + mic_center
    src_pos = mic_center + np.array([0.0, distance_m, 0.0])
    src_pos = np.clip(src_pos, [0.3, 0.3, 0.3], [room_dim[0] - 0.3, room_dim[1] - 0.3, room_dim[2] - 0.3])

    e_absorption, max_order = pra.inverse_sabine(rt60_s, room_dim)
    max_order = int(min(max_order, 80))
    room = pra.ShoeBox(
        room_dim,
        fs=sr,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )
    room.add_microphone_array(mics.T)
    room.add_source(src_pos, signal=src.astype(np.float64))
    room.simulate()
    mix = np.asarray(room.mic_array.signals, dtype=np.float32)
    n = len(src)
    if mix.shape[1] < n:
        mix = np.concatenate([mix, np.zeros((6, n - mix.shape[1]), dtype=np.float32)], axis=1)
    else:
        mix = mix[:, :n]
    return mix


def normalize_mic0_level(mix: np.ndarray, target_dbfs: float) -> np.ndarray:
    cur = db_fs(mix[0])
    gain = 10.0 ** ((target_dbfs - cur) / 20.0)
    return (mix * gain).astype(np.float32)


def simulate_at_distance_elevation(
    distance_m: float,
    elev_deg: float,
    src: np.ndarray,
    sr: int,
    rt60_s: float,
    *,
    room_dim: Optional[Sequence[float]] = None,
    mic_center: Optional[np.ndarray] = None,
    max_order_cap: int = 30,
) -> np.ndarray:
    """Place source at radial distance with elevation (0=horizontal, + = above).

    Azimuth fixed forward (+Y). Uses a taller room by default so ±60° fits.
    """
    room_dim_f = list(room_dim) if room_dim is not None else list(ELEV_ROOM_DIM)
    center = np.asarray(
        mic_center if mic_center is not None else ELEV_MIC_CENTER, dtype=np.float64
    )
    elev = math.radians(float(elev_deg))
    horiz = float(distance_m) * math.cos(elev)
    z_off = float(distance_m) * math.sin(elev)
    src_pos = center + np.array([0.0, horiz, z_off])
    lo = np.array([0.3, 0.3, 0.3])
    hi = np.asarray(room_dim_f, dtype=np.float64) - 0.3
    src_pos = np.clip(src_pos, lo, hi)

    mics = MIC_OFFSETS_M + center
    try:
        e_absorption, max_order = pra.inverse_sabine(float(rt60_s), room_dim_f)
        max_order = int(min(max_order, max_order_cap))
    except ValueError:
        e_absorption = 0.45
        max_order = int(min(20, max_order_cap))
    room = pra.ShoeBox(
        room_dim_f,
        fs=sr,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )
    room.add_microphone_array(mics.T)
    room.add_source(src_pos, signal=src.astype(np.float64))
    room.simulate()
    mix = np.asarray(room.mic_array.signals, dtype=np.float32)
    n = len(src)
    if mix.shape[1] < n:
        mix = np.concatenate([mix, np.zeros((6, n - mix.shape[1]), dtype=np.float32)], axis=1)
    else:
        mix = mix[:, :n]
    return mix


def build_elev_aux_mix_cache(
    *,
    speech_path: str = DEFAULT_SPEECH_PATH,
    elevations_deg: Optional[Sequence[float]] = None,
    distances: Optional[Sequence[float]] = None,
    target_in_dbfs: float = TARGET_IN_DBFS,
    rt60: float = 0.2,
    sr: int = SR,
    dur_s: float = DUR_S,
) -> Tuple[torch.Tensor, List[float], List[float]]:
    """Pre-simulate level-matched mixes on an elevation × in-bubble distance grid.

    Returns
    -------
    mixes : Tensor [E, D, 6, T] float32 CPU
    elevations_deg, distances_m
    """
    elevs = _as_float_list(elevations_deg, DEFAULT_ELEV_AUX_ELEVATIONS_DEG)
    dists = _as_float_list(distances, DEFAULT_ELEV_AUX_DISTANCES)
    src = load_libri_segment(speech_path, sr=sr, dur_s=dur_s)
    rows = []
    for elev in elevs:
        row = []
        for d in dists:
            mix = simulate_at_distance_elevation(float(d), float(elev), src, sr, float(rt60))
            row.append(torch.from_numpy(normalize_mic0_level(mix, float(target_in_dbfs))))
        rows.append(torch.stack(row, dim=0))
    mixes = torch.stack(rows, dim=0)  # [E, D, 6, T]
    return mixes, elevs, dists


def elev_aux_losses_torch(
    levels_db: torch.Tensor,
    elevations_deg: Sequence[float],
    *,
    target_in_dbfs: float,
    passband_slack_db: float = 6.0,
    notch_elevations_deg: Optional[Sequence[float]] = None,
    ref_elevation_deg: float = DEFAULT_ELEV_REF_DEG,
    max_notch_drop_db: float = DEFAULT_ELEV_MAX_NOTCH_DROP_DB,
) -> Dict[str, torch.Tensor]:
    """Differentiable elevation-aux terms from levels [E] or [E, D].

    Keys: keep_hinge, notch_hinge, elev_ripple_db, passband_mean_by_elev,
    notch_drop_db (ref − worst notch).
    """
    if levels_db.dim() == 2:
        passband_mean = levels_db.mean(dim=1)  # [E]
    else:
        passband_mean = levels_db
    elevs = [float(e) for e in elevations_deg]
    keep_hinge = torch.relu(
        (float(target_in_dbfs) - float(passband_slack_db)) - passband_mean
    ).mean()
    elev_ripple = passband_mean.max() - passband_mean.min()

    notch_list = _as_float_list(notch_elevations_deg, DEFAULT_ELEV_NOTCH_DEG)
    ref_idx = int(np.argmin(np.abs(np.asarray(elevs) - float(ref_elevation_deg))))
    ref_mean = passband_mean[ref_idx]
    notch_hinges = []
    notch_drops = []
    for ne in notch_list:
        n_idx = int(np.argmin(np.abs(np.asarray(elevs) - float(ne))))
        drop = ref_mean - passband_mean[n_idx]
        notch_drops.append(drop)
        notch_hinges.append(torch.relu(drop - float(max_notch_drop_db)))
    if notch_hinges:
        notch_hinge = torch.stack(notch_hinges).mean()
        notch_drop = torch.stack(notch_drops).max()
    else:
        notch_hinge = passband_mean.new_tensor(0.0)
        notch_drop = passband_mean.new_tensor(0.0)

    return {
        "keep_hinge": keep_hinge,
        "notch_hinge": notch_hinge,
        "elev_ripple_db": elev_ripple,
        "passband_mean_by_elev": passband_mean,
        "notch_drop_db": notch_drop,
        "ref_passband_db": ref_mean,
    }


def _as_float_list(values: Optional[Sequence[float]], default: Sequence[float]) -> List[float]:
    if values is None:
        return [float(v) for v in default]
    return [float(v) for v in values]


def passband_ripple_db(
    levels_db: Sequence[float],
    distances_m: Sequence[float],
    bubble_m: float,
    floor_dbfs: float = -60.0,
) -> float:
    """Peak-to-peak output level (dB) for active passband sources inside the bubble.

    Uses distances strictly inside the bubble (d < bubble_m) whose output is above
    ``floor_dbfs`` so the transition from stopband is excluded. Lower is flatter /
    less jittery near the threshold. Returns nan if fewer than 2 points qualify.
    """
    interior = [
        float(lvl)
        for d, lvl in zip(distances_m, levels_db)
        if float(d) < float(bubble_m) and float(lvl) > float(floor_dbfs)
    ]
    if len(interior) < 2:
        return float("nan")
    return float(max(interior) - min(interior))


def db_fs_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Differentiable dBFS for a batch of waveforms. x: [B, ...] → [B]."""
    flat = x.reshape(x.shape[0], -1)
    rms = torch.sqrt(torch.mean(flat * flat, dim=-1) + eps)
    return 20.0 * torch.log10(rms.clamp_min(eps))


def build_passband_mix_cache(
    *,
    speech_path: str = DEFAULT_SPEECH_PATH,
    distances: Optional[Sequence[float]] = None,
    bubble_m: float = 1.5,
    rt60: float = 0.2,
    target_in_dbfs: float = TARGET_IN_DBFS,
    sr: int = SR,
    dur_s: float = DUR_S,
) -> Tuple[torch.Tensor, List[float]]:
    """Pre-simulate level-matched mic mixes at RT60 for all d < bubble_m.

    Returns (mixes [N, 6, T] float32 CPU, distances_m).
    """
    all_d = _as_float_list(distances, DEFAULT_DISTANCES)
    interior_d = [d for d in all_d if d < float(bubble_m)]
    if len(interior_d) < 2:
        raise ValueError(
            f"Need ≥2 distances inside bubble ({bubble_m} m); got {interior_d}"
        )
    src = load_libri_segment(speech_path, sr=sr, dur_s=dur_s)
    mixes = []
    for d in interior_d:
        mix = simulate_at_distance(float(d), src, sr, float(rt60))
        mix = normalize_mic0_level(mix, float(target_in_dbfs))
        mixes.append(torch.from_numpy(mix))
    return torch.stack(mixes, dim=0), interior_d


def passband_ripple_torch(levels_db: torch.Tensor, floor_dbfs: float = -60.0) -> torch.Tensor:
    """Differentiable peak-to-peak ripple among levels above ``floor_dbfs``.

    Soft-masks suppressed / transition points so the loss targets flatness of the
    active passband rather than stopband→passband rise height.
    """
    # Soft gate: ~1 when level >> floor, ~0 when level << floor.
    # Temperature 2 dB keeps gradients usable near the threshold.
    gate = torch.sigmoid((levels_db - float(floor_dbfs)) / 2.0)
    # Softmax / softmin over gated levels (add large negative offset when gated out).
    neg_inf = levels_db.new_tensor(-1.0e4)
    gated = levels_db * gate + neg_inf * (1.0 - gate)
    # Soft max / min via logsumexp
    soft_max = torch.logsumexp(gated, dim=0)
    soft_min = -torch.logsumexp(-gated, dim=0)
    # Fall back to hard max−min among hard-active points when enough exist
    active = levels_db > float(floor_dbfs)
    if int(active.sum().item()) >= 2:
        return levels_db[active].max() - levels_db[active].min()
    return soft_max - soft_min


@torch.no_grad()
def run_distance_sweep(
    model: torch.nn.Module,
    device: str,
    *,
    speech_path: str = DEFAULT_SPEECH_PATH,
    distances: Optional[Sequence[float]] = None,
    rt60_list: Optional[Sequence[float]] = None,
    target_in_dbfs: float = TARGET_IN_DBFS,
    bubble_m: float = 1.5,
    sr: int = SR,
    dur_s: float = DUR_S,
) -> Dict[str, object]:
    """Run model on distance×RT60 grid; return levels and compact summary metrics."""
    distances_f = _as_float_list(distances, DEFAULT_DISTANCES)
    rt60_f = _as_float_list(rt60_list, DEFAULT_RT60_LIST)
    src = load_libri_segment(speech_path, sr=sr, dur_s=dur_s)

    results: Dict[float, List[float]] = {}
    input_levels: List[float] = []

    was_training = model.training
    model.eval()
    try:
        for rt60 in rt60_f:
            outs: List[float] = []
            for i, d in enumerate(distances_f):
                mix = simulate_at_distance(float(d), src, sr, float(rt60))
                mix = normalize_mic0_level(mix, float(target_in_dbfs))
                if rt60 == rt60_f[0]:
                    input_levels.append(db_fs(mix[0]))
                mix_t = torch.from_numpy(mix).unsqueeze(0).to(device)
                out = model({"mixture": mix_t})["output"].squeeze(0).detach().cpu().numpy()
                if out.ndim > 1:
                    out = out[0]
                outs.append(db_fs(out))
            results[float(rt60)] = outs
    finally:
        if was_training:
            model.train()

    # Mean across RT60 at each distance
    mean_out = []
    for i in range(len(distances_f)):
        mean_out.append(float(np.mean([results[rt][i] for rt in results])))

    def _level_at(dist_m: float) -> float:
        # nearest grid point
        idx = int(np.argmin(np.abs(np.asarray(distances_f) - dist_m)))
        return mean_out[idx]

    # Passband ripple on the dry (RT60=0.2s) curve: peak−peak inside bubble.
    # Prefer exact 0.2 match; fall back to nearest listed RT60 if needed.
    ripple_rt60 = 0.2
    if results:
        if ripple_rt60 not in results:
            ripple_rt60 = min(results.keys(), key=lambda r: abs(float(r) - 0.2))
        ripple_db = passband_ripple_db(
            results[float(ripple_rt60)],
            distances_f,
            bubble_m=float(bubble_m),
        )
    else:
        ripple_db = float("nan")

    # -6 dB suppression point vs bubble on the dry (ripple) RT60 curve.
    dry_levels = results.get(float(ripple_rt60), mean_out) if results else mean_out
    dry_ref = float(np.mean([
        dry_levels[int(np.argmin(np.abs(np.asarray(distances_f) - d)))]
        for d in (0.8, 1.0, 1.2)
    ])) if distances_f else float("nan")
    supp_m = (
        suppression_point_m_np(dry_levels, distances_f, dry_ref, drop_db=6.0)
        if distances_f else float("nan")
    )

    summary = {
        "out_dbfs_0_5m": _level_at(0.5),
        "out_dbfs_0_8m": _level_at(0.8),
        "out_dbfs_1_0m": _level_at(1.0),
        "out_dbfs_1_2m": _level_at(1.2),
        "out_dbfs_1_5m": _level_at(1.5),
        "out_dbfs_2_0m": _level_at(2.0),
        "out_dbfs_3_0m": _level_at(3.0),
        # Positive = passband louder than far field (desired shape).
        "passband_minus_far_db": _level_at(0.8) - _level_at(3.0),
        "edge_drop_1_5_to_2_0_db": _level_at(1.5) - _level_at(2.0),
        # Lower = flatter passband on RT60≈0.2 (less jitter near threshold).
        "passband_ripple_0_2_db": ripple_db,
        "suppression_m_m6db": supp_m,
        "suppression_dev_from_bubble_m": (
            abs(supp_m - float(bubble_m)) if supp_m == supp_m else float("nan")
        ),
    }

    return {
        "distances_m": distances_f,
        "rt60_list": rt60_f,
        "input_mic0_dbfs": input_levels,
        "out_dbfs_by_rt60": results,
        "out_dbfs_mean": mean_out,
        "summary": summary,
    }


def write_sweep_csv(path: str, sweep: Dict[str, object]) -> None:
    distances = sweep["distances_m"]  # type: ignore[index]
    results = sweep["out_dbfs_by_rt60"]  # type: ignore[index]
    input_levels = sweep["input_mic0_dbfs"]  # type: ignore[index]
    rt60_list = sweep["rt60_list"]  # type: ignore[index]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        header = "distance_m," + ",".join([f"out_rt60_{rt60}" for rt60 in rt60_list]) + ",input_mic0_dbfs\n"
        f.write(header)
        for i, d in enumerate(distances):
            row = [f"{d}"] + [f"{results[rt60][i]:.4f}" for rt60 in rt60_list] + [f"{input_levels[i]:.4f}"]
            f.write(",".join(row) + "\n")


def _nearest_level(levels_db: torch.Tensor, distances_m: Sequence[float], dist_m: float) -> torch.Tensor:
    idx = int(np.argmin(np.abs(np.asarray(distances_m, dtype=np.float64) - float(dist_m))))
    return levels_db[idx]


def build_edge_aux_mix_cache(
    *,
    speech_path: str = DEFAULT_SPEECH_PATH,
    levels_dbfs: Optional[Sequence[float]] = None,
    rt60_list: Optional[Sequence[float]] = None,
    distances: Optional[Sequence[float]] = None,
    sr: int = SR,
    dur_s: float = DUR_S,
) -> Tuple[Dict[Tuple[float, float], torch.Tensor], List[float], List[float], List[float]]:
    """Pre-simulate mic mixes for every (level, RT60) × short distance set.

    Returns
    -------
    cache : dict[(level_dbfs, rt60)] -> Tensor [D, 6, T] float32 CPU
    distances_m, levels_dbfs, rt60_list
    """
    levels = _as_float_list(levels_dbfs, DEFAULT_EDGE_LEVELS_DBFS)
    rt60s = _as_float_list(rt60_list, DEFAULT_EDGE_RT60_LIST)
    dists = _as_float_list(distances, DEFAULT_EDGE_DISTANCES)
    if len(dists) < 2:
        raise ValueError(f"Need ≥2 edge-aux distances; got {dists}")

    src = load_libri_segment(speech_path, sr=sr, dur_s=dur_s)
    cache: Dict[Tuple[float, float], torch.Tensor] = {}
    for rt60 in rt60s:
        # Simulate once per (rt60, d), then re-normalize to each target level.
        raw_by_d = []
        for d in dists:
            raw_by_d.append(simulate_at_distance(float(d), src, sr, float(rt60)))
        for level in levels:
            mixes = [torch.from_numpy(normalize_mic0_level(m, float(level))) for m in raw_by_d]
            cache[(float(level), float(rt60))] = torch.stack(mixes, dim=0)
    return cache, dists, levels, rt60s


def suppression_point_m_torch(
    levels_db: torch.Tensor,
    distances_m: Sequence[float],
    ref_level_db: torch.Tensor,
    *,
    drop_db: float = DEFAULT_SUPPRESSION_DROP_DB,
) -> torch.Tensor:
    """Soft estimate of the distance where level crosses ``ref − drop_db``.

    Prefers linear interpolation on adjacent pairs that straddle the threshold
    (descending edge). Falls back to a soft nearest-bin estimate if no pair
    straddles cleanly. Distances are assumed sorted ascending.
    """
    thresh = ref_level_db - float(drop_db)
    d = torch.as_tensor(distances_m, device=levels_db.device, dtype=levels_db.dtype)
    if levels_db.numel() < 2:
        return d[0]

    crossings = []
    weights = []
    for i in range(int(levels_db.shape[0]) - 1):
        l0 = levels_db[i]
        l1 = levels_db[i + 1]
        d0 = d[i]
        d1 = d[i + 1]
        # High when the pair drops through the threshold (l0 >= thresh >= l1).
        w = torch.sigmoid((l0 - thresh) / 1.0) * torch.sigmoid((thresh - l1) / 1.0)
        denom = (l0 - l1).abs().clamp_min(1e-3)
        frac = ((l0 - thresh) / denom).clamp(0.0, 1.0)
        # If rising instead of falling, still produce a value; weight stays tiny.
        cross = d0 + (d1 - d0) * frac
        crossings.append(cross)
        weights.append(w)

    wstack = torch.stack(weights)
    cstack = torch.stack(crossings)
    wsum = wstack.sum().clamp_min(1e-6)
    soft_cross = (wstack * cstack).sum() / wsum

    dist_to_thresh = (levels_db - thresh).abs()
    softmin = torch.softmax(-dist_to_thresh / 2.0, dim=0)
    nearest = (softmin * d).sum()
    gate = torch.sigmoid((wsum - 0.05) / 0.05)
    return gate * soft_cross + (1.0 - gate) * nearest


def suppression_point_m_np(
    levels_db: Sequence[float],
    distances_m: Sequence[float],
    ref_level_db: float,
    *,
    drop_db: float = DEFAULT_SUPPRESSION_DROP_DB,
) -> float:
    """Hard -drop_db crossing distance (metres); nan if none."""
    thresh = float(ref_level_db) - float(drop_db)
    d = [float(x) for x in distances_m]
    lvl = [float(x) for x in levels_db]
    order = np.argsort(np.asarray(d))
    d = [d[i] for i in order]
    lvl = [lvl[i] for i in order]
    for i in range(len(d) - 1):
        l0, l1 = lvl[i], lvl[i + 1]
        if l0 >= thresh >= l1 and (l0 - l1) > 1e-6:
            frac = (l0 - thresh) / (l0 - l1)
            return float(d[i] + (d[i + 1] - d[i]) * frac)
    return float("nan")


def edge_aux_losses_torch(
    levels_db: torch.Tensor,
    distances_m: Sequence[float],
    *,
    target_in_dbfs: float,
    passband_dists: Optional[Sequence[float]] = None,
    edge_near_m: float = DEFAULT_EDGE_NEAR_M,
    edge_far_m: float = DEFAULT_EDGE_FAR_M,
    passband_slack_db: float = 6.0,
    min_edge_drop_db: float = 20.0,
    bubble_m: float = 1.5,
    ripple_floor_dbfs: float = -60.0,
    suppression_drop_db: float = DEFAULT_SUPPRESSION_DROP_DB,
) -> Dict[str, torch.Tensor]:
    """Differentiable edge-aux terms for one (level, RT60) condition.

    Keys: keep_hinge, edge_hinge, ripple, suppression_dev,
    passband_mean_db, edge_drop_db, suppression_m.
    """
    pb_dists = _as_float_list(passband_dists, DEFAULT_PASSBAND_KEEP_DISTS)
    pb_levels = [_nearest_level(levels_db, distances_m, d) for d in pb_dists]
    passband_mean = torch.stack(pb_levels).mean()
    # Penalize going quiet relative to the matched mic0 target.
    keep_hinge = torch.relu((float(target_in_dbfs) - float(passband_slack_db)) - passband_mean)

    near = _nearest_level(levels_db, distances_m, edge_near_m)
    far = _nearest_level(levels_db, distances_m, edge_far_m)
    edge_drop = near - far
    edge_hinge = torch.relu(float(min_edge_drop_db) - edge_drop)

    interior_mask = [float(d) < float(bubble_m) for d in distances_m]
    if any(interior_mask):
        interior_levels = torch.stack(
            [levels_db[i] for i, m in enumerate(interior_mask) if m]
        )
        ripple = passband_ripple_torch(interior_levels, floor_dbfs=ripple_floor_dbfs)
    else:
        ripple = levels_db.new_tensor(0.0)

    # Sort distances ascending for crossing estimate (edge cache is already sorted).
    order = list(np.argsort(np.asarray(distances_m, dtype=np.float64)))
    levels_sorted = torch.stack([levels_db[i] for i in order])
    dists_sorted = [float(distances_m[i]) for i in order]
    suppression_m = suppression_point_m_torch(
        levels_sorted,
        dists_sorted,
        passband_mean,
        drop_db=float(suppression_drop_db),
    )
    suppression_dev = (suppression_m - float(bubble_m)).abs()

    return {
        "keep_hinge": keep_hinge,
        "edge_hinge": edge_hinge,
        "ripple": ripple,
        "suppression_dev": suppression_dev,
        "passband_mean_db": passband_mean,
        "edge_drop_db": edge_drop,
        "suppression_m": suppression_m,
    }


def transfer_mag_db_hann(
    x: np.ndarray,
    y: np.ndarray,
    sr: int = SR,
    nperseg: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """|H(f)| dB via Welch cross/auto spectrum with an explicit Hann window."""
    from scipy import signal

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(len(x), len(y))
    skip = int(0.25 * sr)
    x = x[skip:n]
    y = y[skip:n]
    win = signal.windows.hann(nperseg, sym=False)
    f, pxy = signal.csd(x, y, fs=sr, window=win, nperseg=nperseg, noverlap=nperseg // 2)
    _, pxx = signal.welch(x, fs=sr, window=win, nperseg=nperseg, noverlap=nperseg // 2)
    h = pxy / (pxx + 1e-20)
    mag_db = 20.0 * np.log10(np.abs(h) + 1e-20)
    return f.astype(np.float64), mag_db.astype(np.float64)


def freq_passband_margin_db(
    mag_db: np.ndarray,
    freqs_hz: np.ndarray,
    f_lo: float = 300.0,
    f_hi: float = 3000.0,
) -> float:
    """Peak−peak |H(f)| margin in [f_lo, f_hi] Hz (eval metric; not a train loss)."""
    mask = (freqs_hz >= float(f_lo)) & (freqs_hz <= float(f_hi))
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    band = np.asarray(mag_db, dtype=np.float64)[mask]
    return float(band.max() - band.min())


@torch.no_grad()
def run_level_rt60_distance_grid(
    model: torch.nn.Module,
    device: str,
    *,
    speech_path: str = DEFAULT_SPEECH_PATH,
    levels_dbfs: Optional[Sequence[float]] = None,
    rt60_list: Optional[Sequence[float]] = None,
    distances: Optional[Sequence[float]] = None,
    bubble_m: float = 1.5,
    sr: int = SR,
    dur_s: float = DUR_S,
) -> Dict[str, object]:
    """Full level×RT60×distance sweep with per-cell summary metrics."""
    levels = _as_float_list(levels_dbfs, [-10.0, -20.0, -25.0, -30.0, -40.0])
    rt60s = _as_float_list(rt60_list, [0.2, 0.3, 0.4, 0.5])
    distances_f = _as_float_list(distances, DEFAULT_DISTANCES)
    src = load_libri_segment(speech_path, sr=sr, dur_s=dur_s)

    rows: List[Dict[str, float]] = []
    was_training = model.training
    model.eval()
    try:
        for level in levels:
            for rt60 in rt60s:
                outs: List[float] = []
                for d in distances_f:
                    mix = simulate_at_distance(float(d), src, sr, float(rt60))
                    mix = normalize_mic0_level(mix, float(level))
                    mix_t = torch.from_numpy(mix).unsqueeze(0).to(device)
                    out = model({"mixture": mix_t})["output"].squeeze(0).detach().cpu().numpy()
                    if out.ndim > 1:
                        out = out[0]
                    outs.append(db_fs(out))

                def _at(dist_m: float) -> float:
                    idx = int(np.argmin(np.abs(np.asarray(distances_f) - dist_m)))
                    return float(outs[idx])

                interior = [
                    float(lvl)
                    for d, lvl in zip(distances_f, outs)
                    if float(d) < float(bubble_m) and float(lvl) > -60.0
                ]
                ripple = float(max(interior) - min(interior)) if len(interior) >= 2 else float("nan")
                near_mean = float(
                    np.mean([_at(d) for d in (0.8, 1.0, 1.2)])
                )
                supp_m = suppression_point_m_np(outs, distances_f, near_mean, drop_db=6.0)
                rows.append(
                    {
                        "level_dbfs": float(level),
                        "rt60": float(rt60),
                        "passband_mean_db": near_mean,
                        "ripple_in_db": ripple,
                        "out_1_5m": _at(1.5),
                        "out_2_0m": _at(2.0),
                        "edge_drop_1_5_to_2_0_db": _at(1.5) - _at(2.0),
                        "pass_minus_far_db": _at(0.8) - _at(3.0),
                        "suppression_m_m6db": supp_m,
                        "suppression_dev_from_bubble_m": (
                            abs(supp_m - float(bubble_m)) if supp_m == supp_m else float("nan")
                        ),
                    }
                )
    finally:
        if was_training:
            model.train()

    return {"rows": rows, "distances_m": distances_f, "levels_dbfs": levels, "rt60_list": rt60s}
