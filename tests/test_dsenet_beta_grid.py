"""Unit tests for DSENet β / full-circle θ0 grid helpers."""
from __future__ import annotations

import numpy as np
import pytest

from src.datasets.dsenet_beta import (
    DEFAULT_LOOK_DELTA_CAP_DB,
    DEFAULT_THETA0_GRID_DEG,
    make_theta0_grid_deg,
    modulate_beta_for_look_delta,
    nearest_theta0_deg,
    wrap_angle_deg,
)


def test_make_theta0_grid_deg_30_full_circle():
    grid = make_theta0_grid_deg(30.0)
    assert len(grid) == 12
    assert grid[0] == 0.0
    assert grid[-1] == 330.0
    assert grid == DEFAULT_THETA0_GRID_DEG


def test_nearest_theta0_wraps_across_zero():
    assert nearest_theta0_deg(350.0) == 0.0
    assert nearest_theta0_deg(-10.0) == 0.0
    assert nearest_theta0_deg(45.0) in (30.0, 60.0)


def test_wrap_angle_deg():
    assert float(wrap_angle_deg(370.0)) == pytest.approx(10.0)
    assert float(wrap_angle_deg(-190.0)) == pytest.approx(170.0)


def test_modulate_beta_caps_look_delta():
    T = 4000
    t = np.linspace(0, 1, T, dtype=np.float32)
    s0 = 0.5 * np.sin(2 * np.pi * 220 * t)
    s1 = 0.5 * np.sin(2 * np.pi * 440 * t)
    mod = modulate_beta_for_look_delta(
        [s0, s1],
        [1.0, 1.2],
        [0.0, 180.0],
        theta0_deg=0.0,
        looked_spk_idx=0,
        dis_threshold=1.5,
        target_delta_db=DEFAULT_LOOK_DELTA_CAP_DB,
    )
    assert np.isfinite(mod["capped_delta_db"])
    assert mod["capped_delta_db"] <= DEFAULT_LOOK_DELTA_CAP_DB + 1e-6
    assert 0.0 < mod["sigma"] <= 1.0
