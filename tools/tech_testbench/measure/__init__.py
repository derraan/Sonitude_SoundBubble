"""Compatibility shim — measure core lives in ``tools.measure``.

Prefer::

    from tools.measure import run_algorithm_trial, config_from_session_dict
"""

from __future__ import annotations

from tools.measure.core import (
    MeasureConfig,
    TrialGeometry,
    config_from_session_dict,
    run_algorithm_trial,
    run_raw_polar_trial,
)
from tools.measure.parity import run_parity
from tools.measure.runtime import Gains, RuntimeLaunch

__all__ = [
    "Gains",
    "MeasureConfig",
    "RuntimeLaunch",
    "TrialGeometry",
    "config_from_session_dict",
    "run_algorithm_trial",
    "run_parity",
    "run_raw_polar_trial",
]
