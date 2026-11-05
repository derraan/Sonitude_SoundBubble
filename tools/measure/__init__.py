"""Host-agnostic tech-test measure core (selectable ONNX RT + taps + HATS)."""

from __future__ import annotations

__all__ = [
    "__version__",
    "run_algorithm_trial",
    "run_parity",
    "analyze_run",
    "Gains",
    "RuntimeLaunch",
]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "run_algorithm_trial":
        from tools.measure.core import run_algorithm_trial as fn

        return fn
    if name == "run_parity":
        from tools.measure.parity import run_parity as fn

        return fn
    if name == "analyze_run":
        from tools.tech_testbench.app.analysis import analyze_run as fn

        return fn
    if name == "Gains":
        from tools.measure.runtime import Gains as cls

        return cls
    if name == "RuntimeLaunch":
        from tools.measure.runtime import RuntimeLaunch as cls

        return cls
    raise AttributeError(name)
