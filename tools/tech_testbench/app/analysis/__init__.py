"""Offline tap / HATS analysis for the tech testbench TrialStore layout."""

from .aggregate import analyze_run, write_aggregate_plots
from .e2e_latency import analyze_e2e_latency_trial, measure_e2e_latency
from .metrics import analyze_trial_dir, write_trial_analysis
from .mic_delay import analyze_mic_delay_trial, relative_delays_from_multich
from .timing_ref import analyze_timing_ref_trial, extract_timing_template, locate_timing_ref
from .cli import main

__all__ = [
    "analyze_trial_dir",
    "write_trial_analysis",
    "analyze_run",
    "write_aggregate_plots",
    "analyze_mic_delay_trial",
    "relative_delays_from_multich",
    "analyze_e2e_latency_trial",
    "measure_e2e_latency",
    "analyze_timing_ref_trial",
    "extract_timing_template",
    "locate_timing_ref",
    "main",
]
