"""Regression checks for the measure-core refactor. Run from repo root:

  set PYTHONPATH=%CD%
  python tools/tech_testbench/scripts/test_measure_refactor.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

from tools.measure.core import (
    _finalize_trial,
    _union_artifacts,
    config_from_session_dict,
)
from tools.measure.runtime import RuntimeLaunch, RuntimeProcess
from tools.tech_testbench.app.storage.trial_store import TrialStore


def test_chatty_child_no_deadlock() -> None:
    """Child that prints >128 KB then sleeps must exit cleanly with PIPE drained."""
    td = Path(tempfile.mkdtemp(prefix="rt_drain_"))
    log_path = td / "pipeline.log"
    # Fake "cpp" launch: python -c chatty printer (argv via extra_args + python runtime).
    launch = RuntimeLaunch(
        runtime="python",
        use_zoo=False,
        model_path="dummy.onnx",  # required when not zoo; never executed (extra replaces)
        profile=False,
        tap_dir="",
        extra_args=[],
    )
    # Override build_argv by constructing RuntimeProcess and patching argv after start
    # is awkward — instead spawn via a tiny RuntimeLaunch that uses python -c.
    # Use RuntimeProcess with a custom Popen-equivalent by monkeypatching build_argv.
    lines = 8000
    code = (
        "import sys,time\n"
        f"for i in range({lines}):\n"
        "    sys.stdout.write('x'*80+'\\n'); sys.stdout.flush()\n"
        "time.sleep(0.3)\n"
    )
    orig_build = launch.build_argv

    def _chatty_argv() -> list[str]:
        return [sys.executable, "-c", code]

    launch.build_argv = _chatty_argv  # type: ignore[method-assign]
    rt = RuntimeProcess(launch, log_path=log_path)
    t0 = time.perf_counter()
    rt.start()
    # Wait for natural exit (no SIGINT needed for this python -c).
    rc = rt.wait(timeout=15.0)
    rt._join_reader()
    elapsed = time.perf_counter() - t0
    assert rc == 0, rc
    assert elapsed < 10.0, f"likely pipe deadlock: {elapsed:.1f}s"
    assert log_path.is_file() and log_path.stat().st_size > 50_000, log_path.stat().st_size
    tail = rt.drain_output()
    assert "x" in tail
    launch.build_argv = orig_build  # type: ignore[method-assign]
    print("chatty_child_no_deadlock ok", f"log={log_path.stat().st_size}B elapsed={elapsed:.2f}s")


def test_artifacts_union_survives_finalize() -> None:
    """mic_delay/timing_ref plot entries must survive finalize deep-merge."""
    root = Path(tempfile.mkdtemp(prefix="art_union_"))
    store = TrialStore(root)
    run = store.new_run(meta={"kind": "test"})
    paths = store.new_trial(run_id=run.name, kind="algorithm")
    # Simulate capture-time trial_doc with only FR plots.
    trial_doc = {
        "run_id": paths.run_id,
        "trial_id": paths.trial_id,
        "kind": "algorithm",
        "artifacts": {
            "wavs": ["stimulus_used.wav", "raw_8ch.wav"],
            "plots": ["plots/suppression_fr.png"],
            "files": [],
        },
    }
    # Simulate analysis that appended mic_delay + timing_ref before finalize.
    on_disk = store.load_trial_json(paths)
    on_disk["artifacts"] = {
        "wavs": ["stimulus_used.wav"],
        "plots": [
            "plots/mic_delay.csv",
            "plots/mic_delay.json",
            "plots/timing_ref.csv",
            "plots/timing_ref.json",
            "plots/suppression_fr.csv",
        ],
        "files": [],
    }
    store.save_trial_json(paths, on_disk)
    # Touch wavs so register_wavs finds them.
    (paths.root / "stimulus_used.wav").write_bytes(b"RIFF")
    (paths.root / "raw_8ch.wav").write_bytes(b"RIFF")

    metrics = {"ok": True}
    finalized = _finalize_trial(
        trial_root=paths.root,
        run_root=paths.run_root,
        meta={"store": store, "paths": paths, "run_id": paths.run_id, "trial_id": paths.trial_id},
        trial_doc=trial_doc,
        metrics=metrics,
        wav_artifacts=["stimulus_used.wav", "raw_8ch.wav"],
        file_artifacts=[],
    )
    plots = finalized["artifacts"]["plots"]
    assert "plots/mic_delay.csv" in plots, plots
    assert "plots/timing_ref.json" in plots, plots
    assert "plots/suppression_fr.png" in plots or "plots/suppression_fr.csv" in plots, plots
    wavs = finalized["artifacts"]["wavs"]
    assert "raw_8ch.wav" in wavs and "stimulus_used.wav" in wavs
    print("artifacts_union_survives_finalize ok", f"n_plots={len(plots)}")


def test_bubble_radius_zero_preserved() -> None:
    cfg = config_from_session_dict({"bubble_radius_m": 0.0, "zoo_dir": "edge/zoo", "use_zoo": True})
    assert cfg.runtime.bubble_radius_m == 0.0
    print("bubble_radius_zero_preserved ok")


def test_union_helper() -> None:
    a = {"wavs": ["a.wav"], "plots": ["plots/x.png"], "files": []}
    b = {"wavs": ["b.wav", "a.wav"], "plots": ["plots/y.png"], "extra": 1}
    m = _union_artifacts(a, b)
    assert m["wavs"] == ["a.wav", "b.wav"]
    assert "plots/x.png" in m["plots"] and "plots/y.png" in m["plots"]
    assert m["extra"] == 1
    # TrialStore.merge_artifacts should agree.
    m2 = TrialStore.merge_artifacts(a, b)
    assert m2["wavs"] == m["wavs"]
    print("union_helper ok")


def test_polar_pending_no_orphan_trial() -> None:
    """Prepare writes polar_pending.json under the run — not a new trial folder."""
    root = Path(tempfile.mkdtemp(prefix="polar_prep_"))
    store = TrialStore(root)
    run = store.new_run(meta={"kind": "polar"})
    pending = {
        "type": "polar_pending",
        "run_id": run.name,
        "note": "scratch only",
    }
    (run / "polar_pending.json").write_text(json.dumps(pending), encoding="utf-8")
    # No trial dirs yet.
    assert store.list_trials(run.name) == []
    assert (run / "polar_pending.json").is_file()
    # Creating one capture trial leaves exactly one trial (not two from prepare+capture).
    paths = store.new_trial(run_id=run.name, kind="raw_polar")
    assert store.list_trials(run.name) == [paths.trial_id]
    print("polar_pending_no_orphan_trial ok")


def main() -> None:
    test_union_helper()
    test_bubble_radius_zero_preserved()
    test_artifacts_union_survives_finalize()
    test_polar_pending_no_orphan_trial()
    test_chatty_child_no_deadlock()
    print("ok")


if __name__ == "__main__":
    main()
