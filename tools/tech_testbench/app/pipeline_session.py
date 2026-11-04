"""Long-lived ONNX RT session shared across Polar / Algorithm / E2E takes."""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from tools.measure.runtime import Gains, REPO_ROOT, RuntimeLaunch, RuntimeProcess, sleep_ms


def _stray_rt_pids() -> list[int]:
    """PIDs of leftover soundbubble-rt processes (exclusive device holders)."""
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None
    pids: list[int] = []
    if psutil is not None:
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = (proc.info.get("name") or "").lower()
            except (psutil.Error, TypeError):
                continue
            if "soundbubble-rt" in name:
                pids.append(int(proc.info["pid"]))
        return pids
    # Fallback without psutil (Windows).
    if os.name == "nt":
        import subprocess

        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", "IMAGENAME eq soundbubble-rt.exe", "/FO", "CSV", "/NH"],
                text=True,
                errors="replace",
            )
            for line in out.splitlines():
                parts = [p.strip().strip('"') for p in line.split(",")]
                if len(parts) >= 2 and parts[0].lower().startswith("soundbubble-rt"):
                    try:
                        pids.append(int(parts[1]))
                    except ValueError:
                        pass
        except (OSError, subprocess.SubprocessError):
            pass
    return pids


def launch_from_session(session: Any) -> RuntimeLaunch:
    """Build a RuntimeLaunch from tech_testbench SessionSettings."""
    g = getattr(session, "gains", None)
    gains = Gains(
        input_gain_db=float(getattr(g, "input_gain_db", 0.0)),
        output_gain_db=float(getattr(g, "output_gain_db", 0.0)),
        mix_dry=float(getattr(g, "mix_dry", 0.0)),
        output_limiter=bool(getattr(g, "output_limiter", True)),
        limiter_release_ms=float(getattr(g, "limiter_release_ms", 25.0)),
        mic_gains=[float(v) for v in (getattr(g, "mic_gains", None) or [])],
    )
    use_zoo = bool(getattr(session, "use_zoo", False))
    win = os.name == "nt"
    return RuntimeLaunch(
        runtime=str(getattr(session, "runtime", "cpp") or "cpp"),
        pipeline_in_device=getattr(session, "pipeline_in_device", None),
        pipeline_out_device=getattr(session, "pipeline_out_device", None),
        input_sr=int(getattr(session, "input_sr", 24000) or 24000),
        output_sr=int(getattr(session, "output_sr", 24000) or 24000),
        channel_map=str(getattr(session, "channel_map", "5,4,3,2,1,0") or ""),
        gains=gains,
        model_path=str(getattr(session, "model_path", "") or ""),
        contract_path=str(getattr(session, "contract_path", "") or ""),
        zoo_dir=str(getattr(session, "zoo_dir", "") or ""),
        bubble_radius_m=float(getattr(session, "bubble_radius_m", 1.5) or 1.5),
        use_zoo=use_zoo,
        soundbubble_rt_path=str(getattr(session, "soundbubble_rt_path", "") or ""),
        tap_dir="",
        tap_select="raw,mapped,dry,wet",
        profile=True,
        monitor=True,
        # Pi deploy uses -80; C++ default -50 zeros quiet mics before input_gain.
        silence_dbfs=-80.0,
        # Windows two-stream WDM-KS: fifo avoids latest-wins drain→permanent shed;
        # deeper queue/ahead cushions absorb callback jitter without dropping.
        latency="high" if win else "low",
        queue_policy="fifo" if win else None,
        max_capture_blocks=16 if win else None,
        target_output_ahead_sec=0.12 if win else None,
        max_output_ahead_sec=0.25 if win else None,
        output_device_block=96 if win else None,
        # 64 ms default overflows under Windows hitch → was hard-clearing DAC cushion.
        model_ring_ms=256.0 if win else None,
        # 1 hop/block avoids dry/silence splice every other hop (buzz when keep opens).
        hops_per_io=1 if win else None,
        # Single ORT thread — multi-thread spikes to 60ms+ and underruns on this box.
        intra_op_threads=1 if win else None,
        inter_op_threads=1 if win else None,
    )


class PipelineState(str, Enum):
    STOPPED = "Stopped"
    STARTING = "Starting"
    WARMING = "Warming"
    READY = "Ready"
    FAULT = "Fault"


_PROFILE_RE = re.compile(
    r"\[profile\]\s+mode=(?P<mode>\w+)"
    r".*?underrun_frames=(?P<underrun>\d+)"
    r".*?model_out=(?P<model_out>-?[0-9.]+)dBFS",
    re.IGNORECASE,
)


@dataclass
class ProfileSnap:
    mode: str = ""
    model_out_db: float | None = None
    underrun_frames: int | None = None
    underrun_delta: int | None = None
    raw: str = ""


def _parse_latest_profile(text: str) -> ProfileSnap | None:
    snap: ProfileSnap | None = None
    for line in text.splitlines():
        m = _PROFILE_RE.search(line)
        if not m:
            continue
        snap = ProfileSnap(
            mode=m.group("mode").lower(),
            model_out_db=float(m.group("model_out")),
            underrun_frames=int(m.group("underrun")),
            raw=line.strip(),
        )
    return snap


class PipelineSession:
    """Owns one RuntimeProcess; Start once, then retarget taps per take."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._rt: RuntimeProcess | None = None
        self._state = PipelineState.STOPPED
        self._status = "Pipeline stopped"
        self._last_profile: ProfileSnap | None = None
        self._underrun_baseline: int | None = None
        self._listeners: list[Callable[[], None]] = []
        self._warm_thread: threading.Thread | None = None
        self._log_path: Path | None = None

    def _annotate_underrun_delta(self, snap: ProfileSnap) -> ProfileSnap:
        """Attach underrun growth since Ready so idle lifetime totals are readable."""
        total = snap.underrun_frames
        if total is None:
            return snap
        if self._underrun_baseline is None:
            self._underrun_baseline = int(total)
        snap.underrun_delta = max(0, int(total) - int(self._underrun_baseline))
        return snap

    @property
    def launch(self) -> RuntimeLaunch | None:
        """Settings the running pipeline was started with (None when stopped)."""
        rt = self._rt
        return rt.launch if rt is not None else None

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def status_text(self) -> str:
        return self._status

    @property
    def last_profile(self) -> ProfileSnap | None:
        return self._last_profile

    @property
    def is_ready(self) -> bool:
        return self._state == PipelineState.READY and self.is_alive()

    def is_alive(self) -> bool:
        with self._lock:
            return self._rt is not None and self._rt.poll() is None

    def add_listener(self, cb: Callable[[], None]) -> None:
        self._listeners.append(cb)

    def _notify(self) -> None:
        for cb in list(self._listeners):
            try:
                cb()
            except Exception:
                pass

    def _set_state(self, state: PipelineState, status: str) -> None:
        self._state = state
        self._status = status
        self._notify()

    def _fault_status(self, prefix: str) -> str:
        """Include audio error line + SR hint when the child dies early."""
        detail = ""
        log_txt = ""
        with self._lock:
            rt = self._rt
            log_path = self._log_path
        if rt is not None:
            log_txt = rt.drain_output()
        if not log_txt and log_path is not None:
            try:
                log_txt = Path(log_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                log_txt = ""
        for key in ("[audio-runtime-error]", "[audio-config-error]", "[worker-error]"):
            for line in log_txt.splitlines():
                if key in line:
                    detail = line.strip()
                    break
            if detail:
                break
        launch = rt.launch if rt is not None else None
        sr_bit = ""
        if launch is not None:
            sr_bit = f" in={launch.pipeline_in_device}@{launch.input_sr}Hz out={launch.pipeline_out_device}@{launch.output_sr}Hz"
        hint = ""
        low = (detail + sr_bit).lower()
        if "invalid device" in low or "invalid sample rate" in low:
            hint = " — Pico WDM-KS is 24000 Hz only; wrong SR often reports as Invalid device"
        if detail:
            return f"Pipeline {prefix}:{sr_bit} | {detail}{hint}"
        return f"Pipeline {prefix}:{sr_bit}{hint}"

    def start(
        self,
        launch: RuntimeLaunch,
        *,
        log_path: str | Path | None = None,
        warm_timeout_s: float = 12.0,
        ready_normals: int = 3,
        warm_settle_ms: float = 1500.0,
    ) -> None:
        """Launch RT (no per-take tap_dir) and warm until Ready or Fault."""
        with self._lock:
            if self._rt is not None and self._rt.poll() is None:
                raise RuntimeError("pipeline already running — Stop first")
            stray = _stray_rt_pids()
            if stray:
                raise RuntimeError(
                    "another soundbubble-rt is still running "
                    f"(pid {', '.join(str(p) for p in stray)}) — Stop Pipeline or "
                    "kill that process; WDM-KS exclusive open will fail as Invalid device"
                )
            # Global session: no tap_dir until a take retargets.
            win = os.name == "nt"
            launch = replace(
                launch,
                tap_dir="",
                profile=True,
                monitor=True,
                silence_dbfs=-80.0,
                latency="high" if win else (launch.latency or "low"),
                queue_policy="fifo" if win else launch.queue_policy,
                max_capture_blocks=16 if win else launch.max_capture_blocks,
                target_output_ahead_sec=0.12 if win else launch.target_output_ahead_sec,
                max_output_ahead_sec=0.25 if win else launch.max_output_ahead_sec,
                model_ring_ms=256.0 if win else launch.model_ring_ms,
                hops_per_io=1 if win else launch.hops_per_io,
                intra_op_threads=1 if win else launch.intra_op_threads,
                inter_op_threads=1 if win else launch.inter_op_threads,
                output_device_block=96 if win else launch.output_device_block,
            )
            self._log_path = Path(log_path) if log_path else (REPO_ROOT / "measure" / "_pipeline_session.log")
            self._underrun_baseline = None
            self._last_profile = None
            self._rt = RuntimeProcess(launch, log_path=self._log_path)
            self._set_state(PipelineState.STARTING, "Starting pipeline…")
            self._rt.start()

        self._warm_thread = threading.Thread(
            target=self._warm_loop,
            kwargs={
                "warm_timeout_s": warm_timeout_s,
                "ready_normals": ready_normals,
                "warm_settle_ms": warm_settle_ms,
            },
            name="pipeline-warm",
            daemon=True,
        )
        self._warm_thread.start()

    def _warm_loop(
        self,
        *,
        warm_timeout_s: float,
        ready_normals: int,
        warm_settle_ms: float,
    ) -> None:
        sleep_ms(warm_settle_ms)
        with self._lock:
            if self._rt is None:
                return
            if self._rt.poll() is not None:
                self._set_state(
                    PipelineState.FAULT,
                    self._fault_status(f"exited during start rc={self._rt.poll()}"),
                )
                return
            self._set_state(PipelineState.WARMING, "Warming — waiting for mode=normal…")

        deadline = time.monotonic() + max(1.0, warm_timeout_s)
        normals = 0
        while time.monotonic() < deadline:
            with self._lock:
                rt = self._rt
            if rt is None:
                return
            if rt.poll() is not None:
                self._set_state(
                    PipelineState.FAULT,
                    self._fault_status(f"exited while warming rc={rt.poll()}"),
                )
                return
            snap = _parse_latest_profile(rt.drain_output())
            if snap is not None:
                snap = self._annotate_underrun_delta(snap)
                self._last_profile = snap
                snippet = (
                    f"mode={snap.mode} model_out={snap.model_out_db:.1f} dBFS "
                    f"underrun=+{snap.underrun_delta} (Σ{snap.underrun_frames})"
                )
                if snap.mode == "normal":
                    normals += 1
                    self._set_state(
                        PipelineState.WARMING,
                        f"Warming ({normals}/{ready_normals} normal) — {snippet}",
                    )
                    if normals >= ready_normals:
                        # Baseline at Ready so post-warm idle growth is obvious.
                        if snap.underrun_frames is not None:
                            self._underrun_baseline = int(snap.underrun_frames)
                            snap.underrun_delta = 0
                        ready_snip = (
                            f"mode={snap.mode} model_out={snap.model_out_db:.1f} dBFS "
                            f"underrun=+0 (Σ{snap.underrun_frames})"
                        )
                        self._set_state(PipelineState.READY, f"Ready — {ready_snip}")
                        return
                else:
                    normals = 0
                    self._set_state(
                        PipelineState.WARMING,
                        f"Warming (reset; saw {snap.mode}) — {snippet}",
                    )
            time.sleep(0.2)

        snap = self._last_profile
        detail = (
            f"mode={snap.mode} model_out={snap.model_out_db} underrun={snap.underrun_frames}"
            if snap
            else "no profile lines"
        )
        with self._lock:
            alive = self._rt is not None and self._rt.poll() is None
        # Never soft-Ready on permanent shed — that allowed polar takes on a dead wet path.
        if alive and normals > 0:
            self._set_state(
                PipelineState.READY,
                f"Ready (warm timeout; {normals} normal seen; {detail})",
            )
        elif alive:
            self._set_state(
                PipelineState.FAULT,
                f"Warm failed — never reached mode=normal ({detail})",
            )
        else:
            self._set_state(PipelineState.FAULT, f"Warm failed — {detail}")

    def stop(self, grace_s: float = 4.0) -> int:
        with self._lock:
            rt = self._rt
            self._rt = None
        if rt is None:
            self._set_state(PipelineState.STOPPED, "Pipeline stopped")
            return 0
        self._set_state(PipelineState.STOPPED, "Stopping pipeline…")
        rc = rt.stop(grace_s=grace_s)
        self._set_state(PipelineState.STOPPED, f"Pipeline stopped (rc={rc})")
        return rc

    def send_cmd(self, line: str) -> None:
        with self._lock:
            if self._rt is None or self._rt.poll() is not None:
                raise RuntimeError("pipeline is not running")
            self._rt.send_cmd(line)

    def tap_begin(self, trial_dir: str | Path, *, select: str = "raw,mapped,dry,wet") -> None:
        """Point taps at trial_dir and arm for the take."""
        path = str(Path(trial_dir))
        self.send_cmd(f'TAP dir "{path}"')
        # Wait briefly for ack when possible.
        with self._lock:
            rt = self._rt
        if rt is not None:
            rt.wait_for_stdout("[tap] dir=", timeout_s=2.0)
        self.send_cmd("TAP arm")
        if rt is not None:
            rt.wait_for_stdout("[tap] armed", timeout_s=2.0)

    def tap_end(self, *, wait_flush_s: float = 6.0, trial_dir: str | Path | None = None) -> None:
        """Flush tap WAVs and disarm; leave RT running.

        Waits for a *new* ``[tap] flush complete`` after this flush command so a
        previous take's ack still sitting in the stdout ring cannot end the wait
        early (that race left ``taps_missing`` on back-to-back live takes).
        """
        with self._lock:
            rt = self._rt
        mark = rt.stdout_mark() if rt is not None else 0
        self.send_cmd("TAP flush")
        if rt is not None:
            ok = rt.wait_for_stdout(
                "[tap] flush complete",
                timeout_s=wait_flush_s,
                after_mark=mark,
            )
            if not ok:
                rt.wait_for_stdout("[tap] wrote", timeout_s=0.5, after_mark=mark)
            # Disk settle: WAVs can appear a beat after the stdout line on Windows.
            if trial_dir is not None:
                root = Path(trial_dir)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    if any(
                        (root / name).is_file() and (root / name).stat().st_size > 1024
                        for name in ("raw_8ch.wav", "mapped_6ch.wav", "dry_ref.wav", "wet_out.wav")
                    ):
                        break
                    time.sleep(0.05)
        try:
            self.send_cmd("TAP disarm")
        except RuntimeError:
            pass

    def append_profile_to_trial_log(self, trial_log: Path) -> None:
        """Copy recent session stdout into the trial pipeline.log for forensics."""
        with self._lock:
            rt = self._rt
        if rt is None:
            return
        try:
            trial_log.parent.mkdir(parents=True, exist_ok=True)
            with trial_log.open("a", encoding="utf-8", errors="replace") as f:
                f.write("\n--- session stdout tail ---\n")
                for line in rt.recent_lines(80):
                    f.write(line if line.endswith("\n") else line + "\n")
        except OSError:
            pass

    def refresh_status_from_profile(self) -> None:
        with self._lock:
            rt = self._rt
        if rt is None or rt.poll() is not None:
            if self._state not in (PipelineState.STOPPED, PipelineState.FAULT):
                self._set_state(PipelineState.FAULT, "Pipeline process exited")
            return
        snap = _parse_latest_profile(rt.drain_output())
        if snap is None:
            return
        snap = self._annotate_underrun_delta(snap)
        self._last_profile = snap
        if self._state == PipelineState.READY:
            self._status = (
                f"Ready — mode={snap.mode} model_out={snap.model_out_db:.1f} dBFS "
                f"underrun=+{snap.underrun_delta} (Σ{snap.underrun_frames})"
            )
            self._notify()

    def runtime_dict(self) -> dict[str, Any]:
        with self._lock:
            if self._rt is None:
                return {}
            return self._rt.launch.as_dict()
