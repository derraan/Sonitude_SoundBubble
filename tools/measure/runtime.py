"""Launch selectable ONNX realtime pipelines (cpp | python) with Class B args."""

from __future__ import annotations

import collections
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

# Keep a bounded in-memory tail so chatty --profile/--monitor children cannot
# grow unbounded while still draining the PIPE (avoids OS pipe-buffer deadlock).
_STDOUT_RING_MAX_LINES = 4000
_STDOUT_JOIN_S = 2.0


@dataclass
class Gains:
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0
    mix_dry: float = 0.0
    output_limiter: bool = True
    limiter_release_ms: float = 25.0
    # Linear per-mic gains by device channel (cpp runtime --mic-gains). Empty/all 1.0 = off.
    mic_gains: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_gain_db": self.input_gain_db,
            "output_gain_db": self.output_gain_db,
            "mix_dry": self.mix_dry,
            "output_limiter": self.output_limiter,
            "limiter_release_ms": self.limiter_release_ms,
            "mic_gains": list(self.mic_gains),
        }


@dataclass
class RuntimeLaunch:
    runtime: str  # cpp | python
    pipeline_in_device: int | None = None
    pipeline_out_device: int | None = None
    input_sr: int = 24000
    output_sr: int = 24000
    input_channels: int = 8
    output_channels: int = 2
    channel_map: str = "5,4,3,2,1,0"  # MIC_INDEX_CANON.md §4; omit CLI flag when empty
    gains: Gains = field(default_factory=Gains)
    model_path: str = ""
    contract_path: str = ""
    zoo_dir: str = ""
    bubble_radius_m: float | None = 1.5
    use_zoo: bool = True
    soundbubble_rt_path: str = ""
    python_exe: str = ""
    tap_dir: str = ""
    tap_select: str = "raw,mapped,dry,wet"
    tap_max_seconds: float = 120.0
    tap_max_mb: float = 1024.0
    duplex_mode: str = "off"
    # PortAudio suggestedLatency hint: low | high (Windows Class B global session uses high).
    latency: str = "low"
    # Match Pi deploy (SILENCE_DBFS=-80). C++ default is -50 and zeros channels pre-gain.
    silence_dbfs: float = -80.0
    max_capture_blocks: int | None = None
    target_output_ahead_sec: float | None = None
    max_output_ahead_sec: float | None = None
    output_device_block: int | None = None
    model_ring_ms: float | None = None
    hops_per_io: int | None = None
    intra_op_threads: int | None = None
    inter_op_threads: int | None = None
    # Capture dequeue: adaptive|latest|fifo. Windows global session prefers fifo.
    queue_policy: str | None = None
    profile: bool = True
    monitor: bool = False
    extra_args: list[str] = field(default_factory=list)

    def resolve_binary(self) -> str:
        if self.runtime == "python":
            return self.python_exe or sys.executable
        candidates = []
        if self.soundbubble_rt_path:
            candidates.append(Path(self.soundbubble_rt_path))
        candidates.extend(
            [
                REPO_ROOT / "edge" / "cpp" / "build" / "win" / "soundbubble-rt.exe",
                REPO_ROOT / "edge" / "cpp" / "build" / "win" / "Release" / "soundbubble-rt.exe",
                REPO_ROOT / "edge" / "cpp" / "build-local" / "soundbubble-rt.exe",
                REPO_ROOT / "edge" / "cpp" / "build" / "pi" / "soundbubble-rt",
                Path("/opt/soundbubble/bin/soundbubble-rt"),
            ]
        )
        for p in candidates:
            if p.is_file():
                return str(p)
        raise FileNotFoundError(
            "soundbubble-rt not found; set soundbubble_rt_path or build edge/cpp "
            "(see tools/measure/README.md Windows build)."
        )

    def build_argv(self) -> list[str]:
        g = self.gains
        common: list[str] = [
            "--input-sr",
            str(self.input_sr),
            "--output-sr",
            str(self.output_sr),
            "--input-channels",
            str(self.input_channels),
            "--output-channels",
            str(self.output_channels),
            "--input-gain-db",
            str(g.input_gain_db),
            "--output-gain-db",
            str(g.output_gain_db),
            "--mix-dry",
            str(g.mix_dry),
            "--limiter-release-ms",
            str(g.limiter_release_ms),
            "--duplex-mode",
            self.duplex_mode,
        ]
        lat = (self.latency or "low").strip().lower()
        if lat in ("low", "high"):
            common += ["--latency", lat]
        common += ["--silence-dbfs", str(float(self.silence_dbfs))]
        if self.max_capture_blocks is not None:
            common += ["--max-capture-blocks", str(int(self.max_capture_blocks))]
        if self.target_output_ahead_sec is not None:
            common += ["--target-output-ahead-sec", str(float(self.target_output_ahead_sec))]
        if self.max_output_ahead_sec is not None:
            common += ["--max-output-ahead-sec", str(float(self.max_output_ahead_sec))]
        if self.output_device_block is not None:
            common += ["--output-device-block", str(int(self.output_device_block))]
        if self.model_ring_ms is not None:
            common += ["--model-ring-ms", str(float(self.model_ring_ms))]
        if self.hops_per_io is not None:
            common += ["--hops-per-io", str(int(self.hops_per_io))]
        if self.intra_op_threads is not None:
            common += ["--intra-op-threads", str(int(self.intra_op_threads))]
        if self.inter_op_threads is not None:
            common += ["--inter-op-threads", str(int(self.inter_op_threads))]
        qp = (self.queue_policy or "").strip().lower()
        if qp in ("adaptive", "latest", "fifo"):
            common += ["--queue-policy", qp]
        if self.channel_map.strip():
            common += ["--channel-map", self.channel_map.strip()]
        if not g.output_limiter:
            common.append("--no-output-limiter")
        mic_g = [float(v) for v in (g.mic_gains or [])]
        if any(v <= 0 for v in mic_g):
            raise ValueError(f"mic_gains must be > 0 (linear): {mic_g}")
        if any(abs(v - 1.0) > 1e-9 for v in mic_g):
            if self.runtime == "cpp":
                common += ["--mic-gains", ",".join(f"{v:.6g}" for v in mic_g)]
            else:
                print("[runtime] mic_gains ignored: python runtime has no --mic-gains")
        if self.pipeline_in_device is not None:
            common += ["--input-device", str(self.pipeline_in_device)]
        if self.pipeline_out_device is not None:
            common += ["--output-device", str(self.pipeline_out_device)]
        if self.use_zoo and self.zoo_dir:
            common += ["--zoo-dir", self.zoo_dir]
            if self.bubble_radius_m is not None:
                common += ["--bubble-radius", str(self.bubble_radius_m)]
        else:
            if not self.model_path:
                raise ValueError("model_path required when not using zoo")
            common += ["--model", self.model_path]
            if self.contract_path:
                common += ["--contract-path", self.contract_path]
            if self.bubble_radius_m is not None:
                common += ["--bubble-radius", str(self.bubble_radius_m)]
        # Always advertise tap-select so a later stdin "TAP dir" inherits raw+mapped+dry+wet.
        if (self.tap_select or "").strip():
            common += ["--tap-select", self.tap_select.strip()]
        if self.tap_dir:
            common += [
                "--tap-dir",
                self.tap_dir,
                "--tap-max-seconds",
                str(self.tap_max_seconds),
                "--tap-max-mb",
                str(self.tap_max_mb),
            ]
        elif self.tap_max_seconds or self.tap_max_mb:
            common += [
                "--tap-max-seconds",
                str(self.tap_max_seconds),
                "--tap-max-mb",
                str(self.tap_max_mb),
            ]
        if self.profile:
            common.append("--profile")
        if self.monitor:
            common.append("--monitor")
        common += list(self.extra_args)

        if self.runtime == "cpp":
            return [self.resolve_binary(), *common]
        # python
        return [
            self.resolve_binary(),
            "-m",
            "edge.inference_pipeline",
            *common,
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "binary_or_entry": self.resolve_binary() if self.runtime == "cpp" else "edge.inference_pipeline",
            "model": self.model_path,
            "contract": self.contract_path,
            "zoo_dir": self.zoo_dir if self.use_zoo else "",
            "bubble_radius_m": self.bubble_radius_m,
            "gains": self.gains.to_dict(),
            "pipeline_in_device": self.pipeline_in_device,
            "pipeline_out_device": self.pipeline_out_device,
            "input_sr": self.input_sr,
            "output_sr": self.output_sr,
            "channel_map": self.channel_map,
            "tap_dir": self.tap_dir,
            "tap_select": self.tap_select,
            "tap_caps": {"max_seconds": self.tap_max_seconds, "max_mb": self.tap_max_mb},
        }


class RuntimeProcess:
    """Launch + stop ONNX RT with stdout drained so PIPE cannot deadlock."""

    def __init__(
        self,
        launch: RuntimeLaunch,
        *,
        cwd: str | Path | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        self.launch = launch
        self.cwd = str(cwd or REPO_ROOT)
        self.proc: subprocess.Popen[str] | None = None
        self.argv: list[str] = []
        self.log_path: Path | None = Path(log_path) if log_path else None
        self._reader: threading.Thread | None = None
        self._lines: collections.deque[str] = collections.deque(maxlen=_STDOUT_RING_MAX_LINES)
        self._stdout_seq: int = 0
        self._log_fp: Any = None
        self._reader_done = threading.Event()

    def set_log_path(self, path: str | Path | None) -> None:
        """Set tee path before ``start()`` (e.g. trial_root / pipeline.log)."""
        self.log_path = Path(path) if path else None

    def start(self) -> None:
        self.argv = self.launch.build_argv()
        env = os.environ.copy()
        # Ensure repo root import for python -m edge.inference_pipeline
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        # Prefer the ORT DLL sitting beside soundbubble-rt.exe over an older
        # System32\onnxruntime.dll (API mismatch → crash before taps).
        if self.launch.runtime == "cpp" and self.argv:
            exe_dir = str(Path(self.argv[0]).resolve().parent)
            env["PATH"] = os.pathsep.join([exe_dir, env.get("PATH", "")])
        kw: dict[str, Any] = {
            "cwd": self.cwd,
            "env": env,
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        if os.name == "nt":
            # New process group so CTRL_BREAK can stop the child for tap flush.
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        self.proc = subprocess.Popen(self.argv, **kw)
        self._reader_done.clear()
        self._lines.clear()
        self._stdout_seq = 0
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fp = self.log_path.open("w", encoding="utf-8", errors="replace")
            try:
                header = (
                    f"[launch] pid={self.proc.pid} "
                    f"in={self.launch.pipeline_in_device} out={self.launch.pipeline_out_device} "
                    f"sr={self.launch.input_sr}/{self.launch.output_sr} "
                    f"runtime={self.launch.runtime}\n"
                    f"[launch] argv={' '.join(self.argv)}\n"
                )
                self._log_fp.write(header)
                self._log_fp.flush()
            except OSError:
                pass
        self._reader = threading.Thread(
            target=self._drain_stdout,
            name="runtime-stdout-drain",
            daemon=True,
        )
        self._reader.start()

    def _drain_stdout(self) -> None:
        """Read child stdout until EOF so the OS pipe buffer never fills."""
        try:
            if self.proc is None or self.proc.stdout is None:
                return
            for line in self.proc.stdout:
                self._lines.append(line)
                self._stdout_seq += 1
                if self._log_fp is not None:
                    try:
                        self._log_fp.write(line)
                        self._log_fp.flush()
                    except OSError:
                        pass
        finally:
            if self._log_fp is not None:
                try:
                    self._log_fp.close()
                except OSError:
                    pass
                self._log_fp = None
            self._reader_done.set()

    def drain_output(self) -> str:
        """Return the in-memory stdout ring (tail) as a single string."""
        return "".join(self._lines)

    def recent_lines(self, n: int = 40) -> list[str]:
        """Return the last ``n`` stdout lines (newest last)."""
        if n <= 0:
            return []
        lines = list(self._lines)
        return lines[-n:]

    def stdout_mark(self) -> int:
        """Opaque cursor: number of lines observed so far in the ring lifecycle."""
        # deque maxlen drops oldest; use a monotonic append counter instead.
        return int(getattr(self, "_stdout_seq", 0))

    def send_cmd(self, line: str) -> None:
        """Write one stdin control line (TAP … / STOP). Raises if process is dead."""
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("runtime process is not running")
        if self.proc.stdin is None:
            raise RuntimeError("runtime stdin is not available")
        payload = line if line.endswith("\n") else line + "\n"
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

    def wait_for_stdout(
        self,
        needle: str,
        *,
        timeout_s: float = 8.0,
        poll_s: float = 0.05,
        after_mark: int | None = None,
    ) -> bool:
        """Block until ``needle`` appears in stdout (optionally only after ``after_mark``).

        Without ``after_mark``, a previous take's ``[tap] flush complete`` still sitting
        in the ring would make the next wait return immediately — back-to-back live
        takes then disarmed before the new flush finished writing WAVs.
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        mark = 0 if after_mark is None else int(after_mark)

        def _hit() -> bool:
            if after_mark is None:
                return needle in self.drain_output()
            # Only scan lines appended after the mark.
            seq = int(getattr(self, "_stdout_seq", 0))
            if seq <= mark:
                return False
            # Ring may have dropped old lines; match against the whole current ring
            # only for lines that arrived after mark by tracking recent slice length.
            lines = list(self._lines)
            # Approximate: if ring holds fewer than (seq-mark) lines, all current
            # lines are post-mark; else take the trailing (seq-mark).
            n_new = min(len(lines), max(0, seq - mark))
            if n_new <= 0:
                return False
            return needle in "".join(lines[-n_new:])

        while time.monotonic() < deadline:
            if _hit():
                return True
            if self.poll() is not None:
                return _hit()
            time.sleep(poll_s)
        return _hit()

    def wait(self, timeout: float | None = None) -> int:
        if self.proc is None:
            return 0
        return int(self.proc.wait(timeout=timeout))

    def stop(self, grace_s: float = 3.0) -> int:
        """Prefer stdin STOP, then SIGINT / CTRL_BREAK, so pipelines flush taps."""
        if self.proc is None:
            self._join_reader()
            return 0
        if self.proc.poll() is not None:
            rc = int(self.proc.returncode or 0)
            self._join_reader()
            return rc

        def _signal_stop() -> None:
            assert self.proc is not None
            try:
                if os.name == "nt":
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                else:
                    self.proc.send_signal(signal.SIGINT)
            except (OSError, ValueError):
                self.proc.terminate()

        used_stdin = False
        try:
            self.send_cmd("STOP")
            used_stdin = True
        except (OSError, RuntimeError, ValueError):
            _signal_stop()

        try:
            rc = int(self.proc.wait(timeout=grace_s))
        except subprocess.TimeoutExpired:
            if used_stdin:
                _signal_stop()
                try:
                    rc = int(self.proc.wait(timeout=grace_s))
                except subprocess.TimeoutExpired:
                    self.proc.terminate()
                    try:
                        rc = int(self.proc.wait(timeout=grace_s))
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        rc = int(self.proc.wait(timeout=5.0))
            else:
                self.proc.terminate()
                try:
                    rc = int(self.proc.wait(timeout=grace_s))
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    rc = int(self.proc.wait(timeout=5.0))
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except OSError:
            pass
        self._join_reader()
        return rc

    def _join_reader(self) -> None:
        if self._reader is not None and self._reader.is_alive():
            self._reader.join(timeout=_STDOUT_JOIN_S)
        self._reader = None
        # Ensure log handle closed if reader exited early / never started.
        if self._log_fp is not None:
            try:
                self._log_fp.close()
            except OSError:
                pass
            self._log_fp = None

    def poll(self) -> int | None:
        if self.proc is None:
            return None
        return self.proc.poll()


def run_offline_wav(
    launch: RuntimeLaunch,
    *,
    input_wav: str | Path,
    output_wav: str | Path,
    extra: Sequence[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """One-shot ``--input-wav`` / ``--output-wav`` (no live devices)."""
    from dataclasses import replace

    offline = replace(launch, pipeline_in_device=None, pipeline_out_device=None)
    argv = offline.build_argv()
    # Strip device flags if present
    cleaned: list[str] = []
    skip = 0
    for i, a in enumerate(argv):
        if skip:
            skip -= 1
            continue
        if a in ("--input-device", "--output-device"):
            skip = 1
            continue
        cleaned.append(a)
    cleaned += ["--input-wav", str(input_wav), "--output-wav", str(output_wav)]
    if extra:
        cleaned += list(extra)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return subprocess.run(
        cleaned,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def sleep_ms(ms: float) -> None:
    time.sleep(max(0.0, ms) / 1000.0)
