"""Stage J — cost gate for directional steered aux vs 8 ms hop / 16 ms block.

Compares:
  - promote bubble-only (default ship path)
  - directional model with steered_output (one aux)
  - Stage E lesson: full K-stream softmask decode (reference upper bound)

Ship rule: default remain promote b_hat + g=1 unless steered-only fits budget.
Promote directional → default ONLY on explicit user request (never rename-in-place).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.inbubble_mixer.common import (
    DEFAULT_RUN,
    DEFAULT_SR,
    ensure_dir,
    load_promoted_model,
    write_json,
)
from src.utils import load_torch_pretrained


DIRECTIONAL_RUN = "runs/optim_pretrain_sonitude_dist_count_directional_1_5m"
HOP_MS = 8.0
BLOCK_MS = 16.0  # 2 hops


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--promote-run", type=str, default=DEFAULT_RUN)
    p.add_argument("--directional-run", type=str, default=DIRECTIONAL_RUN)
    p.add_argument("--out-dir", type=str, default="runs/inbubble_mixer_stage_j")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=80)
    p.add_argument("--chunk", type=int, default=192)
    p.add_argument("--pad", type=int, default=96)
    p.add_argument("--channels", type=int, default=6)
    p.add_argument("--hops-per-io", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--pi5-host",
        type=str,
        default="",
        help="Optional SSH host for on-device ORT (e.g. rpi5-soundbubble). Empty = probe defaults.",
    )
    return p.parse_args()


def probe_pi5_ort(host: str) -> dict:
    """Attempt reachability for Pi 5 ORT bench; never blocks ship on failure."""
    import shutil
    import socket
    import subprocess

    candidates = [h for h in [host, "rpi5-soundbubble", "raspberrypi.local"] if h]
    checked: list[dict] = []
    for cand in candidates:
        entry: dict = {"host": cand, "reachable": False, "ort_available": False}
        try:
            socket.getaddrinfo(cand, 22, type=socket.SOCK_STREAM)
            entry["dns_ok"] = True
        except OSError as exc:
            entry["dns_ok"] = False
            entry["dns_error"] = str(exc)
            checked.append(entry)
            continue
        if shutil.which("ssh") is None:
            entry["ssh_error"] = "ssh binary not found"
            checked.append(entry)
            continue
        try:
            proc = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=3",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    cand,
                    "python3 -c \"import onnxruntime as ort; print(ort.__version__); print(','.join(ort.get_available_providers()))\"",
                ],
                capture_output=True,
                text=True,
                timeout=12,
            )
            entry["ssh_rc"] = proc.returncode
            if proc.returncode == 0:
                lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
                entry["reachable"] = True
                entry["ort_available"] = True
                if lines:
                    entry["ort_version"] = lines[0]
                if len(lines) > 1:
                    entry["ort_providers"] = lines[1].split(",")
                checked.append(entry)
                return {
                    "pi5_available": True,
                    "pi5_ort_available": True,
                    "pi5_host": cand,
                    "attempts": checked,
                    "note": "Pi 5 reachable with onnxruntime; on-device ORT timing still requires exported ONNX.",
                }
            entry["ssh_stderr"] = (proc.stderr or "")[:400]
        except Exception as exc:
            entry["ssh_error"] = str(exc)
        checked.append(entry)
    return {
        "pi5_available": False,
        "pi5_ort_available": False,
        "pi5_host": None,
        "attempts": checked,
        "note": (
            "Pi 5 / ORT device unavailable on this machine "
            "(no Tailscale MagicDNS / SSH reachability). "
            "Host torch proxy timing used instead; keep laptop-only directional zoom."
        ),
    }


def _bench_torch(fn, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n": int(len(arr)),
    }


@torch.no_grad()
def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    torch.set_num_threads(args.threads)

    t_len = args.chunk * args.hops_per_io
    # Input includes lookahead pad like streaming block.
    mix = torch.zeros(1, args.channels, args.pad + t_len, dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mix = mix.to(device)

    promote_model, _ = load_promoted_model(args.promote_run, device=device)

    def run_promote():
        out = promote_model({"mixture": mix})
        return out["output"]

    promote_stats = _bench_torch(run_promote, args.warmup, args.runs)

    dir_stats = None
    dir_loaded = False
    dir_path = Path(args.directional_run)
    if (dir_path / "checkpoints" / "best.pt").is_file():
        try:
            pl = load_torch_pretrained(str(dir_path), map_location=device)
            dir_model = pl.model.eval().to(device)
            dir_loaded = True
            theta0 = torch.zeros(1, device=device)

            def run_steered():
                out = dir_model({"mixture": mix, "theta0_rad": theta0})
                # Force both bubble + steered aux materialization.
                b = out["output"]
                z = out.get("steered_output")
                return b, z

            dir_stats = _bench_torch(run_steered, args.warmup, args.runs)
        except Exception as exc:
            dir_stats = {"error": str(exc)}

    budget_ms = BLOCK_MS
    promote_ok = promote_stats["p95_ms"] < budget_ms
    steered_ok = bool(
        dir_stats
        and "p95_ms" in dir_stats
        and dir_stats["p95_ms"] < budget_ms
    )

    # Stage E previously failed multi-out; document ship policy.
    # Gate (plan): sustainable under budget OR document laptop-only directional zoom.
    laptop_only = not steered_ok
    ship = {
        "default_listen": "promote b_hat (g_k=1 / alpha=0)",
        "steered_enabled_by_default": False,
        "steered_affordable_on_this_host": steered_ok,
        "mode": "realtime_candidate" if steered_ok else "laptop_only_directional_zoom",
        "recommendation": (
            "Enable steered/mixer only if hop budget holds on Pi 5 proxy; "
            "otherwise keep promote b_hat+g=1 (laptop-only directional zoom)."
            if not steered_ok
            else "Steered aux p95 under block budget on this host — still requires Pi 5 confirmation before ship."
        ),
        "promote_directional_requires_explicit_user_request": True,
        "note_timing": (
            "Torch full-block timing on this host is a proxy, not Pi 5 ORT. "
            "Stage E already showed multi-out wet decode exceeds budget; "
            "prefer one steered aux only after on-device re-bench."
        ),
    }

    pi5 = probe_pi5_ort(args.pi5_host)
    if not pi5.get("pi5_available"):
        ship["mode"] = "laptop_only_directional_zoom"
        ship["pi5_ort_bench"] = "skipped_device_unavailable"
        ship["recommendation"] = (
            "Pi 5 ORT unavailable — keep promote b_hat+g=1 default; "
            "steered remains laptop-only directional zoom until on-device re-bench."
        )
        laptop_only = True
        steered_ok = False

    summary = {
        "stage": "J",
        "host_device": device,
        "hop_ms": HOP_MS,
        "block_ms": BLOCK_MS,
        "hops_per_io": args.hops_per_io,
        "chunk": args.chunk,
        "promote_run": args.promote_run,
        "directional_run": args.directional_run,
        "directional_loaded": dir_loaded,
        "promote_bubble_ms": promote_stats,
        "directional_steered_aux_ms": dir_stats,
        "promote_within_budget": promote_ok,
        "steered_within_budget": steered_ok,
        "pi5": pi5,
        "timing_source": "pi5_ort" if pi5.get("pi5_ort_available") else "host_torch_proxy",
        "ship": ship,
        "gate": {
            "name": "cost_gate_default_promote",
            "passed": bool(
                dir_loaded
                and ship["steered_enabled_by_default"] is False
                and (steered_ok or laptop_only)
            ),
            "steered_ship_default": False,
            "laptop_only_directional_zoom": laptop_only,
            "note": (
                "Pass if default remains promote b_hat and either steered fits the "
                "16 ms block budget or we document laptop-only directional zoom."
            ),
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
