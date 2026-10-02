"""Stage E — multi-out ONNX cost envelope vs 8 ms hop / 16 ms block budget."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
from scripts.inbubble_mixer.stage_b_separation_probe import SoftMaskBubbleSplit


class MultiOutWrapper(nn.Module):
    """Exportable wrapper: promoted bubble + K softmask streams (sum ≈ b_hat)."""

    def __init__(self, bubble_model: nn.Module, splitter: SoftMaskBubbleSplit):
        super().__init__()
        self.bubble = bubble_model
        self.splitter = splitter

    def forward(self, mixture: torch.Tensor):
        # mixture: [B, C, T]
        out = self.bubble({"mixture": mixture})
        b = out["output"]
        if b.ndim == 3:
            b_mono = b[:, 0, :]
        else:
            b_mono = b
        s = self.splitter(b_mono)
        return b_mono, s[:, 0, :], s[:, 1, :]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=str, default=DEFAULT_RUN)
    p.add_argument("--split-ckpt", type=str, default="runs/inbubble_mixer_stage_b/softmask_split.pt")
    p.add_argument("--out-dir", type=str, default="runs/inbubble_mixer_stage_e")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--chunk", type=int, default=192)
    p.add_argument("--pad", type=int, default=96)
    p.add_argument("--channels", type=int, default=6)
    p.add_argument("--hops-per-io", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def export_splitter_onnx(splitter: SoftMaskBubbleSplit, path: Path, t_len: int = 384) -> None:
    splitter.eval()

    class SplitOnly(nn.Module):
        def __init__(self, s):
            super().__init__()
            self.s = s

        def forward(self, b_hat: torch.Tensor):
            streams = self.s(b_hat)
            return streams[:, 0, :], streams[:, 1, :]

    model = SplitOnly(splitter).eval()
    dummy = torch.zeros(1, t_len, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["b_hat"],
        output_names=["s0", "s1"],
        opset_version=17,
        dynamo=False,
    )


def bench_ort_splitter(onnx_path: Path, t_len: int, warmup: int, runs: int, threads: int) -> dict:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    x = np.zeros((1, t_len), dtype=np.float32)
    for _ in range(warmup):
        sess.run(None, {"b_hat": x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"b_hat": x})
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {"mean_ms": float(arr.mean()), "std_ms": float(arr.std()), "p95_ms": float(np.percentile(arr, 95))}


def bench_torch_cpu(wrapper: nn.Module, chunk: int, channels: int, warmup: int, runs: int) -> dict:
    wrapper.eval()
    x = torch.zeros(1, channels, chunk)
    with torch.no_grad():
        try:
            wrapper(x)
        except Exception:
            x = torch.zeros(1, channels, 96 + chunk + 96)
            wrapper(x)
        for _ in range(warmup):
            wrapper(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            wrapper(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p95_ms": float(np.percentile(arr, 95)),
        "input_shape": list(x.shape),
    }


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    torch.set_num_threads(args.threads)

    pl_model, device = load_promoted_model(args.run_dir, device="cpu")
    bubble = pl_model
    splitter = SoftMaskBubbleSplit(k=2)
    ckpt_path = Path(args.split_ckpt)
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        splitter.load_state_dict(ckpt["state_dict"])
    splitter.eval()
    bubble.eval()

    wrapper = MultiOutWrapper(bubble, splitter).eval()

    class BubbleOnly(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, mixture):
            out = self.m({"mixture": mixture})
            y = out["output"]
            return y[:, 0] if y.ndim == 3 else y

    bubble_only = BubbleOnly(bubble).eval()
    base = bench_torch_cpu(bubble_only, args.chunk, args.channels, args.warmup, args.runs)
    multi = bench_torch_cpu(wrapper, args.chunk, args.channels, args.warmup, args.runs)

    splitter_onnx = out_dir / "softmask_split.onnx"
    ort_split = None
    export_error = None
    try:
        export_splitter_onnx(splitter, splitter_onnx, t_len=max(384, args.chunk * 2))
        ort_split = bench_ort_splitter(
            splitter_onnx, t_len=max(384, args.chunk * 2), warmup=args.warmup, runs=args.runs, threads=args.threads
        )
    except Exception as e:
        export_error = repr(e)

    # Always time splitter alone in torch (ONNX STFT export may fail).
    t_len = max(384, args.chunk * 2)
    x_b = torch.zeros(1, t_len)
    with torch.no_grad():
        for _ in range(args.warmup):
            splitter(x_b)
        split_times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            splitter(x_b)
            split_times.append((time.perf_counter() - t0) * 1000.0)
    torch_split = {
        "mean_ms": float(np.mean(split_times)),
        "std_ms": float(np.std(split_times)),
        "p95_ms": float(np.percentile(split_times, 95)),
    }

    hop_ms = 1000.0 * args.chunk / DEFAULT_SR
    block_ms = hop_ms * args.hops_per_io
    infer_ms = multi["mean_ms"]
    hops_fit = block_ms / infer_ms if infer_ms > 0 else 0.0
    split_ms = (ort_split or torch_split)["mean_ms"]
    combo_est = base["mean_ms"] + split_ms

    summary = {
        "stage": "E",
        "hop_ms": hop_ms,
        "block_ms": block_ms,
        "hops_per_io": args.hops_per_io,
        "torch_bubble_only_ms": base,
        "torch_multi_out_ms": multi,
        "torch_splitter_only_ms": torch_split,
        "splitter_onnx_path": str(splitter_onnx) if splitter_onnx.is_file() else None,
        "ort_splitter_ms": ort_split,
        "export_error": export_error,
        "overhead_ratio_torch": (multi["mean_ms"] / base["mean_ms"]) if base["mean_ms"] else None,
        "bubble_plus_splitter_est_ms": combo_est,
        "sustainable_hops_estimate": hops_fit,
        "recommendation": (
            "multi-out exceeds block budget on this host proxy; keep K=0 mixer pass-through "
            "or lighter mask-on-b_hat until Pi 5 re-measure"
            if hops_fit < args.hops_per_io
            else "multi-out fits host proxy budget"
        ),
        "gate": {
            "name": "pi5_cost_envelope",
            "passed": bool(infer_ms > 0 and hops_fit >= args.hops_per_io),
            "note": (
                f"Pass if multi-out mean infer ({infer_ms:.2f} ms) fits "
                f"{args.hops_per_io} hops in {block_ms:.1f} ms block. "
                "Host CPU proxy — re-measure on Pi 5 before deploy."
            ),
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    # Stage E completing with FAIL is an investigation success (cost gate denied product path).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

