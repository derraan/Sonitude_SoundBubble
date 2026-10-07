from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort

from runtime_contract import load_contract


DEFAULT_WARMUP = 20
DEFAULT_RUNS = 100
DEFAULT_THREADS = "4"
DEFAULT_BATCH = 1
DEFAULT_CHANNELS = 6
DEFAULT_FRAME_LEN = 288
DEFAULT_BUDGET_MS = 6.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Sound_Bubble ONNX inference latency.")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model.")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup forward passes.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Timed benchmark runs.")
    parser.add_argument(
        "--num-threads",
        type=str,
        default=DEFAULT_THREADS,
        help="ORT intra-op thread count, or a comma-separated sweep such as 1,2,3.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Batch size for dummy input.")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS, help="Channel count for mixture input.")
    parser.add_argument("--frame-len", type=int, default=DEFAULT_FRAME_LEN, help="Frame length for mixture input.")
    parser.add_argument("--contract-path", type=str, default=None, help="Runtime contract JSON produced during export.")
    parser.add_argument(
        "--budget-ms",
        type=float,
        default=DEFAULT_BUDGET_MS,
        help="Per-hop worst-case budget; runs above it are counted as over-budget.",
    )
    return parser


def _shape_from_ort(meta_shape: list[int | str | None], batch: int, channels: int, frame_len: int) -> list[int]:
    resolved: list[int] = []
    for idx, dim in enumerate(meta_shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        else:
            if idx == 0:
                resolved.append(batch)
            else:
                resolved.append(1)
    return resolved


def _parse_threads(text: str) -> list[int]:
    threads = [int(tok) for tok in text.split(",") if tok.strip()]
    if not threads or min(threads) < 1:
        raise SystemExit("--num-threads must be positive integers, e.g. 2 or 1,2,3")
    return threads


def _benchmark(args: argparse.Namespace, num_threads: int) -> list[float]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(num_threads)
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(args.model, sess_options=options, providers=["CPUExecutionProvider"])
    input_metas = session.get_inputs()
    output_metas = session.get_outputs()

    if not input_metas:
        raise RuntimeError("ONNX model has no inputs.")

    feeds: dict[str, np.ndarray] = {}
    for input_meta in input_metas:
        if input_meta.name == "mixture":
            shape = [args.batch_size, args.channels, args.frame_len]
        else:
            shape = _shape_from_ort(input_meta.shape, args.batch_size, args.channels, args.frame_len)
        feeds[input_meta.name] = np.zeros(shape, dtype=np.float32)

    state_input_names = sorted([m.name for m in input_metas if m.name.startswith("state_")], key=lambda n: int(n.split("_")[1]))
    state_output_names = sorted(
        [m.name for m in output_metas if m.name.startswith("next_state_")],
        key=lambda n: int(n.split("_")[2]),
    )
    if len(state_input_names) != len(state_output_names):
        raise RuntimeError("State input/output count mismatch.")

    output_names = [m.name for m in output_metas]
    timings_ms: list[float] = []
    for i in range(args.warmup + args.runs):
        t0 = time.perf_counter()
        outputs = session.run(None, feeds)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= args.warmup:
            timings_ms.append(elapsed_ms)
        out_map = dict(zip(output_names, outputs))
        for in_name, out_name in zip(state_input_names, state_output_names):
            feeds[in_name] = out_map[out_name].astype(np.float32, copy=False)
    return timings_ms


def main() -> None:
    args = _build_parser().parse_args()
    if args.contract_path:
        contract = load_contract(args.contract_path)
        args.channels = contract.model_num_ch
        args.frame_len = contract.frame_len
    threads = _parse_threads(args.num_threads)

    print(f"[info] model={args.model}")
    print(f"[info] provider=CPUExecutionProvider, inter_op=1, threads={threads}")
    print(f"[info] warmup={args.warmup}, runs={args.runs}, budget={args.budget_ms:.2f} ms")

    for num_threads in threads:
        t = np.asarray(_benchmark(args, num_threads), dtype=np.float64)
        p50, p95, p99 = np.percentile(t, [50.0, 95.0, 99.0])
        over = int(np.count_nonzero(t > args.budget_ms))
        print(
            f"[result] intra_op={num_threads}  mean={t.mean():.3f} ms  std={t.std():.3f} ms  "
            f"p50={p50:.3f} ms  p95={p95:.3f} ms  p99={p99:.3f} ms  max={t.max():.3f} ms  "
            f"over_budget={over}/{t.size}"
        )


if __name__ == "__main__":
    main()
