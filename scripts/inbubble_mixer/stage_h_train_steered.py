"""Stage H — smoke / short train of steered head into directional run dir.

Writes ONLY under runs/optim_pretrain_sonitude_dist_count_directional_1_5m.
Init from promote best.pt; backbone frozen; steered_head trainable.

Full overnight::

    python -m src.train_pt \\
      --config real_experiments/optim_pretrain_sonitude_dist_count_directional_1_5m.json \\
      --run_dir runs/optim_pretrain_sonitude_dist_count_directional_1_5m \\
      --no_wandb
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.utils as utils
from scripts.inbubble_mixer.common import ensure_dir, write_json


DEFAULT_CONFIG = "real_experiments/optim_pretrain_sonitude_dist_count_directional_1_5m.json"
DEFAULT_RUN = "runs/optim_pretrain_sonitude_dist_count_directional_1_5m"
PROMOTE_RUN = "runs/optim_pretrain_sonitude_dist_count_1_5m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    p.add_argument("--run-dir", type=str, default=DEFAULT_RUN)
    p.add_argument("--steps", type=int, default=40, help="Smoke train steps (0 = config only + init dump).")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-train-samples", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = ensure_dir(args.run_dir)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    # Guard: never write into the promote path.
    promote = Path(PROMOTE_RUN).resolve()
    if run_dir.resolve() == promote:
        raise SystemExit("Refusing to write directional train into the promote run_dir.")

    with open(args.config, "rb") as f:
        params = json.load(f)

    # Copy config into run dir (train_pt does this too).
    shutil.copy2(args.config, run_dir / "config.json")

    train_args = dict(params["train_data_args"])
    # Cap samples for smoke.
    capped = []
    remaining = args.max_train_samples
    for d in train_args["dataset_dirs"]:
        take = min(int(d["max_samples"]), remaining)
        if take <= 0:
            break
        capped.append({**d, "max_samples": take})
        remaining -= take
    train_args["dataset_dirs"] = capped

    data_train = utils.import_attr(params["train_dataset"])(**train_args, split="train")
    loader = DataLoader(
        data_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pl_args = dict(params["pl_module_args"])
    pl_args["use_dp"] = False
    pl = utils.import_attr(params["pl_module"])(**pl_args)
    pl.model.to(device)
    pl.train()

    trainable = sum(p.numel() for p in pl.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in pl.model.parameters())
    print(f"Device={device} trainable={trainable}/{total}")

    losses = []
    audio_losses = []
    steered_losses = []
    n_steps = 0
    if args.steps > 0:
        it = iter(loader)
        while n_steps < args.steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            # Move batch to device
            inputs, targets = batch
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            targets = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in targets.items()}
            loss, _B = pl.training_step((inputs, targets), n_steps)
            pl.reset_grad()
            loss.backward()
            pl.backprop()
            losses.append(float(loss.detach().cpu()))
            n_steps += 1
            if n_steps % 10 == 0 or n_steps == 1:
                print(f"step {n_steps}/{args.steps} loss={losses[-1]:.4f}")

    # Dump checkpoints into directional run only.
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"
    pl.dump_state(str(last_path))
    if not best_path.exists() or args.steps > 0:
        shutil.copy2(last_path, best_path)

    mean_loss = float(sum(losses) / len(losses)) if losses else None
    summary = {
        "stage": "H",
        "run_dir": str(run_dir),
        "init_ckpt": pl_args.get("init_ckpt"),
        "promote_untouched": str(promote),
        "trainable_params": trainable,
        "total_params": total,
        "steps": n_steps,
        "mean_loss": mean_loss,
        "gate": {
            "name": "steered_head_train_dir_ready",
            "passed": bool(best_path.is_file() and last_path.is_file() and trainable > 0),
            "note": (
                "Pass if directional run has best/last checkpoints and steered_head params train. "
                "Full SI-SDR gate vs bubble baseline requires overnight train_pt."
            ),
        },
    }
    write_json(run_dir / "stage_h_smoke_summary.json", summary)
    stage_h_dir = ensure_dir("runs/inbubble_mixer_stage_h")
    write_json(stage_h_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
