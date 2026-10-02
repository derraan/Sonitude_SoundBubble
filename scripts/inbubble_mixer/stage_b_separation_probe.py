"""Stage B — mix-consistent K<=2 soft-mask separation probe on promoted b_hat."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from edge.realtime.inbubble_mixer import mix_inbubble, si_sdr_np
from scripts.inbubble_mixer.common import (
    DEFAULT_DATASET,
    DEFAULT_DIS_THRESHOLD,
    DEFAULT_RUN,
    DEFAULT_SR,
    ensure_dir,
    infer_bubble,
    iter_samples,
    load_promoted_model,
    write_json,
)


class SoftMaskBubbleSplit(nn.Module):
    """Split mono b_hat into K streams with softmax masks (sum_k s_k == b_hat)."""

    def __init__(self, n_fft: int = 512, hop: int = 128, k: int = 2, hidden: int = 64):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.k = k
        self.window = torch.hann_window(n_fft)
        n_freq = n_fft // 2 + 1
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, k, kernel_size=1),
        )
        self.register_buffer("freq_bias", torch.zeros(1, k, n_freq, 1))

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window.to(x.device),
            return_complex=True,
            center=True,
        )

    def istft(self, X: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window.to(X.device),
            length=length,
            center=True,
        )

    def forward(self, b_hat: torch.Tensor) -> torch.Tensor:
        # returns s_hats [B, K, T]
        B, T = b_hat.shape
        X = self.stft(b_hat)
        feats = torch.stack([X.real, X.imag], dim=1)  # [B,2,F,TT]
        logits = self.net(feats) + self.freq_bias
        masks = torch.softmax(logits, dim=1)  # [B,K,F,TT]
        streams = []
        for k in range(self.k):
            Yk = masks[:, k] * X
            streams.append(self.istft(Yk, length=T))
        return torch.stack(streams, dim=1)


def sisdr_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # est/ref: [B, T]
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    dot = (est * ref).sum(dim=-1, keepdim=True)
    s_target = dot * ref / (ref.pow(2).sum(dim=-1, keepdim=True) + eps)
    e_noise = est - s_target
    ratio = (s_target.pow(2).sum(dim=-1) + eps) / (e_noise.pow(2).sum(dim=-1) + eps)
    return -10.0 * torch.log10(ratio + eps)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET)
    p.add_argument("--run-dir", type=str, default=DEFAULT_RUN)
    p.add_argument("--max-train", type=int, default=48)
    p.add_argument("--max-val", type=int, default=16)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dis-threshold", type=float, default=DEFAULT_DIS_THRESHOLD)
    p.add_argument("--sr", type=int, default=DEFAULT_SR)
    p.add_argument("--out-dir", type=str, default="runs/inbubble_mixer_stage_b")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def collect(args, model, device, split: str, max_samples: int):
    bundles = []
    for sample in iter_samples(
        args.dataset_root,
        split=split,
        max_samples=max_samples,
        require_k=2,
        sr=args.sr,
        dis_threshold=args.dis_threshold,
        k_max=2,
    ):
        b_hat = infer_bubble(model, sample.mixture, device)
        T = min(b_hat.shape[0], sample.inside_stems[0].shape[0], sample.oracle_bubble.shape[0])
        bundles.append(
            {
                "b_hat": b_hat[:T],
                "oracle_b": sample.oracle_bubble[:T],
                "s0": sample.inside_stems[0][:T],
                "s1": sample.inside_stems[1][:T],
                "id": sample.sample_id,
            }
        )
    return bundles


def batchify(items, device, bubble_key: str = "oracle_b"):
    T = min(x[bubble_key].shape[0] for x in items)
    b = torch.stack([torch.from_numpy(x[bubble_key][:T]) for x in items], dim=0).to(device)
    s0 = torch.stack([torch.from_numpy(x["s0"][:T]) for x in items], dim=0).to(device)
    s1 = torch.stack([torch.from_numpy(x["s1"][:T]) for x in items], dim=0).to(device)
    return b, s0, s1


def eval_rows(split_model, items, device, bubble_key: str):
    rows = []
    with torch.no_grad():
        for item in items:
            b_np = item[bubble_key]
            b = torch.from_numpy(b_np).unsqueeze(0).to(device)
            s_hat = split_model(b)[0].cpu().numpy()
            p0 = si_sdr_np(s_hat[0], item["s0"]) + si_sdr_np(s_hat[1], item["s1"])
            p1 = si_sdr_np(s_hat[0], item["s1"]) + si_sdr_np(s_hat[1], item["s0"])
            if p1 > p0:
                s_hat = s_hat[::-1].copy()
            cons_rms = float(np.sqrt(np.mean((s_hat.sum(0) - b_np) ** 2) + 1e-12))
            y_boost = mix_inbubble(b_np, s_hat, gains=(2.0, 1.0)).y
            y_ideal = mix_inbubble(b_np, [item["s0"], item["s1"]], gains=(2.0, 1.0)).y
            y_cut = mix_inbubble(b_np, s_hat, gains=(0.25, 1.0)).y
            y_cut_ideal = mix_inbubble(b_np, [item["s0"], item["s1"]], gains=(0.25, 1.0)).y
            rows.append(
                {
                    "sample_id": item["id"],
                    "bubble": bubble_key,
                    "s0_sisdr": si_sdr_np(s_hat[0], item["s0"]),
                    "s1_sisdr": si_sdr_np(s_hat[1], item["s1"]),
                    "consistency_rms": cons_rms,
                    "boost_ctrl_sisdr": si_sdr_np(y_boost, y_ideal),
                    "cut_ctrl_sisdr": si_sdr_np(y_cut, y_cut_ideal),
                    "identity_max_abs": float(
                        np.max(np.abs(mix_inbubble(b_np, s_hat, (1, 1)).y - b_np))
                    ),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = ensure_dir(args.out_dir)

    promoted, device = load_promoted_model(args.run_dir)
    train_data = collect(args, promoted, device, "train", args.max_train)
    val_data = collect(args, promoted, device, "val", args.max_val)
    if len(train_data) < 4:
        val_data = train_data[-max(4, len(train_data) // 4) :]
        train_data = train_data[: max(4, len(train_data) - len(val_data))]

    split = SoftMaskBubbleSplit(k=2).to(device)
    opt = torch.optim.Adam(split.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        split.train()
        np.random.shuffle(train_data)
        losses = []
        for i in range(0, len(train_data), 4):
            chunk = train_data[i : i + 4]
            if not chunk:
                continue
            # Train mix-consistent split on oracle bubble (cleaner supervision), keep promoted b_hat frozen.
            b, s0, s1 = batchify(chunk, device, bubble_key="oracle_b")
            s_hat = split(b)
            loss_p0 = sisdr_loss(s_hat[:, 0], s0).mean() + sisdr_loss(s_hat[:, 1], s1).mean()
            loss_p1 = sisdr_loss(s_hat[:, 0], s1).mean() + sisdr_loss(s_hat[:, 1], s0).mean()
            sep = torch.minimum(loss_p0, loss_p1)
            cons = F.mse_loss(s_hat.sum(dim=1), b)
            loss = sep + 10.0 * cons
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None})
        print(f"epoch {epoch}: train_loss={history[-1]['train_loss']}")

    split.eval()
    rows_oracle = eval_rows(split, val_data, device, "oracle_b")
    rows_model = eval_rows(split, val_data, device, "b_hat")

    def agg(rows):
        if not rows:
            return {"mean_stream_sisdr": float("nan"), "mean_boost": float("nan"), "mean_cut": float("nan"), "mean_cons": float("nan")}
        return {
            "mean_stream_sisdr": float(np.nanmean([0.5 * (r["s0_sisdr"] + r["s1_sisdr"]) for r in rows])),
            "mean_boost": float(np.nanmean([r["boost_ctrl_sisdr"] for r in rows])),
            "mean_cut": float(np.nanmean([r["cut_ctrl_sisdr"] for r in rows])),
            "mean_cons": float(np.nanmean([r["consistency_rms"] for r in rows])),
        }

    ao, am = agg(rows_oracle), agg(rows_model)
    summary = {
        "stage": "B",
        "n_train": len(train_data),
        "n_val": len(val_data),
        "history": history,
        "oracle_bubble_eval": ao,
        "model_bhat_eval": am,
        "gate": {
            "name": "non_oracle_streams_stable_control",
            "passed": bool(
                rows_oracle
                and ao["mean_boost"] > 3.0
                and ao["mean_cut"] > 3.0
                and ao["mean_cons"] < 0.05
            ),
            "model_transfer_ok": bool(am["mean_boost"] > 0.0 and am["mean_cut"] > 0.0),
            "note": (
                "Pass if K=2 soft-mask probe on oracle bubble yields boost/cut control SI-SDR > 3 dB "
                "with low mix residual. Model-b_hat transfer reported separately."
            ),
        },
    }

    ckpt = {"state_dict": split.state_dict(), "n_fft": 512, "hop": 128, "k": 2}
    torch.save(ckpt, out_dir / "softmask_split.pt")
    with (out_dir / "eval_rows.csv").open("w", newline="", encoding="utf-8") as f:
        all_rows = rows_oracle + rows_model
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["gate"]["passed"] else 2


if __name__ == "__main__":
    import json

    raise SystemExit(main())
