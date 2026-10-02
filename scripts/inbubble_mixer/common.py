"""Shared helpers for in-bubble mixer investigation stages."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from src.utils import load_torch_pretrained, read_audio_file


DEFAULT_DATASET = r"C:\Users\darre\Sound_Bubble\datasets\syn_libri_1_5m_rt60"
DEFAULT_RUN = "runs/optim_pretrain_sonitude_dist_count_1_5m"
DEFAULT_SR = 24000
DEFAULT_DIS_THRESHOLD = 1.5
K_MAX = 2


@dataclass
class SampleBundle:
    sample_id: str
    mixture: np.ndarray  # [C, T]
    inside_stems: list[np.ndarray]  # each [T] at mic00
    inside_meta: list[dict]
    outside_stems: list[np.ndarray]
    oracle_bubble: np.ndarray  # sum of inside stems
    metadata: dict


def list_stem_samples(dataset_root: str | Path, split: str = "train") -> list[Path]:
    root = Path(dataset_root) / split
    if not root.is_dir():
        raise FileNotFoundError(root)
    out: list[Path] = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "mic00_voice00.wav").is_file() and (d / "mixture.wav").is_file():
            out.append(d)
    return out


def load_sample(
    sample_dir: Path,
    *,
    sr: int = DEFAULT_SR,
    dis_threshold: float = DEFAULT_DIS_THRESHOLD,
    k_max: int = K_MAX,
) -> SampleBundle:
    meta = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    mixture = read_audio_file(str(sample_dir / "mixture.wav"), sr)
    if mixture.ndim == 1:
        mixture = mixture[None, :]

    voices = sorted([k for k in meta.keys() if k.startswith("voice")])
    inside: list[tuple[float, str, dict, np.ndarray]] = []
    outside: list[np.ndarray] = []
    for v in voices:
        info = meta[v]
        d = float(info["dis"])
        stem_path = sample_dir / f"mic00_{v}.wav"
        if not stem_path.is_file():
            continue
        stem = np.asarray(read_audio_file(str(stem_path), sr), dtype=np.float32).reshape(-1)
        if d <= dis_threshold:
            inside.append((d, v, info, stem))
        else:
            outside.append(stem)

    # Prefer closest-first ordering for stable K_max truncation (matches count slots loosely).
    inside.sort(key=lambda x: x[0])
    if len(inside) > k_max:
        # Keep K_max closest; remainder still belongs in oracle bubble sum.
        kept = inside[:k_max]
        extra = inside[k_max:]
        inside_stems = [x[3] for x in kept]
        inside_meta = [{**x[2], "voice": x[1]} for x in kept]
        oracle = np.zeros_like(inside_stems[0] if inside_stems else mixture[0], dtype=np.float32)
        for x in inside:
            oracle = oracle + x[3]
        # Fold extras into residual of oracle vs kept (still in bubble).
        return SampleBundle(
            sample_id=sample_dir.name,
            mixture=np.asarray(mixture, dtype=np.float32),
            inside_stems=inside_stems,
            inside_meta=inside_meta,
            outside_stems=outside,
            oracle_bubble=oracle.astype(np.float32),
            metadata=meta,
        )

    inside_stems = [x[3] for x in inside]
    inside_meta = [{**x[2], "voice": x[1]} for x in inside]
    if inside_stems:
        oracle = np.sum(np.stack(inside_stems, axis=0), axis=0).astype(np.float32)
    else:
        oracle = np.zeros(mixture.shape[-1], dtype=np.float32)

    return SampleBundle(
        sample_id=sample_dir.name,
        mixture=np.asarray(mixture, dtype=np.float32),
        inside_stems=inside_stems,
        inside_meta=inside_meta,
        outside_stems=outside,
        oracle_bubble=oracle,
        metadata=meta,
    )


def iter_samples(
    dataset_root: str | Path,
    *,
    split: str = "train",
    max_samples: int | None = None,
    require_k: int | None = 2,
    **kwargs,
) -> Iterator[SampleBundle]:
    dirs = list_stem_samples(dataset_root, split=split)
    n = 0
    for d in dirs:
        sample = load_sample(d, **kwargs)
        if require_k is not None and len(sample.inside_stems) != require_k:
            continue
        yield sample
        n += 1
        if max_samples is not None and n >= max_samples:
            break


def load_promoted_model(run_dir: str = DEFAULT_RUN, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pl = load_torch_pretrained(run_dir, map_location=device)
    model = pl.model.eval()
    model.to(device)
    return model, device


@torch.no_grad()
def infer_bubble(model, mixture: np.ndarray, device: str) -> np.ndarray:
    """Return mono b_hat [T] from promoted model."""
    x = torch.from_numpy(np.asarray(mixture, dtype=np.float32)).unsqueeze(0).to(device)
    out = model({"mixture": x})
    y = out["output"]
    # [B, n_src, T] or [B, T]
    if y.ndim == 3:
        y = y[:, 0, :]
    return y[0].detach().cpu().numpy().astype(np.float32)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
