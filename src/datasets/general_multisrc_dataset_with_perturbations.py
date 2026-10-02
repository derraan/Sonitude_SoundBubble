"""
Torch dataset object for synthetically rendered spatial data.
"""
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

import helpers.utils as utils
from src.datasets.dsenet_beta import (
    DEFAULT_RHO,
    DEFAULT_SIGMA,
    DEFAULT_THETA0_GRID_DEG,
    product_weights,
    sample_theta0_deg,
)
from src.datasets.perturbations.audio_perturbations import AudioPerturbations


class Dataset(torch.utils.data.Dataset):
    """
    Dataset of mixed waveforms and corresponding ground-truth waveforms.

    Data format:
      - mixture: (n_microphones, duration)
      - target:  (n_reference_channels, duration)
      - target_steered (optional): DSENet-style distance×angle product mix
    """

    def __init__(
        self,
        dataset_dirs,
        n_mics=6,
        sr=48000,
        dis_threshold=1.5,
        directional=True,
        fair_compare=False,
        prob_neg=0,  # Unused; kept for compatibility with config files.
        perturbations=None,
        downsample=1,
        mic_config=None,
        sig_len=4.5,
        reference_channels=None,
        split="val",
        steered_targets=None,
    ):
        super().__init__()
        if perturbations is None:
            perturbations = []
        if mic_config is None:
            mic_config = []

        self.dirs = []
        for dataset_cfg in dataset_dirs:
            dirpath = dataset_cfg["path"]
            limit = int(dataset_cfg["max_samples"])
            samples = sorted(list(Path(dirpath).glob("[0-9]*")))[:limit]
            self.dirs.extend(samples)

        self.downsample = downsample
        self.mic_lists = mic_config
        if reference_channels is None:
            reference_channels = [0]
        self.reference_mics = reference_channels

        self.valid_dirs = []
        self.directional = directional
        self.n_mics = n_mics
        self.sr = sr
        self.dis_threshold = dis_threshold
        self.fair_compare = fair_compare
        self.sig_len = int(sig_len * sr / downsample)
        self.perturbations = AudioPerturbations(perturbations)
        self.split = split

        steered_cfg = steered_targets or {}
        self.steered_enabled = bool(steered_cfg.get("enabled", False))
        self.beta_sigma = float(steered_cfg.get("sigma", DEFAULT_SIGMA))
        self.beta_rho = float(steered_cfg.get("rho", DEFAULT_RHO))
        grid = steered_cfg.get("theta0_grid_deg", list(DEFAULT_THETA0_GRID_DEG))
        self.theta0_grid_deg = tuple(float(x) for x in grid)
        # Fixed look for val (reproducible); train samples from grid.
        self.theta0_fixed_deg = steered_cfg.get("theta0_fixed_deg", None)
        if self.theta0_fixed_deg is not None:
            self.theta0_fixed_deg = float(self.theta0_fixed_deg)

        for curr_dir in self.dirs:
            if os.path.exists(Path(curr_dir) / "metadata.json"):
                self.valid_dirs.append(curr_dir)

    def __len__(self) -> int:
        return len(self.valid_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        curr_dir = self.valid_dirs[idx % len(self.valid_dirs)]
        return self.get_mixture_and_gt(curr_dir, self.dis_threshold)

    def get_mixture_and_gt(self, curr_dir, dis_threshold):
        metadata = utils.read_json(os.path.join(curr_dir, "metadata.json"))
        voices = [key for key in metadata.keys() if "voice" in key]
        mics = self.mic_lists
        mics_all = [key for key in metadata.keys() if "mic" in key]

        assert self.n_mics == len(mics)

        mixture = utils.read_audio_file_torch(
            os.path.join(curr_dir, "mixture.wav"),
            self.downsample,
        )
        if len(mics) < mixture.shape[0]:
            mics_num = [int(mi[-2:]) for mi in mics]
            mixture = mixture[mics_num, :]

        target_voice_inside = torch.zeros((len(self.reference_mics), mixture.shape[-1]))
        target_voice_outside = torch.zeros((1, mixture.shape[-1]))
        num_tgt_speakers = 0
        inside_speakers = []

        real = metadata["real"]
        for voice in voices:
            # Synthetic metadata stores metres from array center. Real metadata
            # stores centimetres from recording mic origin and is converted here.
            d = int(metadata[voice]["dis"]) / 100 if real else metadata[voice]["dis"]
            if d <= dis_threshold:
                ref_paths = [
                    os.path.join(curr_dir, f"{mics_all[mic]}_{voice}.wav")
                    for mic in self.reference_mics
                ]
                # Some generated samples may mark a voice as inside by distance
                # while the corresponding target stem is absent; skip such voices
                # instead of hard-failing the entire batch.
                if not all(os.path.exists(p) for p in ref_paths):
                    continue
                ref_audio = utils.read_audio_file_torch(
                    ref_paths[0],
                    self.downsample,
                )[0]
                inside_speakers.append(
                    {
                        "distance_m": float(d),
                        "angle_deg": float(metadata[voice].get("angle", 0.0)),
                        "voice": voice,
                        "ref_audio": ref_audio,
                    }
                )
                for ch_idx, mic in enumerate(self.reference_mics):
                    audio = utils.read_audio_file_torch(ref_paths[ch_idx], self.downsample)
                    target_voice_inside[ch_idx] += audio[0]
                num_tgt_speakers += 1

        if num_tgt_speakers == 0:
            assert (
                torch.abs(target_voice_inside).max() == 0
            ), "When there are no inside speakers, the target should be zero"
        else:
            assert (
                torch.abs(target_voice_inside).max() > 0
            ), "When there is at least one speaker, the target should be more than zero"

        if self.sig_len < mixture.shape[-1]:
            delta_len = mixture.shape[-1] - self.sig_len
            begin_idx = np.random.randint(low=1000, high=delta_len - 1)
            mixture = mixture[..., begin_idx : begin_idx + self.sig_len]
            target_voice_inside = target_voice_inside[
                ..., begin_idx : begin_idx + self.sig_len
            ]
            for spk_meta in inside_speakers:
                spk_meta["ref_audio"] = spk_meta["ref_audio"][begin_idx : begin_idx + self.sig_len]

        inside_speakers = sorted(inside_speakers, key=lambda x: x["distance_m"])
        max_slots = 2
        spk_distance_m = torch.zeros(max_slots, dtype=torch.float32)
        spk_angle_deg = torch.zeros(max_slots, dtype=torch.float32)
        spk_valid = torch.zeros(max_slots, dtype=torch.float32)
        spk_ref = torch.zeros((max_slots, target_voice_inside.shape[-1]), dtype=target_voice_inside.dtype)
        for slot_idx, spk_meta in enumerate(inside_speakers[:max_slots]):
            spk_distance_m[slot_idx] = float(spk_meta["distance_m"])
            spk_angle_deg[slot_idx] = float(spk_meta.get("angle_deg", 0.0))
            spk_valid[slot_idx] = 1.0
            spk_ref[slot_idx] = spk_meta["ref_audio"].to(dtype=spk_ref.dtype)

        # DSENet product target over *all* inside speakers (not just K=2 slots).
        # Hard bubble GT (target_voice_inside) stays unchanged for b_hat.
        if self.steered_enabled:
            if self.theta0_fixed_deg is not None:
                theta0_deg = float(self.theta0_fixed_deg)
            elif self.split == "train":
                theta0_deg = sample_theta0_deg(grid_deg=self.theta0_grid_deg)
            else:
                theta0_deg = 0.0
            if inside_speakers:
                dists = [float(s["distance_m"]) for s in inside_speakers]
                angs = [float(s.get("angle_deg", 0.0)) for s in inside_speakers]
                w = product_weights(
                    dists,
                    angs,
                    theta0_deg=theta0_deg,
                    dis_threshold=self.dis_threshold,
                    sigma=self.beta_sigma,
                    rho=self.beta_rho,
                )
                stacked = torch.stack([s["ref_audio"] for s in inside_speakers], dim=0)
                target_steered = (stacked * torch.from_numpy(w).to(stacked.dtype).unsqueeze(-1)).sum(dim=0)
                target_steered = target_steered.unsqueeze(0).expand(len(self.reference_mics), -1)
                beta_weights = torch.zeros(max_slots, dtype=torch.float32)
                for slot_idx in range(min(max_slots, len(w))):
                    beta_weights[slot_idx] = float(w[slot_idx])
            else:
                target_steered = torch.zeros_like(target_voice_inside)
                beta_weights = torch.zeros(max_slots, dtype=torch.float32)
            theta0_rad = torch.tensor(np.deg2rad(theta0_deg), dtype=torch.float32)
        else:
            target_steered = None
            beta_weights = None
            theta0_rad = None
            theta0_deg = 0.0

        if self.split == "train":
            if target_steered is not None:
                gt_stack = torch.cat([target_voice_inside, spk_ref, target_steered], dim=0)
            else:
                gt_stack = torch.cat([target_voice_inside, spk_ref], dim=0)
            mixture, gt_stack = self.perturbations.apply_random_perturbations(
                mixture, gt_stack
            )
            n_ref = len(self.reference_mics)
            target_voice_inside = gt_stack[:n_ref]
            spk_ref = gt_stack[n_ref : n_ref + max_slots]
            if target_steered is not None:
                target_steered = gt_stack[n_ref + max_slots : n_ref + max_slots + n_ref]

        if num_tgt_speakers == 1:
            distance_m = float(spk_distance_m[0].item())
            distance_valid = 1.0
        else:
            distance_m = 0.0
            distance_valid = 0.0

        inputs = {
            "mixture": mixture.float(),
            "reference_channels": self.reference_mics,
        }
        if theta0_rad is not None:
            inputs["theta0_rad"] = theta0_rad
            inputs["theta0_deg"] = torch.tensor(theta0_deg, dtype=torch.float32)

        targets = {
            "target": target_voice_inside.float(),
            "targets_outside": target_voice_outside.float(),
            "num_target_speakers": num_tgt_speakers,
            "num_target_speakers_clamped": min(num_tgt_speakers, max_slots),
            "num_interfering_speakers": len(voices) - num_tgt_speakers,
            "num_noises": metadata["n_BG"],
            "distance_m": torch.tensor(distance_m, dtype=torch.float32),
            "distance_valid": torch.tensor(distance_valid, dtype=torch.float32),
            "spk_distance_m": spk_distance_m,
            "spk_angle_deg": spk_angle_deg,
            "spk_valid": spk_valid,
            "spk_ref": spk_ref.float(),
        }
        if target_steered is not None:
            targets["target_steered"] = target_steered.float()
            targets["beta_weights"] = beta_weights
            targets["beta_sigma"] = torch.tensor(self.beta_sigma, dtype=torch.float32)
            targets["beta_rho"] = torch.tensor(self.beta_rho, dtype=torch.float32)
        return inputs, targets
