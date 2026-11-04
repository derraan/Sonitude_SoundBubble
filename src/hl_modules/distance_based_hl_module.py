import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    import wandb
except ImportError:  # missing or broken install; --no_wandb still trains
    class _WandbStub:
        class Audio:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class Table:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class plot:
            @staticmethod
            def histogram(*args, **kwargs):
                return {"type": "histogram", "args": args, "kwargs": kwargs}

        @staticmethod
        def init(*args, **kwargs):
            raise RuntimeError("wandb is not installed in this environment.")

    wandb = _WandbStub()
import torch
from numpy import mean
from src.metrics.metrics import Metrics
from src.metrics.metrics import compute_decay
import src.utils as utils
from transformers.debug_utils import DebugUnderflowOverflow
import numpy as np 

class FakeModel(nn.Module):
    def __init__(self, model):
        super(FakeModel, self).__init__()
        self.model = model


class PLModule(object):
    def __init__(self, model, model_params, sr,
                 optimizer, optimizer_params,
                 scheduler=None, scheduler_params=None,
                 loss=None, loss_params=None, 
                 metrics=[], init_ckpt=None,
                 init_head_ckpt=None,
                 grad_clip = None,
                 use_dp=True,
                 val_log_interval=10, # Unused, only kept for compatibility TODO: Remove
                 samples_per_speaker_number=3,
                 count_distance_head=None,
                 distance_head=None,
                 speaker_head=None,
                 steered_head=None,
                 distance_sweep_val=None,
                 pos_keep_loss=None,
                 elev_aux=None):

        if distance_head is not None and distance_head.get("enabled", False):
            raise ValueError(
                "distance_head is removed. Use count_distance_head for combined speaker count + distance."
            )
        if speaker_head is not None and speaker_head.get("enabled", False):
            raise ValueError(
                "speaker_head is renamed to count_distance_head (combined speaker count + distance)."
            )

        self.count_distance_head_cfg = count_distance_head or {}
        self.count_distance_head_enabled = bool(self.count_distance_head_cfg.get('enabled', False))
        self.distance_head_enabled = False

        self.steered_head_cfg = steered_head or {}
        self.steered_head_enabled = bool(self.steered_head_cfg.get('enabled', False))
        self.steered_loss_weight = float(self.steered_head_cfg.get('w_steered', 1.0))
        # When steered training freezes backbone, count head is also frozen unless
        # train_count_heads is explicitly true.
        self.steered_train_count_heads = bool(self.steered_head_cfg.get('train_count_heads', False))
        # θ0 sweep: full 360° at 30° steps (12 directions); look−no-look SI-SDR
        # delta capped at look_delta_cap_db.
        from src.datasets.dsenet_beta import DEFAULT_LOOK_DELTA_CAP_DB, DEFAULT_THETA0_GRID_DEG
        grid = self.steered_head_cfg.get("theta0_grid_deg", list(DEFAULT_THETA0_GRID_DEG))
        self.steered_theta0_grid_deg = tuple(float(x) % 360.0 for x in grid)
        self.look_delta_cap_db = float(self.steered_head_cfg.get("look_delta_cap_db", DEFAULT_LOOK_DELTA_CAP_DB))
        self.look_delta_loss_weight = float(self.steered_head_cfg.get("w_look_delta", 0.5))
        # Run full θ0 sweep every N train steps (always on val). Prefer N>=4 when
        # the grid has 12 angles (12 extra forwards per sweep).
        self.theta0_sweep_every_n = int(self.steered_head_cfg.get("theta0_sweep_every_n", 4))
        self._steered_train_step = 0

        self.freeze_backbone = bool(
            self.count_distance_head_cfg.get('freeze_backbone', False)
            or self.steered_head_cfg.get('freeze_backbone', False)
        )
        self.count_loss_weight = float(self.count_distance_head_cfg.get('w_count', 1.0))
        self.spk_distance_loss_weight = float(self.count_distance_head_cfg.get('w_distance', 1.0))
        self.spk_activity_loss_weight = float(self.count_distance_head_cfg.get('w_activity', 0.2))
        self.spk_vad_threshold_db = float(self.count_distance_head_cfg.get('vad_threshold_db', -35.0))
        self.has_aux_heads = bool(self.count_distance_head_enabled or self.steered_head_enabled)

        model_params = dict(model_params)
        if self.count_distance_head_enabled:
            model_cd_cfg = dict(model_params.get("count_distance_head") or {})
            model_cd_cfg.setdefault("enabled", True)
            model_params["count_distance_head"] = model_cd_cfg
            model_params.pop("distance_head", None)
            model_params.pop("speaker_head", None)
        if self.steered_head_enabled:
            model_st_cfg = dict(model_params.get("steered_head") or {})
            model_st_cfg.setdefault("enabled", True)
            model_params["steered_head"] = model_st_cfg

        self.model = utils.import_attr(model)(**model_params)
        self.use_dp = use_dp
        if use_dp:
            self.model = nn.DataParallel(self.model)
        
        self.sr = sr
        
        #debug_overflow = DebugUnderflowOverflow(self.model)
        # Log a val sample every this many intervals
        # self.val_log_interval = val_log_interval
        self.samples_per_speaker_number = samples_per_speaker_number
        
        # Initialize metrics
        self.metrics = [Metrics(metric) for metric in metrics]

        # Metric values
        self.metric_values = {}
        
        # Dataset statistics
        self.statistics = {}
       
        # Assine metric to monitor, and how to judge different models based on it
        # i.e. How do we define the best model (Here, we minimize val loss)
        self.monitor = 'val/loss'
        self.monitor_mode = 'min'

        # Mode, either train or val
        self.mode = None

        self.val_samples = {}
        self.train_samples = {}

        self.input_snr_calculated = False
        self.input_snr = []
        self.snr_metric = Metrics("snr")

        # Initialize loss function
        self.loss_fn = utils.import_attr(loss)(**loss_params)
        
        # Initaize weights if checkpoint is provided
        # Warning: This will only load the weights of the module
        # called "model" in this class
        # Optional resume metadata from a .pt init_ckpt (epoch counter / metrics).
        # Optimizer is intentionally NOT restored here — finetune uses a fresh Adam.
        self._init_ckpt_resume_epoch = None
        self._init_ckpt_resume_metrics = None
        self._init_ckpt_resume_statistics = None

        if init_ckpt is not None:
            if init_ckpt.endswith('.ckpt'):
                state = torch.load(init_ckpt)['state_dict']
                # print(state.keys())
                
                if self.use_dp:
                    _model = self.model.module
                else:
                    _model = self.model
                
                mdl = FakeModel(_model)
                if self.count_distance_head_enabled or self.steered_head_enabled:
                    missing, unexpected = mdl.load_state_dict(state, strict=False)
                    missing_backbone = [k for k in missing if not self._is_head_key(k)]
                    unexpected_backbone = [k for k in unexpected if not self._is_head_key(k)]
                    if missing_backbone or unexpected_backbone:
                        raise RuntimeError(
                            "Backbone strict-load failed for Lightning init_ckpt. "
                            f"Missing backbone keys: {missing_backbone[:5]}, "
                            f"Unexpected backbone keys: {unexpected_backbone[:5]}"
                        )
                else:
                    mdl.load_state_dict(state)
                self.model = nn.DataParallel(mdl.model)
            else:
                blob = torch.load(init_ckpt, map_location="cpu")
                if isinstance(blob, dict) and "model" in blob:
                    self._load_init_state(blob["model"])
                    if "current_epoch" in blob and blob["current_epoch"] is not None:
                        self._init_ckpt_resume_epoch = int(blob["current_epoch"])
                    if "metric_values" in blob and blob["metric_values"] is not None:
                        self._init_ckpt_resume_metrics = blob["metric_values"]
                    if "statistics" in blob and blob["statistics"] is not None:
                        self._init_ckpt_resume_statistics = blob["statistics"]
                else:
                    self._load_init_state(blob)

        # Overlay count/distance head weights from a second checkpoint (e.g. ep93
        # backbone + later head-only or joint-head checkpoint).
        if init_head_ckpt is not None:
            if not self.count_distance_head_enabled:
                raise ValueError("init_head_ckpt requires count_distance_head.enabled=true")
            head_blob = torch.load(init_head_ckpt, map_location="cpu")
            head_state = head_blob["model"] if isinstance(head_blob, dict) and "model" in head_blob else head_blob
            self._load_head_state(head_state)
            print(f"Loaded count/distance head weights from {init_head_ckpt}")

        self.distance_sweep_val_cfg = distance_sweep_val or {}
        self.distance_sweep_val_enabled = bool(self.distance_sweep_val_cfg.get("enabled", False))

        # Edge / passband aux (multi-level × mild RT60). Val sweep still logs ripple.
        # Train aux when any weight > 0 (needs trainable backbone).
        pr_cfg = self.distance_sweep_val_cfg.get("passband_ripple") or {}
        edge_cfg = self.distance_sweep_val_cfg.get("edge_aux") or pr_cfg
        self.passband_ripple_w = float(
            edge_cfg.get("w_passband_ripple", self.distance_sweep_val_cfg.get("w_passband_ripple", 0.0))
        )
        self.passband_keep_w = float(edge_cfg.get("w_passband_keep", 0.0))
        self.edge_w = float(edge_cfg.get("w_edge", 0.0))
        self.suppression_w = float(edge_cfg.get("w_suppression", 0.0))
        self.passband_ripple_every_n = int(
            edge_cfg.get(
                "every_n_train_steps",
                self.distance_sweep_val_cfg.get("passband_ripple_every_n_steps", 40),
            )
        )
        self.passband_ripple_rt60 = float(edge_cfg.get("rt60", 0.2))  # legacy single-condition
        self.passband_ripple_bubble_m = float(
            edge_cfg.get("bubble_m", self.distance_sweep_val_cfg.get("bubble_m", 1.5))
        )
        self.edge_levels_dbfs = edge_cfg.get("levels_dbfs")
        self.edge_rt60_list = edge_cfg.get("rt60_list")
        self.passband_ripple_distances = (
            edge_cfg.get("distances")
            or pr_cfg.get("distances")
            or self.distance_sweep_val_cfg.get("distances")
        )
        self.passband_slack_db = float(edge_cfg.get("passband_slack_db", 6.0))
        self.min_edge_drop_db = float(edge_cfg.get("min_edge_drop_db", 20.0))
        self.suppression_drop_db = float(edge_cfg.get("suppression_drop_db", 6.0))
        self.edge_near_m = float(edge_cfg.get("edge_near_m", 1.4))
        self.edge_far_m = float(edge_cfg.get("edge_far_m", 2.0))
        self._edge_aux_enabled = bool(
            self.passband_ripple_w > 0.0
            or self.passband_keep_w > 0.0
            or self.edge_w > 0.0
            or self.suppression_w > 0.0
        )
        self._use_multi_condition_edge = bool(
            self.edge_levels_dbfs is not None or self.edge_rt60_list is not None
            or self.passband_keep_w > 0.0 or self.edge_w > 0.0 or self.suppression_w > 0.0
        )
        self._passband_mix_cache = None  # legacy single-condition [N, 6, T]
        self._passband_mix_distances = None
        self._edge_mix_cache = None  # dict[(level, rt60)] -> [D, 6, T]
        self._edge_mix_distances = None
        self._edge_cache_levels = None
        self._edge_cache_rt60s = None
        self._passband_train_step = 0
        self._last_edge_aux_metrics = {}
        if self._edge_aux_enabled and self.freeze_backbone:
            print(
                "[edge_aux] weights>0 but freeze_backbone=True — "
                "train edge/passband aux is skipped until backbone is trainable; "
                "val/sweep_* metrics still log from distance sweep."
            )

        # Positive keep-energy hinge: when n_speakers>0, penalize est quieter than GT
        # by more than slack_db. Fights over-gating to ~-90 dBFS on in-bubble clips.
        pos_keep_cfg = pos_keep_loss or {}
        self.pos_keep_enabled = bool(pos_keep_cfg.get("enabled", False))
        self.pos_keep_w = float(pos_keep_cfg.get("weight", 0.5))
        self.pos_keep_slack_db = float(pos_keep_cfg.get("slack_db", 6.0))
        if self.pos_keep_enabled:
            print(
                f"[pos_keep] enabled weight={self.pos_keep_w} "
                f"slack_db={self.pos_keep_slack_db}"
            )
            if self.freeze_backbone:
                print(
                    "[pos_keep] freeze_backbone=True — keep hinge skipped "
                    "(needs trainable backbone)."
                )

        # Elevation aux: flatten vertical response / shrink ≈−20° notch on in-bubble keep.
        elev_cfg = elev_aux or {}
        # Also allow nesting under distance_sweep_val.elev_aux
        if not elev_cfg and isinstance(self.distance_sweep_val_cfg.get("elev_aux"), dict):
            elev_cfg = self.distance_sweep_val_cfg["elev_aux"]
        self.elev_keep_w = float(elev_cfg.get("w_elev_keep", 0.0))
        self.elev_notch_w = float(elev_cfg.get("w_elev_notch", 0.0))
        self.elev_ripple_w = float(elev_cfg.get("w_elev_ripple", 0.0))
        self.elev_aux_every_n = int(
            elev_cfg.get("every_n_train_steps", self.passband_ripple_every_n)
        )
        self.elev_aux_elevations = elev_cfg.get("elevations_deg")
        self.elev_aux_distances = elev_cfg.get("distances")
        self.elev_notch_elevations = elev_cfg.get("notch_elevations_deg")
        self.elev_ref_elevation = float(elev_cfg.get("ref_elevation_deg", 0.0))
        self.elev_max_notch_drop_db = float(elev_cfg.get("max_notch_drop_db", 3.0))
        self.elev_keep_slack_db = float(
            elev_cfg.get("passband_slack_db", self.passband_slack_db)
        )
        self.elev_target_in_dbfs = float(
            elev_cfg.get(
                "target_in_dbfs",
                self.distance_sweep_val_cfg.get("target_in_dbfs", -25.0),
            )
        )
        self.elev_rt60 = float(elev_cfg.get("rt60", 0.2))
        self._elev_aux_enabled = bool(
            self.elev_keep_w > 0.0 or self.elev_notch_w > 0.0 or self.elev_ripple_w > 0.0
        )
        self._elev_mix_cache = None  # [E, D, 6, T]
        self._elev_cache_elevations = None
        self._elev_cache_distances = None
        self._elev_train_step = 0
        self._last_elev_aux_metrics = {}
        if self._elev_aux_enabled:
            print(
                f"[elev_aux] enabled w_keep={self.elev_keep_w} "
                f"w_notch={self.elev_notch_w} w_ripple={self.elev_ripple_w} "
                f"every_n={self.elev_aux_every_n} "
                f"max_notch_drop_db={self.elev_max_notch_drop_db}"
            )
            if self.freeze_backbone:
                print(
                    "[elev_aux] freeze_backbone=True — elev aux skipped "
                    "(needs trainable backbone)."
                )

        if self.freeze_backbone:
            self._freeze_backbone_for_heads()

         # Initialize optimizer
        optim_params = self.model.parameters()
        if self.freeze_backbone:
            optim_params = [p for p in self.model.parameters() if p.requires_grad]
            if len(optim_params) == 0:
                raise RuntimeError("freeze_backbone=True but no trainable head parameters were found.")
        self.optimizer = utils.import_attr(optimizer)(optim_params, **optimizer_params)
        self.optim_name = optimizer
        self.opt_params = optimizer_params

        # Grad clip
        self.grad_clip = grad_clip

        if self.grad_clip is not None:
            print(f"USING GRAD CLIP: {self.grad_clip}")
        else:
            print("ERROR! NOT USING GRAD CLIP" * 100)

        # Initialize scheduler
        self.scheduler = self.init_scheduler(scheduler, scheduler_params)
        self.scheduler_name = scheduler
        self.scheduler_params = scheduler_params
        
        self.epoch = 0
        # Continue epoch numbering from init_ckpt when present (contingency finetune).
        # Note: best.pt is dumped inside on_epoch_end *before* epoch += 1, so if
        # metric_values already contains that epoch index, advance by one.
        if self._init_ckpt_resume_epoch is not None:
            self.epoch = self._init_ckpt_resume_epoch
            if self._init_ckpt_resume_metrics is not None:
                self.metric_values = self._init_ckpt_resume_metrics
            if self._init_ckpt_resume_statistics is not None:
                self.statistics = self._init_ckpt_resume_statistics
            if self.epoch in self.metric_values:
                print(
                    f"[init_ckpt] epoch {self.epoch} already logged in metric_values; "
                    f"starting next epoch {self.epoch + 1}"
                )
                self.epoch += 1
            print(
                f"[init_ckpt] continuing from current_epoch={self.epoch} "
                f"(metric_values entries={len(self.metric_values)}; fresh optimizer)"
            )
        # First epoch index trained by this run (best.pt only competes from here).
        self._run_start_epoch = self.epoch

    @staticmethod
    def _is_count_distance_head_key(key: str) -> bool:
        # Accept legacy speaker_head.* keys from older checkpoints.
        return ("count_distance_head" in key) or ("speaker_head" in key)

    @staticmethod
    def _is_steered_head_key(key: str) -> bool:
        return "steered_head" in key

    @classmethod
    def _is_head_key(cls, key: str) -> bool:
        return cls._is_count_distance_head_key(key) or cls._is_steered_head_key(key)

    @staticmethod
    def _remap_legacy_head_keys(state_dict: dict) -> dict:
        """Map old module name speaker_head -> count_distance_head."""
        return {
            key.replace("speaker_head", "count_distance_head"): value
            for key, value in state_dict.items()
        }

    def _inner_model(self):
        return self.model.module if self.use_dp else self.model

    def _load_init_state(self, state_dict):
        model = self._inner_model()
        state_dict = self._remap_legacy_head_keys(state_dict)
        if not (self.count_distance_head_enabled or self.steered_head_enabled):
            model.load_state_dict(state_dict)
            return

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        missing_backbone = [k for k in missing if not self._is_head_key(k)]
        unexpected_backbone = [k for k in unexpected if not self._is_head_key(k)]
        if missing_backbone or unexpected_backbone:
            raise RuntimeError(
                "Backbone strict-load failed for init_ckpt. "
                f"Missing backbone keys: {missing_backbone[:5]}, "
                f"Unexpected backbone keys: {unexpected_backbone[:5]}"
            )

    def _load_head_state(self, state_dict):
        """Load only count_distance_head / legacy speaker_head keys into the model."""
        model = self._inner_model()
        state_dict = self._remap_legacy_head_keys(state_dict)
        head_only = {k: v for k, v in state_dict.items() if self._is_head_key(k)}
        if not head_only:
            raise RuntimeError("init_head_ckpt contained no count_distance_head / speaker_head keys.")
        missing, unexpected = model.load_state_dict(head_only, strict=False)
        missing_heads = [k for k in missing if self._is_head_key(k)]
        if missing_heads:
            raise RuntimeError(
                "Head load incomplete for init_head_ckpt. "
                f"Missing head keys: {missing_heads[:8]}"
            )
        unexpected_heads = [k for k in unexpected if self._is_head_key(k)]
        if unexpected_heads:
            raise RuntimeError(
                "Unexpected head keys while loading init_head_ckpt: "
                f"{unexpected_heads[:8]}"
            )

    def _freeze_backbone_for_heads(self):
        total = 0
        trainable = 0
        for name, param in self.model.named_parameters():
            if self.steered_head_enabled and not self.steered_train_count_heads:
                # Stage H default: only train steered_head; freeze backbone + count head.
                trainable_key = self._is_steered_head_key(name)
            else:
                trainable_key = self._is_head_key(name)
            param.requires_grad = trainable_key
            total += param.numel()
            if trainable_key:
                trainable += param.numel()
        print(f"[aux_head] backbone frozen. trainable params: {trainable}/{total}")

    def _compute_slot_activity_mask(
        self,
        spk_ref: torch.Tensor,
        spk_valid: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        # spk_ref: [B, S, T_audio] -> [B, T_frames, S]
        bsz, num_slots, _ = spk_ref.shape
        ratio = 10.0 ** (self.spk_vad_threshold_db / 10.0)
        power = spk_ref.pow(2).reshape(bsz * num_slots, 1, -1)
        pooled = F.adaptive_avg_pool1d(power, num_frames).reshape(bsz, num_slots, num_frames)
        peak = pooled.max(dim=-1, keepdim=True).values.clamp_min(1e-8)
        active = pooled > (peak * ratio)
        active = active & (spk_valid.unsqueeze(-1) > 0.0)
        return active.float().permute(0, 2, 1)

    def _si_sdr_db_torch(self, est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Per-item SI-SDR in dB. est/ref: [B, T] or [B, 1, T]."""
        from src.datasets.dsenet_beta import si_sdr_db_torch
        return si_sdr_db_torch(est, ref, eps=eps)

    def _looked_spk_indices(
        self,
        theta0_deg: torch.Tensor,
        spk_angle_deg: torch.Tensor,
        spk_valid: torch.Tensor,
    ) -> torch.Tensor:
        """For each batch item, index of valid speaker nearest to θ0 (or -1)."""
        bsz = spk_angle_deg.shape[0]
        out = torch.full((bsz,), -1, device=spk_angle_deg.device, dtype=torch.long)
        th = theta0_deg.reshape(-1).to(spk_angle_deg.device)
        if th.numel() == 1:
            th = th.expand(bsz)
        for b in range(bsz):
            valid = spk_valid[b] > 0.5
            if not valid.any():
                continue
            angs = spk_angle_deg[b]
            d = torch.abs(((angs - th[b] + 180.0) % 360.0) - 180.0)
            d = torch.where(valid, d, torch.full_like(d, 1e9))
            out[b] = int(torch.argmin(d).item())
        return out

    @torch.no_grad()
    def _theta0_sweep_look_deltas(
        self,
        inputs: dict,
        targets: dict,
        bubble: torch.Tensor,
        step: str,
        batch_size: int,
    ) -> None:
        """Forward steered head at each θ0 on the 360°/30° grid; log SI-SDR look/no-look deltas (capped)."""
        if not self.steered_head_enabled:
            return
        if "spk_ref" not in targets or "spk_angle_deg" not in targets:
            return
        spk_ref = targets["spk_ref"].float()
        spk_ang = targets["spk_angle_deg"].float()
        spk_valid = targets["spk_valid"].float()
        device = bubble.device
        cap = float(self.look_delta_cap_db)

        # Aggregate per-grid-angle and overall look delta.
        deltas_all = []
        for th_deg in self.steered_theta0_grid_deg:
            th_norm = float(th_deg) % 360.0
            th_key = f"{int(round(th_norm)):03d}"
            th = torch.full((batch_size,), th_norm, device=device, dtype=torch.float32)
            sweep_in = dict(inputs)
            sweep_in["theta0_rad"] = th * (np.pi / 180.0)
            sweep_in["theta0_deg"] = th
            out = self.model(sweep_in)
            z = out.get("steered_output")
            if z is None:
                continue
            if z.ndim == 3:
                z = z[:, 0, :]
            b = bubble
            if b.ndim == 3:
                b = b[:, 0, :]
            tlen = min(z.shape[-1], b.shape[-1], spk_ref.shape[-1])
            z = z[..., :tlen]
            b = b[..., :tlen]
            looked = self._looked_spk_indices(th, spk_ang, spk_valid)
            for i in range(batch_size):
                k = int(looked[i].item())
                if k < 0:
                    continue
                ref_k = spk_ref[i, k, :tlen]
                if float(ref_k.abs().max()) <= 0:
                    continue
                d_look = self._si_sdr_db_torch(z[i : i + 1], ref_k.unsqueeze(0))
                d_base = self._si_sdr_db_torch(b[i : i + 1], ref_k.unsqueeze(0))
                delta = float((d_look - d_base).item())
                capped = min(cap, delta) if np.isfinite(delta) else float("nan")
                if np.isfinite(capped):
                    deltas_all.append(capped)
                    self.log_metric(
                        f"{step}/look_delta_db_th{th_key}",
                        capped,
                        batch_size=1,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
                    # Per-speaker SI-SDR under this look
                    self.log_metric(
                        f"{step}/sisdr_look_th{th_key}",
                        float(d_look.item()),
                        batch_size=1,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )

        if deltas_all:
            mean_delta = float(np.mean(deltas_all))
            self.log_metric(
                f"{step}/look_delta_db_mean",
                mean_delta,
                batch_size=batch_size,
                on_step=(step == "train"),
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
    
    def load_state(self, path, map_location=None):
        state = torch.load(path, map_location=map_location)
        model_state = self._remap_legacy_head_keys(state['model'])

        if self.use_dp:
            self.model.module.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)
        
        # Re-initialize optimizer
        optim_params = self.model.parameters()
        if self.freeze_backbone:
            optim_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = utils.import_attr(self.optim_name)(optim_params, **self.opt_params)
        
        # Re-initialize scheduler (Order might be important?)
        if self.scheduler is not None:
            self.scheduler = self.init_scheduler(self.scheduler_name, self.scheduler_params)

        self.optimizer.load_state_dict(state['optimizer'])

        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        
        self.epoch = state['current_epoch']
        self.metric_values = state['metric_values']
        
        if 'statistics' in state:
            self.statistics = state['statistics']

    def dump_state(self, path):
        if self.use_dp:
            _model = self.model.module
        else:
            _model = self.model
        
        state = dict(model = _model.state_dict(),
                     optimizer = self.optimizer.state_dict(),
                     current_epoch = self.epoch,
                     metric_values=self.metric_values, 
                     statistics = self.statistics)
        
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        
        torch.save(state, path)

    def get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def on_epoch_start(self):
        print()
        print("=" * 25, "STARTING EPOCH", self.epoch, "=" * 25)
        print()

    def get_avg_metric_at_epoch(self, metric, epoch = None):
        if epoch is None:
            epoch = self.epoch
        
        return self.metric_values[epoch][metric]['epoch'] / \
            self.metric_values[epoch][metric]['num_elements']

    def _try_get_avg_metric_at_epoch(self, metric, epoch=None, default=float('nan')):
        try:
            return self.get_avg_metric_at_epoch(metric, epoch=epoch)
        except Exception:
            return default

    def _ensure_passband_mix_cache(self):
        if self._passband_mix_cache is not None:
            return
        from src.validation.distance_sweep_dbfs import build_passband_mix_cache

        cfg = self.distance_sweep_val_cfg
        mixes, dists = build_passband_mix_cache(
            speech_path=cfg.get(
                "speech_path",
                r"C:\Users\darre\Sound_Bubble\datasets\_speech_cache\5694-64038-0000.flac",
            ),
            distances=self.passband_ripple_distances,
            bubble_m=self.passband_ripple_bubble_m,
            rt60=self.passband_ripple_rt60,
            target_in_dbfs=float(cfg.get("target_in_dbfs", -25.0)),
            sr=int(cfg.get("sr", self.sr)),
            dur_s=float(cfg.get("dur_s", 2.0)),
        )
        self._passband_mix_cache = mixes  # CPU float32 [N, 6, T]
        self._passband_mix_distances = dists
        print(
            f"[passband_ripple] cached {len(dists)} mixes "
            f"(RT60={self.passband_ripple_rt60}s, d<={max(dists):.2f}m, "
            f"w={self.passband_ripple_w}, every_n={self.passband_ripple_every_n})"
        )

    def _ensure_edge_mix_cache(self):
        if self._edge_mix_cache is not None:
            return
        from src.validation.distance_sweep_dbfs import (
            DEFAULT_EDGE_DISTANCES,
            DEFAULT_EDGE_LEVELS_DBFS,
            DEFAULT_EDGE_RT60_LIST,
            build_edge_aux_mix_cache,
        )

        cfg = self.distance_sweep_val_cfg
        cache, dists, levels, rt60s = build_edge_aux_mix_cache(
            speech_path=cfg.get(
                "speech_path",
                r"C:\Users\darre\Sound_Bubble\datasets\_speech_cache\5694-64038-0000.flac",
            ),
            levels_dbfs=self.edge_levels_dbfs or DEFAULT_EDGE_LEVELS_DBFS,
            rt60_list=self.edge_rt60_list or DEFAULT_EDGE_RT60_LIST,
            distances=self.passband_ripple_distances or DEFAULT_EDGE_DISTANCES,
            sr=int(cfg.get("sr", self.sr)),
            dur_s=float(cfg.get("dur_s", 2.0)),
        )
        self._edge_mix_cache = cache
        self._edge_mix_distances = dists
        self._edge_cache_levels = levels
        self._edge_cache_rt60s = rt60s
        print(
            f"[edge_aux] cached {len(cache)} conditions "
            f"(levels={levels}, rt60={rt60s}, d={dists}, "
            f"w_keep={self.passband_keep_w}, w_edge={self.edge_w}, "
            f"w_supp={self.suppression_w}, w_ripple={self.passband_ripple_w}, "
            f"every_n={self.passband_ripple_every_n})"
        )

    @staticmethod
    def _signal_dbfs(x: torch.Tensor) -> torch.Tensor:
        """Per-sample dBFS from mean-square energy. x: [B, C, T] or [B, T] -> [B]."""
        if x.dim() == 3:
            x = x[:, 0]
        power = torch.mean(x * x, dim=-1).clamp_min(1e-12)
        return 10.0 * torch.log10(power)

    def _compute_pos_keep_hinge(self, est, gt, n_speakers):
        """Hinge when in-bubble est is quieter than GT by more than slack_db.

        Returns (hinge_scalar, mean_est_db_on_positives). Both are 0 if no positives.
        """
        pos_mask = n_speakers > 0
        if not bool(pos_mask.any()):
            z = est.new_tensor(0.0)
            return z, z
        est_db = self._signal_dbfs(est[pos_mask])
        gt_db = self._signal_dbfs(gt[pos_mask])
        hinge = torch.relu((gt_db - self.pos_keep_slack_db) - est_db).mean()
        return hinge, est_db.mean()

    def _ensure_elev_mix_cache(self):
        if self._elev_mix_cache is not None:
            return
        from src.validation.distance_sweep_dbfs import (
            DEFAULT_ELEV_AUX_DISTANCES,
            DEFAULT_ELEV_AUX_ELEVATIONS_DEG,
            build_elev_aux_mix_cache,
        )

        cfg = self.distance_sweep_val_cfg
        mixes, elevs, dists = build_elev_aux_mix_cache(
            speech_path=cfg.get(
                "speech_path",
                r"C:\Users\darre\Sound_Bubble\datasets\_speech_cache\5694-64038-0000.flac",
            ),
            elevations_deg=self.elev_aux_elevations or DEFAULT_ELEV_AUX_ELEVATIONS_DEG,
            distances=self.elev_aux_distances or DEFAULT_ELEV_AUX_DISTANCES,
            target_in_dbfs=self.elev_target_in_dbfs,
            rt60=self.elev_rt60,
            sr=int(cfg.get("sr", self.sr)),
            dur_s=float(cfg.get("dur_s", 2.0)),
        )
        self._elev_mix_cache = mixes
        self._elev_cache_elevations = elevs
        self._elev_cache_distances = dists
        print(
            f"[elev_aux] cached elev×dist grid "
            f"E={elevs} D={dists} rt60={self.elev_rt60} "
            f"target={self.elev_target_in_dbfs} dBFS"
        )

    def _compute_elev_aux_loss(self):
        """Forward elevation×in-bubble grid; return weighted elev-aux loss."""
        from src.validation.distance_sweep_dbfs import (
            DEFAULT_ELEV_NOTCH_DEG,
            db_fs_torch,
            elev_aux_losses_torch,
        )

        self._ensure_elev_mix_cache()
        device = next(self._inner_model().parameters()).device
        mixes = self._elev_mix_cache.to(device=device, non_blocking=True)  # [E,D,6,T]
        e_dim, d_dim = mixes.shape[0], mixes.shape[1]
        flat = mixes.reshape(e_dim * d_dim, mixes.shape[2], mixes.shape[3])
        out = self.model({"mixture": flat})["output"]
        levels = db_fs_torch(out).reshape(e_dim, d_dim)
        terms = elev_aux_losses_torch(
            levels,
            self._elev_cache_elevations,
            target_in_dbfs=self.elev_target_in_dbfs,
            passband_slack_db=self.elev_keep_slack_db,
            notch_elevations_deg=self.elev_notch_elevations or DEFAULT_ELEV_NOTCH_DEG,
            ref_elevation_deg=self.elev_ref_elevation,
            max_notch_drop_db=self.elev_max_notch_drop_db,
        )
        loss = (
            self.elev_keep_w * terms["keep_hinge"]
            + self.elev_notch_w * terms["notch_hinge"]
            + self.elev_ripple_w * terms["elev_ripple_db"]
        )
        self._last_elev_aux_metrics = {
            "elev_keep_db": float(terms["keep_hinge"].detach().item()),
            "elev_notch_db": float(terms["notch_hinge"].detach().item()),
            "elev_ripple_db": float(terms["elev_ripple_db"].detach().item()),
            "elev_notch_drop_db": float(terms["notch_drop_db"].detach().item()),
            "elev_ref_passband_db": float(terms["ref_passband_db"].detach().item()),
        }
        return loss

    def _compute_passband_ripple_loss(self):
        """Legacy single-condition peak−peak dBFS ripple (RT60≈0.2 interior)."""
        from src.validation.distance_sweep_dbfs import db_fs_torch, passband_ripple_torch

        self._ensure_passband_mix_cache()
        device = next(self._inner_model().parameters()).device
        mixes = self._passband_mix_cache.to(device=device, non_blocking=True)
        out = self.model({"mixture": mixes})["output"]
        levels = db_fs_torch(out)
        return passband_ripple_torch(levels)

    def _compute_edge_aux_loss(self):
        """Sample one (level, RT60) and return weighted edge-aux loss + metrics."""
        import random

        from src.validation.distance_sweep_dbfs import db_fs_torch, edge_aux_losses_torch

        self._ensure_edge_mix_cache()
        level = float(random.choice(self._edge_cache_levels))
        rt60 = float(random.choice(self._edge_cache_rt60s))
        device = next(self._inner_model().parameters()).device
        mixes = self._edge_mix_cache[(level, rt60)].to(device=device, non_blocking=True)
        out = self.model({"mixture": mixes})["output"]
        levels = db_fs_torch(out)
        terms = edge_aux_losses_torch(
            levels,
            self._edge_mix_distances,
            target_in_dbfs=level,
            edge_near_m=self.edge_near_m,
            edge_far_m=self.edge_far_m,
            passband_slack_db=self.passband_slack_db,
            min_edge_drop_db=self.min_edge_drop_db,
            bubble_m=self.passband_ripple_bubble_m,
            suppression_drop_db=self.suppression_drop_db,
        )
        loss = (
            self.passband_keep_w * terms["keep_hinge"]
            + self.edge_w * terms["edge_hinge"]
            + self.passband_ripple_w * terms["ripple"]
            + self.suppression_w * terms["suppression_dev"]
        )
        self._last_edge_aux_metrics = {
            "passband_keep_db": float(terms["keep_hinge"].detach().item()),
            "edge_drop_db": float(terms["edge_drop_db"].detach().item()),
            "passband_ripple_db": float(terms["ripple"].detach().item()),
            "suppression_m": float(terms["suppression_m"].detach().item()),
            "suppression_dev_m": float(terms["suppression_dev"].detach().item()),
            "passband_mean_db": float(terms["passband_mean_db"].detach().item()),
            "edge_level_dbfs": level,
            "edge_rt60": rt60,
        }
        return loss

    def _run_and_log_distance_sweep_val(self, best_path):
        """Epoch-end LibriSpeech distance→dBFS sweep (validation monitor)."""
        from src.validation.distance_sweep_dbfs import run_distance_sweep, write_sweep_csv

        cfg = self.distance_sweep_val_cfg
        device = next(self._inner_model().parameters()).device
        # DataParallel forward expects the wrapped module.
        model = self.model
        bubble_m = float(cfg.get("bubble_m", 1.5))
        sweep = run_distance_sweep(
            model,
            device=str(device),
            speech_path=cfg.get(
                "speech_path",
                r"C:\Users\darre\Sound_Bubble\datasets\_speech_cache\5694-64038-0000.flac",
            ),
            distances=cfg.get("distances"),
            rt60_list=cfg.get("rt60_list"),
            target_in_dbfs=float(cfg.get("target_in_dbfs", -25.0)),
            bubble_m=bubble_m,
            sr=int(cfg.get("sr", self.sr)),
            dur_s=float(cfg.get("dur_s", 2.0)),
        )
        summary = sweep["summary"]
        ripple = summary.get("passband_ripple_0_2_db", float("nan"))
        supp_m = summary.get("suppression_m_m6db", float("nan"))
        supp_dev = summary.get("suppression_dev_from_bubble_m", float("nan"))
        print(
            "[VAL distance-dBFS sweep] "
            f"0.5m={summary['out_dbfs_0_5m']:.1f}  "
            f"0.8m={summary['out_dbfs_0_8m']:.1f}  "
            f"1.0m={summary['out_dbfs_1_0m']:.1f}  "
            f"1.2m={summary['out_dbfs_1_2m']:.1f}  "
            f"1.5m={summary['out_dbfs_1_5m']:.1f}  "
            f"2.0m={summary['out_dbfs_2_0m']:.1f}  "
            f"3.0m={summary['out_dbfs_3_0m']:.1f}  "
            f"pass-far={summary['passband_minus_far_db']:.1f}dB  "
            f"edge(1.5->2.0)={summary['edge_drop_1_5_to_2_0_db']:.1f}dB  "
            f"passband_ripple_0.2={ripple:.1f}dB  "
            f"supp@-6dB={supp_m:.2f}m  "
            f"|supp-bubble|={supp_dev:.2f}m  "
            f"(bubble={bubble_m}m)"
        )
        for key, value in summary.items():
            self.log_metric(f"val/sweep_{key}", float(value), batch_size=1, on_step=False, on_epoch=True)

        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(best_path)))
        sweep_dir = os.path.join(run_dir, "val_distance_sweep")
        os.makedirs(sweep_dir, exist_ok=True)
        csv_path = os.path.join(sweep_dir, f"epoch_{self.epoch:03d}.csv")
        write_sweep_csv(csv_path, sweep)
        # Keep a rolling "latest" copy for quick inspection.
        write_sweep_csv(os.path.join(sweep_dir, "latest.csv"), sweep)

    def on_epoch_end(self, best_path, wandb_run):
        assert self.epoch + 1 == len(self.metric_values), \
            "Current epoch must be equal to length of metrics (0-indexed)"

        monitor_metric_last = self.get_avg_metric_at_epoch(self.monitor)

        # Go over all epochs
        save = True
        if not np.isfinite(monitor_metric_last):
            # NaN/Inf must never become best.pt (NaN comparisons are always False).
            save = False
            print(f"Skipping best checkpoint save: monitor '{self.monitor}' is {monitor_metric_last}")
        else:
            # Only compare against epochs trained in *this* run. Epochs inherited
            # from init_ckpt metric_values were scored on a different val split
            # (e.g. syn-only EP35 lineage), so comparing to them is meaningless.
            for epoch in range(self._run_start_epoch, len(self.metric_values) - 1):
                monitor_metric_at_epoch = self.get_avg_metric_at_epoch(self.monitor, epoch)
                if not np.isfinite(monitor_metric_at_epoch):
                    continue

                if self.monitor_mode == 'max':
                    # If there is any model with monitor larger than current, then
                    # this is not the best model
                    if monitor_metric_last < monitor_metric_at_epoch:
                        save = False
                        break

                if self.monitor_mode == 'min':
                    # If there is any model with monitor smaller than current, then
                    # this is not the best model
                    if monitor_metric_last > monitor_metric_at_epoch:
                        save = False
                        break
        
        # If this is best, save it
        if save:
            print("Current checkpoint is the best! Saving it...")
            self.dump_state(best_path)

        if self.distance_sweep_val_enabled:
            self._run_and_log_distance_sweep_val(best_path)
        
        val_loss = self._try_get_avg_metric_at_epoch('val/loss')
        print(f'Val loss: {val_loss:.04f}')

        if self.count_distance_head_enabled:
            val_count_loss = self._try_get_avg_metric_at_epoch('val/count_loss')
            val_count_acc = self._try_get_avg_metric_at_epoch('val/count_acc')
            val_spk_distance_loss = self._try_get_avg_metric_at_epoch('val/spk_distance_loss')
            val_spk_activity_loss = self._try_get_avg_metric_at_epoch('val/spk_activity_loss')
            val_spk_distance_mae = self._try_get_avg_metric_at_epoch('val/spk_distance_mae')
            print(f'Val count loss (CE): {val_count_loss:.04f}')
            print(f'Val count acc: {val_count_acc:.04f}')
            print(f'Val spk distance loss (SmoothL1): {val_spk_distance_loss:.04f}')
            print(f'Val spk activity loss (BCE): {val_spk_activity_loss:.04f}')
            print(f'Val spk distance MAE: {val_spk_distance_mae:.04f} m')

        if self.steered_head_enabled:
            val_steered_loss = self._try_get_avg_metric_at_epoch('val/steered_loss')
            val_look_delta = self._try_get_avg_metric_at_epoch('val/look_delta_db')
            val_look_delta_mean = self._try_get_avg_metric_at_epoch('val/look_delta_db_mean')
            val_oracle_delta = self._try_get_avg_metric_at_epoch('val/oracle_look_delta_db')
            print(f'Val steered loss: {val_steered_loss:.04f}')
            print(f'Val look-delta SI-SDR (capped {self.look_delta_cap_db:.0f} dB): {val_look_delta:.02f} dB')
            print(f'Val θ0-sweep mean look-delta: {val_look_delta_mean:.02f} dB')
            print(f'Val oracle look-delta (β-modulated): {val_oracle_delta:.02f} dB')

        # Audio enhancement metrics are secondary for frozen-backbone aux-head runs.
        if not (self.freeze_backbone and self.has_aux_heads):
            val_snr_i = self._try_get_avg_metric_at_epoch('val/snr_i')
            val_si_snr_i = self._try_get_avg_metric_at_epoch('val/si_snr_i')
            print(f'Val SNRi: {val_snr_i:.02f}dB')
            print(f'Val SI-SDRi: {val_si_snr_i:.02f}dB')
        else:
            print('(Frozen backbone: SNRi/SI-SDRi omitted; use audio parity eval if needed.)')


        def log_audio(run, key, samples, sr):
            columns = ['mixture', 'target', 'output']
            wandb_samples = []
            for i, sample in enumerate(samples):
                # TODO: Save spectrograms as well
                for k in columns:
                    # soundfile/wandb.Audio reject float16 (common under AMP).
                    audio = sample[k].detach().permute(1, 0).float().cpu().numpy()
                    wandb_samples.append(wandb.Audio(
                        audio, sample_rate=sr, caption=f'{i}/{k}'))
            run.log({key: wandb_samples}, commit=False, step=self.epoch + 1)

        # Log stuff on wandb
        wandb_run.log({'lr-Adam': self.get_current_lr()}, commit=False, step=self.epoch + 1)

        for metric in self.metric_values[self.epoch]:
            wandb_run.log({metric: self.get_avg_metric_at_epoch(metric)}, commit=False, step=self.epoch + 1)
        
        for statistic in self.statistics:
            if not self.statistics[statistic]['logged']:
                data = self.statistics[statistic]['data']
                reduction = self.statistics[statistic]['reduction']
                if reduction == 'mean':
                    val = mean(data)
                elif reduction == 'sum':
                    val = sum(data)
                elif reduction == 'histogram':
                    data = [[d] for d in data]
                    table = wandb.Table(data=data, columns=[statistic])
                    val = wandb.plot.histogram(table, statistic, title=statistic)
                else:
                    assert 0, f"Unknown reduction {reduction}."
                wandb_run.log({statistic: val}, commit=False)
                self.statistics[statistic]['logged'] = True
        
        for spk_num in self.train_samples:
            log_audio(wandb_run, f"train/audio_samples_{spk_num}spk", self.train_samples[spk_num], sr=self.sr)
        self.train_samples.clear()

        for spk_num in self.val_samples:
            log_audio(wandb_run, f"val/audio_samples_{spk_num}spk", self.val_samples[spk_num], sr=self.sr)
        self.val_samples.clear()

        wandb_run.log({'epoch': self.epoch}, commit=True, step=self.epoch + 1)
        
        if self.scheduler is not None:
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                # Get last metric
                self.scheduler.step(monitor_metric_last)
            else:
                self.scheduler.step()

        self.epoch += 1

    def log_statistic(self, name, value, reduction='mean'):
        if name not in self.statistics:
            self.statistics[name] = dict(logged=False, data=[], reduction=reduction)
        
        self.statistics[name]['data'].append(value)

    def log_metric(self, name, value, batch_size=1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True):
        """
        Logs a metric
        value must be the AVERAGE value across the batch
        Must provide batch size for accurate average computation
        """
        
        epoch_str = self.epoch
        if epoch_str not in self.metric_values:
            self.metric_values[epoch_str] = {}

        if (name not in self.metric_values[epoch_str]):
            self.metric_values[epoch_str][name] = dict(step=None, epoch=None)
        
        if type(value) == torch.Tensor:
            value = value.item()

        if on_step:            
            if self.metric_values[epoch_str][name]['step'] is None:
                self.metric_values[epoch_str][name]['step'] = []
            
            self.metric_values[epoch_str][name]['step'].append(value)
        
        if on_epoch:
            if self.metric_values[epoch_str][name]['epoch'] is None:
                self.metric_values[epoch_str][name]['epoch'] = 0
                self.metric_values[epoch_str][name]['num_elements'] = 0
            
            self.metric_values[epoch_str][name]['epoch'] += (value * batch_size)
            self.metric_values[epoch_str][name]['num_elements'] += batch_size

    def _step(self, batch, batch_idx, step='train'):
        inputs, targets = batch
        batch_size = inputs['mixture'].shape[0]
       
        #print(targets['num_target_speakers'])
        # Forward pass
        outputs = self.model(inputs)
        #print(targets.keys()) 

        mix = inputs['mixture'][:, 0:1].clone() # Take first channel in mixture as reference
        est = outputs['output'].clone()
        
        gt = targets['target'].clone()
        n_speakers = targets['num_target_speakers'].clone()
        n_far_speakers = targets['num_interfering_speakers'].clone()
        num_noises = targets['num_noises'].clone()

        # Compute base audio loss (always logged; only optimized when backbone is trainable).
        audio_loss = self.loss_fn(est=est, gt=gt).mean()
        loss = audio_loss
        count_loss = est.new_tensor(0.0)
        count_acc = est.new_tensor(0.0)
        spk_distance_loss = est.new_tensor(0.0)
        spk_distance_mae = est.new_tensor(0.0)
        spk_activity_loss = est.new_tensor(0.0)
        steered_loss = est.new_tensor(0.0)

        if self.count_distance_head_enabled:
            required_keys = ("speaker_count_logits", "spk_distance_m")
            for key in required_keys:
                if key not in outputs:
                    raise RuntimeError(f"Speaker head is enabled but model outputs miss {key}.")
            for key in ("spk_distance_m", "spk_valid", "spk_ref"):
                if key not in targets:
                    raise RuntimeError(f"Speaker head is enabled but targets miss {key}.")

            pred_count_logits = outputs["speaker_count_logits"]
            pred_spk_distance = outputs["spk_distance_m"]
            pred_spk_logit = outputs.get("spk_active_logit")
            if pred_spk_logit is None:
                pred_spk_valid = outputs.get("spk_active")
                if pred_spk_valid is None:
                    raise RuntimeError("Speaker head requires spk_active_logit or spk_active output.")
                pred_spk_valid = pred_spk_valid.clamp(1e-6, 1.0 - 1e-6)
                pred_spk_logit = torch.log(pred_spk_valid) - torch.log(1.0 - pred_spk_valid)

            num_frames = pred_count_logits.shape[1]
            num_slots = pred_spk_distance.shape[2]
            gt_count = targets.get("num_target_speakers_clamped", torch.clamp(n_speakers, max=2)).long().to(pred_count_logits.device)
            gt_count = gt_count.clamp_min(0).clamp_max(pred_count_logits.shape[-1] - 1)
            gt_count_frames = gt_count.unsqueeze(1).expand(-1, num_frames)
            count_loss = F.cross_entropy(
                pred_count_logits.reshape(-1, pred_count_logits.shape[-1]),
                gt_count_frames.reshape(-1),
                reduction='mean',
            )
            pred_count = pred_count_logits.argmax(dim=-1)
            count_acc = (pred_count == gt_count_frames).float().mean()

            gt_spk_distance = targets["spk_distance_m"].float().to(pred_spk_distance.device)
            gt_spk_valid = targets["spk_valid"].float().to(pred_spk_distance.device)
            gt_spk_ref = targets["spk_ref"].float().to(pred_spk_distance.device)
            if gt_spk_distance.shape[1] != num_slots:
                raise RuntimeError(
                    f"spk_distance_m slots ({gt_spk_distance.shape[1]}) != model slots ({num_slots})"
                )
            if gt_spk_valid.shape[1] != num_slots:
                raise RuntimeError(
                    f"spk_valid slots ({gt_spk_valid.shape[1]}) != model slots ({num_slots})"
                )
            if gt_spk_ref.shape[1] != num_slots:
                raise RuntimeError(
                    f"spk_ref slots ({gt_spk_ref.shape[1]}) != model slots ({num_slots})"
                )

            slot_activity = self._compute_slot_activity_mask(gt_spk_ref, gt_spk_valid, num_frames).to(pred_spk_distance.device)
            distance_mask = slot_activity
            gt_spk_distance_frames = gt_spk_distance.unsqueeze(1).expand(-1, num_frames, -1)
            if distance_mask.sum() > 0:
                # Optimisation loss (SmoothL1) — logged separately from MAE.
                per_frame_dist = F.smooth_l1_loss(
                    pred_spk_distance,
                    gt_spk_distance_frames,
                    reduction='none',
                )
                spk_distance_loss = (per_frame_dist * distance_mask).sum() / distance_mask.sum().clamp_min(1.0)
                # Reporting metric vs GT slot distances (metres), same supervised frames.
                per_frame_mae = (pred_spk_distance - gt_spk_distance_frames).abs()
                spk_distance_mae = (per_frame_mae * distance_mask).sum() / distance_mask.sum().clamp_min(1.0)

            spk_activity_loss = F.binary_cross_entropy_with_logits(
                pred_spk_logit,
                slot_activity,
                reduction='mean',
            )

            if self.freeze_backbone:
                loss = (
                    self.count_loss_weight * count_loss
                    + self.spk_distance_loss_weight * spk_distance_loss
                    + self.spk_activity_loss_weight * spk_activity_loss
                )
            else:
                loss = (
                    audio_loss
                    + self.count_loss_weight * count_loss
                    + self.spk_distance_loss_weight * spk_distance_loss
                    + self.spk_activity_loss_weight * spk_activity_loss
                )

        if self.steered_head_enabled:
            if "steered_output" not in outputs:
                raise RuntimeError("Steered head enabled but model outputs miss steered_output.")
            if "target_steered" not in targets:
                raise RuntimeError("Steered head enabled but targets miss target_steered.")
            z_hat = outputs["steered_output"]
            z_gt = targets["target_steered"].to(z_hat.device)
            # Align lengths (pad path / ISTFT edge).
            t = min(z_hat.shape[-1], z_gt.shape[-1], est.shape[-1])
            z_hat = z_hat[..., :t]
            z_gt = z_gt[..., :t]
            if z_hat.ndim == 2:
                z_hat = z_hat.unsqueeze(1)
            if z_gt.ndim == 2:
                z_gt = z_gt.unsqueeze(1)
            steered_loss = self.loss_fn(est=z_hat, gt=z_gt).mean()

            # Look vs no-look SI-SDR delta (no-look = hard bubble est).
            # Maximize delta up to look_delta_cap_db (default 6 dB).
            look_delta_loss = est.new_tensor(0.0)
            look_delta_mean = est.new_tensor(0.0)
            if "spk_ref" in targets and "spk_angle_deg" in targets and "spk_valid" in targets:
                spk_ref = targets["spk_ref"].float().to(z_hat.device)
                spk_ang = targets["spk_angle_deg"].float().to(z_hat.device)
                spk_valid = targets["spk_valid"].float().to(z_hat.device)
                if "theta0_deg" in inputs:
                    th_deg = inputs["theta0_deg"].float().to(z_hat.device)
                else:
                    th_deg = inputs.get("theta0_rad", est.new_zeros(batch_size)) * (180.0 / np.pi)
                    th_deg = th_deg.float().reshape(-1)
                if "looked_spk_idx" in targets:
                    looked = targets["looked_spk_idx"].long().to(z_hat.device).reshape(-1)
                else:
                    looked = self._looked_spk_indices(th_deg, spk_ang, spk_valid)

                z_mono = z_hat[:, 0, :] if z_hat.ndim == 3 else z_hat
                b_mono = est[:, 0, :t] if est.ndim == 3 else est[..., :t]
                cap = float(self.look_delta_cap_db)
                deltas = []
                for i in range(batch_size):
                    k = int(looked[i].item()) if looked.numel() > i else -1
                    if k < 0 or k >= spk_ref.shape[1]:
                        continue
                    if float(spk_valid[i, k].item()) <= 0.5:
                        continue
                    ref_k = spk_ref[i, k, :t]
                    if float(ref_k.abs().max()) <= 0:
                        continue
                    sisdr_look = self._si_sdr_db_torch(z_mono[i : i + 1], ref_k.unsqueeze(0))
                    sisdr_noolook = self._si_sdr_db_torch(b_mono[i : i + 1].detach(), ref_k.unsqueeze(0))
                    raw_delta = sisdr_look - sisdr_noolook
                    # Loss: push delta up to cap (0 when delta >= cap).
                    deltas.append(torch.relu(raw_delta.new_tensor(cap) - raw_delta))
                    look_delta_mean = look_delta_mean + torch.clamp(raw_delta, max=cap).detach()
                if deltas:
                    look_delta_loss = torch.stack(deltas).mean()
                    look_delta_mean = look_delta_mean / len(deltas)

            if self.freeze_backbone and not self.steered_train_count_heads:
                loss = (
                    self.steered_loss_weight * steered_loss
                    + self.look_delta_loss_weight * look_delta_loss
                )
            elif self.freeze_backbone:
                loss = (
                    loss
                    + self.steered_loss_weight * steered_loss
                    + self.look_delta_loss_weight * look_delta_loss
                )
            else:
                loss = (
                    loss
                    + self.steered_loss_weight * steered_loss
                    + self.look_delta_loss_weight * look_delta_loss
                )
        else:
            look_delta_loss = est.new_tensor(0.0)
            look_delta_mean = est.new_tensor(0.0)

        # Edge / passband aux (multi-level × mild RT60, or legacy single-condition ripple).
        edge_aux_loss = est.new_tensor(0.0)
        if self._edge_aux_enabled and step == "train" and not self.freeze_backbone:
            self._passband_train_step += 1
            if self._passband_train_step % max(1, self.passband_ripple_every_n) == 0:
                if self._use_multi_condition_edge:
                    edge_aux_loss = self._compute_edge_aux_loss()
                else:
                    edge_aux_loss = self.passband_ripple_w * self._compute_passband_ripple_loss()
                    self._last_edge_aux_metrics = {
                        "passband_ripple_db": float(edge_aux_loss.detach().item()),
                    }
                loss = loss + edge_aux_loss

        # Elevation aux (shrink ≈−20° vertical notch on in-bubble keep).
        elev_aux_loss = est.new_tensor(0.0)
        if self._elev_aux_enabled and step == "train" and not self.freeze_backbone:
            self._elev_train_step += 1
            if self._elev_train_step % max(1, self.elev_aux_every_n) == 0:
                elev_aux_loss = self._compute_elev_aux_loss()
                loss = loss + elev_aux_loss

        # Positive keep-energy hinge (in-bubble samples must not collapse to silence).
        pos_keep_hinge = est.new_tensor(0.0)
        pos_keep_est_db = est.new_tensor(0.0)
        if self.pos_keep_enabled and not self.freeze_backbone:
            pos_keep_hinge, pos_keep_est_db = self._compute_pos_keep_hinge(est, gt, n_speakers)
            if step == "train":
                loss = loss + self.pos_keep_w * pos_keep_hinge

        est_detached = est.detach().clone()
        
        
        with torch.no_grad():
            # Log loss
            self.log_metric(f'{step}/loss', loss.item(), batch_size=batch_size, on_step=(step == 'train'), on_epoch=True, prog_bar=True, sync_dist=True)
            self.log_metric(f'{step}/audio_loss', audio_loss.item(), batch_size=batch_size, on_step=(step == 'train'), on_epoch=True, prog_bar=False, sync_dist=True)
            if self.pos_keep_enabled and not self.freeze_backbone:
                self.log_metric(
                    f'{step}/pos_keep_db',
                    float(pos_keep_hinge.detach().item()),
                    batch_size=batch_size,
                    on_step=(step == 'train'),
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log_metric(
                    f'{step}/pos_keep_est_db',
                    float(pos_keep_est_db.detach().item()),
                    batch_size=batch_size,
                    on_step=(step == 'train'),
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
            if self.count_distance_head_enabled:
                self.log_metric(f'{step}/count_loss', count_loss.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log_metric(f'{step}/count_acc', count_acc.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log_metric(f'{step}/spk_distance_loss', spk_distance_loss.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log_metric(f'{step}/spk_distance_mae', spk_distance_mae.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log_metric(f'{step}/spk_activity_loss', spk_activity_loss.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            if self.steered_head_enabled:
                self.log_metric(f'{step}/steered_loss', steered_loss.item(), batch_size=batch_size, on_step=(step == 'train'), on_epoch=True, prog_bar=True, sync_dist=True)
                if look_delta_loss.item() != 0.0 or look_delta_mean.item() != 0.0:
                    self.log_metric(
                        f'{step}/look_delta_loss',
                        look_delta_loss.item(),
                        batch_size=batch_size,
                        on_step=(step == 'train'),
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                    )
                    self.log_metric(
                        f'{step}/look_delta_db',
                        look_delta_mean.item(),
                        batch_size=batch_size,
                        on_step=(step == 'train'),
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                    )
                if "oracle_look_delta_db" in targets:
                    od = targets["oracle_look_delta_db"].float().mean().item()
                    self.log_metric(
                        f'{step}/oracle_look_delta_db',
                        od,
                        batch_size=batch_size,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
                # θ0 sweep (30° grid): look vs bubble SI-SDR delta per direction.
                do_sweep = (step == "val") or (
                    step == "train"
                    and (self._steered_train_step % max(1, self.theta0_sweep_every_n) == 0)
                )
                if step == "train":
                    self._steered_train_step += 1
                if do_sweep:
                    self._theta0_sweep_look_deltas(inputs, targets, est_detached, step, batch_size)
            if self._edge_aux_enabled and step == "train" and edge_aux_loss.item() != 0.0:
                self.log_metric(
                    f'{step}/edge_aux_loss',
                    edge_aux_loss.item(),
                    batch_size=1,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
                for key, value in self._last_edge_aux_metrics.items():
                    self.log_metric(
                        f'{step}/{key}',
                        float(value),
                        batch_size=1,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=(key in ("suppression_m", "edge_drop_db", "passband_ripple_db")),
                        sync_dist=True,
                    )
            if self._elev_aux_enabled and step == "train" and elev_aux_loss.item() != 0.0:
                self.log_metric(
                    f'{step}/elev_aux_loss',
                    elev_aux_loss.item(),
                    batch_size=1,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
                for key, value in self._last_elev_aux_metrics.items():
                    self.log_metric(
                        f'{step}/{key}',
                        float(value),
                        batch_size=1,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=(key in ("elev_notch_drop_db", "elev_ripple_db")),
                        sync_dist=True,
                    )

            # Log metrics
            for metric in self.metrics:
                if step == "train" and (metric.name == "PESQ" or metric.name == "STOI"):
                    continue
                metric_val = metric(est=est_detached, gt=gt, mix=mix)
                for i in range(batch_size):
                    #print(inputs['sample_dir'][i], batch_idx)
                    if n_speakers[i] > 0:
                        assert torch.abs(gt[i]).max() > 0, "Expected gt > 0"
                        #if metric.name == 'si_sdr_i' and metric_val[i] < -10:
                        #    print(metric_val[i].item(), inputs['sample_dir'][i], batch_idx)
                        val = metric_val[i].item()
                        self.log_metric(f'{step}/{metric.name}', val, batch_size=1,
                                on_step=False, on_epoch=True, prog_bar=True,
                                sync_dist=True)
            
            # Log metrics for zero speakers
            for i in range(batch_size):
                if n_speakers[i] == 0:
                    decay = compute_decay(est_detached[i].unsqueeze(0), mix[i].unsqueeze(0)).item()
                    self.log_metric(f'{step}/decay', decay, batch_size=1,
                            on_step=False, on_epoch=True, sync_dist=True)
            
            # Log metrics for non-zero speakers
            for spk_num in range(1, max(n_speakers) + 1):
                for metric in self.metrics:
                    if metric.name == 'si_sdr_i':
                        for i in range(batch_size):
                            if n_speakers[i] == spk_num:
                                si_sdri_i_val = metric(est=est_detached[i].unsqueeze(0), gt=gt[i].unsqueeze(0), mix=mix[i].unsqueeze(0))
                                self.log_metric(f'{step}/{metric.name}_{spk_num}spk', si_sdri_i_val.item(), batch_size=1,
                                        on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            # Log input snr
            if (f'stat/{step}_input_snr' not in self.statistics) or (not self.statistics[f'stat/{step}_input_snr']['logged']):
                for i in range(batch_size):
                    if n_speakers[i] > 0:
                        snr_val = self.snr_metric(est=mix[i].unsqueeze(0), gt=gt[i].unsqueeze(0), mix=mix[i].unsqueeze(0))
                        
                        self.log_statistic(f'stat/{step}_input_snr', snr_val.item(), reduction='histogram')
                    
                    self.log_statistic(f'stat/{step}_num_tgt_speakers', n_speakers[i].item(), reduction='histogram')
                    self.log_statistic(f'stat/{step}_num_far_speakers', n_far_speakers[i].item(), reduction='histogram')
                    self.log_statistic(f'stat/{step}_num_noises', num_noises[i].item(), reduction='histogram')
        
        # Create collection of things to show in a sample on wandb
        sample = {
            'mixture': mix,
            'output': est_detached,
            'target': gt,
            'n_tgt_speakers': n_speakers,
        }

        return loss, sample

    def train(self):
        self.model.train()
        self.mode = 'train'
    
    def eval(self):
        self.model.eval()
        self.mode = 'val'

    def training_step(self, batch, batch_idx):
        loss, sample = self._step(batch, batch_idx, step='train')

        n_speakers = sample['n_tgt_speakers']
        for i in range(n_speakers.shape[0]):
            spk_num = n_speakers[i].item()
            if spk_num not in self.train_samples:
                self.train_samples[spk_num] = []
            
            if len(self.train_samples[spk_num]) < 3:
                sample_at_batch = {}
                for k in sample:
                    sample_at_batch[k] = sample[k][i]
                self.train_samples[spk_num].append(sample_at_batch)
        
        return loss, n_speakers.shape[0]

    def validation_step(self, batch, batch_idx):
        loss, sample = self._step(batch, batch_idx, step='val')
        
        n_speakers = sample['n_tgt_speakers']
        for i in range(n_speakers.shape[0]):
            spk_num = n_speakers[i].item()
            if spk_num not in self.val_samples:
                self.val_samples[spk_num] = []
            
            if len(self.val_samples[spk_num]) < self.samples_per_speaker_number:
                sample_at_batch = {}
                for k in sample:
                    sample_at_batch[k] = sample[k][i]
                self.val_samples[spk_num].append(sample_at_batch)

        # if not self.trainer.sanity_checking:
        #     self.input_snr_calculated = True
        
        return loss, n_speakers.shape[0]
    
    def reset_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def backprop(self):
        #print("BACKPROP")
        #print(self.grad_clip)
        # Gradient clipping
        if self.grad_clip is not None:
            #print("Clipping grad norm")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip) 
        
        self.optimizer.step()
    
    def configure_optimizers(self):        
        if self.scheduler is not None:
            # For reduce LR on plateau, we need to provide more information
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler_cfg = {
                    "scheduler": self.scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": self.monitor,
                    "strict": False
                }
            else:
                scheduler_cfg = self.scheduler
            return [self.optimizer], [scheduler_cfg]
        else:
            return self.optimizer

    def init_scheduler(self, scheduler, scheduler_params):
        if scheduler is not None:
            if scheduler == 'sequential':
                schedulers = []
                milestones = []
                for scheduler_param in scheduler_params:
                    sched = utils.import_attr(scheduler_param['name'])(self.optimizer, **scheduler_param['params'])
                    schedulers.append(sched)
                    milestones.append(scheduler_param['epochs'])

                # Cumulative sum for milestones
                for i in range(1, len(milestones)):
                    milestones[i] = milestones[i-1] + milestones[i]

                # Remove last milestone as it is implied by num epochs
                milestones.pop()

                scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers, milestones)
            else:
                scheduler = utils.import_attr(scheduler)(self.optimizer, **scheduler_params)

        return scheduler

# class DistanceBasedLogger(Callback):
#     def _log_audio(self, logger, key, samples, sr):
#         columns = ['mixture', 'target', 'output']
#         wandb_samples = []
#         for i, sample in enumerate(samples):
#             # TODO: Save spectrograms as well
#             for k in columns:
#                 wandb_samples.append(wandb.Audio(
#                     sample[k].permute(1, 0).cpu().numpy(),
#                     sample_rate=sr, caption=f'{i}/{k}'))
#         logger.experiment.log({key: wandb_samples})

#     def on_train_epoch_end(self, trainer, pl_module):
#         for spk_num in pl_module.train_samples:
#             self._log_audio(
#                 trainer.logger, f"train/audio_samples_{spk_num}spk", pl_module.train_samples[spk_num],
#                 sr=pl_module.sr)
#         pl_module.train_samples.clear()

#     def on_validation_end(self, trainer, pl_module):
#         for spk_num in pl_module.val_samples:
#             self._log_audio(
#                 trainer.logger, f"val/audio_samples_{spk_num}spk", pl_module.val_samples[spk_num],
#                 sr=pl_module.sr)
#         pl_module.val_samples.clear()
