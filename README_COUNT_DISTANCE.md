# Count–Distance Head (speaker # + per-slot distance)

Optional aux head on the Raspberry Pi optim Sound Bubble model that reports:

- inside-bubble **speaker count** classes `0 / 1 / 2+`
- up to **2 nearest-first distance slots** (metres) with per-slot activity

Audio separation behavior is unchanged when the head is enabled (frozen-backbone training).

## Promoted 1.5 m model

| Item | Path |
|------|------|
| **Default run** | `runs/optim_pretrain_sonitude_dist_count_1_5m` |
| Config | `real_experiments/optim_pretrain_sonitude_dist_count_1_5m.json` (also live `runs/.../config.json`) |
| Checkpoints | `best.pt` ≈ ep23 (val/loss); `last.pt` ≈ ep31 |
| Progress UI | `canvases/promoted-epoch-progress.canvas.tsx` + `epoch_progress/` |
| Guardrails | `GUARDRAILS.md`, `.cursor/rules/promoted-count-distance-model.mdc` |

Init / eval / export / demos should use this path unless another run is named. Do **not** promote experimental dirs (e.g. `suppress_v1`) by renaming — copy weights into the promoted slot.

Directional steered explore (separate run, never overwrites promote):

- Config: `real_experiments/optim_pretrain_sonitude_dist_count_directional_1_5m.json`
- Run dir: `runs/optim_pretrain_sonitude_dist_count_directional_1_5m`

## Scope

- Model family only: `src.models.tfgridnet_realtime_clean_optim`
- Canonical aux-head path: **`count_distance_head`** (`CountDistanceHead`)
- Legacy names `distance_head` / `speaker_head` are rejected; distance-only and count-only paths are removed
- Backbone-only pretrain configs (`raspberrypi_*`, `orangpi_*`) remain unchanged

## Radius checkpoint mapping

Each count–distance head run starts from the matching optim backbone checkpoint:

| Radius | Backbone `init_ckpt` | Config | Run dir |
|--------|----------------------|--------|---------|
| 1.0 m | `.../runs/optim_pretrain_1m/checkpoints/best.pt` | `real_experiments/count_distance_head_optim_1m.json` | `runs/count_distance_head_optim_1m` |
| 1.5 m | Prefer **promoted** `runs/optim_pretrain_sonitude_dist_count_1_5m/checkpoints/best.pt` | `real_experiments/optim_pretrain_sonitude_dist_count_1_5m.json` | `runs/optim_pretrain_sonitude_dist_count_1_5m` |
| 1.5 m (legacy head-only) | `.../runs/optim_pretrain_1_5m/checkpoints/best.pt` (epoch 93) | `real_experiments/count_distance_head_optim_1_5m.json` | `runs/count_distance_head_optim_1_5m` |
| 2.0 m | `.../runs/optim_pretrain_2m/checkpoints/best.pt` | `real_experiments/count_distance_head_optim_2m.json` | `runs/count_distance_head_optim_2m` |

Optional joint / suppress finetune (weight source only, not promoted): `real_experiments/count_distance_head_optim_1_5m_suppress_v1.json`.

Configs keep backbone `model_params` aligned to the pretrain runs (`D=16`, `B=3`, `H=64`, `directional=false`, `spectral_masking=false`, …).

## What the head does

- Tap point: after the last GridNet block (`[B,D,T,F]`), before deconv concat
- Dual causal EMA (`fast_tau_s` for count/activity, `slow_tau_s` for distance) with state `count_distance_head_buf`
- Update interval: one hop (`192` samples @ 24 kHz ≈ 8 ms); no extra look-ahead
- Outputs: `speaker_count_logits`, `spk_distance_m`, `spk_active_logit` / `spk_active`
- Head disabled by default preserves baseline buffers/outputs and headless ONNX layout

## Training losses / metrics

In `src/hl_modules/distance_based_hl_module.py` with `count_distance_head.enabled`:

- `w_count *` CE on count logits
- `w_activity *` BCE on per-slot activity
- `w_distance *` masked SmoothL1 on slot distances (active frames only)
- Logged/printed: `count_loss`, `count_acc`, `spk_distance_loss`, `spk_distance_mae`, `spk_activity_loss`
- With `freeze_backbone: true`, SNRi/SI-SDRi are omitted from the epoch summary

### Distance → dBFS sweep + passband ripple

When `distance_sweep_val.enabled` is true, each epoch runs a LibriSpeech distance×RT60 level sweep (`src/validation/distance_sweep_dbfs.py`) and logs:

| Metric | Meaning |
|--------|---------|
| `val/sweep_out_dbfs_*m` | Output level at fixed distances |
| `val/sweep_passband_minus_far_db` | Near (0.8 m) − far (3.0 m); higher = clearer bubble contrast |
| `val/sweep_edge_drop_1_5_to_2_0_db` | Edge sharpness at the bubble |
| `val/sweep_passband_ripple_0_2_db` | **Passband ripple (dB)** on RT60 **0.2 s** |

**Passband ripple definition:** peak−peak output dBFS for sources strictly inside the bubble (`d < bubble_m`) whose level is above −60 dBFS (active passband; excludes stopband→passband transition). Lower is better — less jittery / muffled perception near the 1.5 m threshold.

**Train loss** (optional, under `distance_sweep_val.passband_ripple`):

```json
"passband_ripple": {
  "w_passband_ripple": 0.05,
  "every_n_train_steps": 10,
  "rt60": 0.2
}
```

Caches RT60=0.2 level-matched mixes once, then every N train steps adds `w * ripple_db` to the total loss. Requires a **trainable backbone** (`freeze_backbone: false`); otherwise the aux is skipped and only the val metric logs.

CLI / plots: `scripts/distance_sweep_dbfs.py`, `scripts/plot_promoted_epoch_progress.py`. Progress canvas shows ripple as measurement cursors on the best-epoch RT60 overlay.

## Train

```powershell
cd C:\Users\darre\Sound_Bubble_optim_distance
# Promoted 1.5 m (preferred)
python -m src.train_pt --config real_experiments\optim_pretrain_sonitude_dist_count_1_5m.json --run_dir runs\optim_pretrain_sonitude_dist_count_1_5m --no_wandb
# Legacy radius configs
python -m src.train_pt --config real_experiments\count_distance_head_optim_1_5m.json --run_dir runs\count_distance_head_optim_1_5m --no_wandb
python -m src.train_pt --config real_experiments\count_distance_head_optim_1m.json   --run_dir runs\count_distance_head_optim_1m   --no_wandb
python -m src.train_pt --config real_experiments\count_distance_head_optim_2m.json   --run_dir runs\count_distance_head_optim_2m   --no_wandb
```

## Eval

```powershell
python -m src.eval_count_distance `
  C:/Users/darre/Sound_Bubble/datasets/syn_test/syn_1_5m/test `
  C:/Users/darre/Sound_Bubble/runs/optim_pretrain_1_5m `
  runs/optim_pretrain_sonitude_dist_count_1_5m `
  runs/count_distance_eval_1_5m `
  --distance_threshold 1.5 --use_cuda
```

Reports frame/clip count accuracy, slot distance MAE/RMSE/bias on active slots, false-active rates, and SI-SDRi/audio parity vs the frozen backbone.

## ONNX export / benchmark

```powershell
python edge/export_to_onnx.py --run-dir runs/optim_pretrain_sonitude_dist_count_1_5m --output edge/zoo/optim_count_distance_1_5m.onnx --opset 9 --frames-for-check 3 --batch-size 1
python edge/benchmark.py --model edge/zoo/optim_count_distance_1_5m.onnx --contract-path edge/zoo/optim_count_distance_1_5m.runtime.json --runs 200 --warmup 30 --num-threads 4
```

Headless (no aux head):

```powershell
python edge/export_to_onnx.py --run-dir C:/Users/darre/Sound_Bubble/runs/optim_pretrain_1_5m --output edge/zoo/optim_headless.onnx --opset 9 --frames-for-check 3 --batch-size 1
```

Untrained count–distance fixture exports require `--allow-untrained-count-distance-head` (tests only).

Streamer returns `speaker_count` / `speaker_count_status` and per-slot `speaker_slot_distance_m` / `speaker_slot_status` (`valid|inactive|unavailable`).

## Tests

- `tests/test_count_distance_head.py` — causality, reject stale `distance_head`, audio-path identity
- `tests/test_count_distance_data_loss.py` — labels/masks/losses and freeze-backbone trainable set
- `tests/test_count_distance_onnx.py` — export parity and streamer speaker status

## Limitations

- At most 2 inside-bubble slots; `2+` is a count class, not identity tracking
- Slots are nearest-first by GT distance during training, not stable speaker IDs over time
- Count/distance do not change separator suppression
- Existing backbone checkpoints do not include trained `count_distance_head.*` weights until you train them
