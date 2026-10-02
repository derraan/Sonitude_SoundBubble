# Sonitude_SoundBubble

Private Sonitude archive of Sound Bubble work: Raspberry Pi optim model, **count–distance head**, training/eval configs, ONNX exports, and edge/RPi realtime runtime.

## What’s included

- Latest model code (`src/models/tfgridnet_realtime_clean_optim`) with `count_distance_head`
- **Promoted 1.5 m model:** `runs/optim_pretrain_sonitude_dist_count_1_5m` (see `README_COUNT_DISTANCE.md`)
- Training configs under `real_experiments/` (promoted, suppress finetune, directional)
- Distance→dBFS validation + **passband ripple** (`src/validation/distance_sweep_dbfs.py`)
- Edge runtime: ONNX zoo, streamer/pipeline, RPi5 scripts (`edge/rpi5`), in-bubble mixer helpers
- Docs: `README_COUNT_DISTANCE.md`, `GUARDRAILS.md`, `edge/README_EDGE.md`, etc.

## What’s not included

Raw audio datasets (~100GB+). See `datasets/README.md` for expected local paths.

## Quick train (promoted 1.5 m)

```powershell
python -m src.train_pt --config real_experiments\optim_pretrain_sonitude_dist_count_1_5m.json --run_dir runs\optim_pretrain_sonitude_dist_count_1_5m --no_wandb
```

## Eval

```powershell
python -m src.eval_count_distance `
  datasets\syn_test\syn_1_5m\test `
  runs\optim_pretrain_1_5m `
  runs\optim_pretrain_sonitude_dist_count_1_5m `
  runs\speaker_eval_1_5m `
  --distance_threshold 1.5 --use_cuda
```

## Guardrails

Do not overwrite live `runs/**/checkpoints/*.pt` or datasets without explicit approval. See `GUARDRAILS.md`.
