# Guardrails — do not overwrite training assets

Agent-enforced copy lives in [`.cursor/rules/no-overwrite-training-assets.mdc`](.cursor/rules/no-overwrite-training-assets.mdc) (always apply). Related: [`promoted-count-distance-model.mdc`](.cursor/rules/promoted-count-distance-model.mdc), [`train-val-active-paths.mdc`](.cursor/rules/train-val-active-paths.mdc).

## Short version

- **Never** force-overwrite `runs/**/checkpoints/*.pt`, `datasets/**`, or active train/val Python without an explicit user OK.
- **Never** `git checkout stash… -- runs/.../checkpoints/` onto a live run that already has weights.
- Recover into `runs/_recovery_<reason>/`, compare epoch/size, then copy only if the destination is missing/empty or the user confirmed.

## Protected runs (this tree)

| Run | Role |
|---|---|
| `runs/optim_pretrain_sonitude_dist_count_1_5m` | Promoted (best ep23 / last ep31) |
| `runs/optim_pretrain_sonitude_dist_count_directional_1_5m` | Directional steered (do not clobber promote) |
| `runs/count_distance_head_optim_1_5m_suppress_v1` | Legacy finetune source |
| `runs/archived_stopband_v2_ep46` | Head archive |
| `runs/_recovery_safe/` | Stash recovery mirror — leave intact |

## Resume overnight finetune

```powershell
cd C:\Users\darre\Sound_Bubble_optim_distance
python -m src.train_pt --config real_experiments\count_distance_head_optim_1_5m_suppress_v1.json --run_dir runs\count_distance_head_optim_1_5m_suppress_v1 --no_wandb
```
