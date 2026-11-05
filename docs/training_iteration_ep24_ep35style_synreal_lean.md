# Training iteration: EP35-style program + lean syn+real (from EP24) — **FAILED**

**Date:** 2026-11-04  
**Status:** **FAILED / superseded** — dry passband collapsed by epoch 25. Do not resume.  
**Superseded by:** [`training_iteration_ep24_synreal_lean_passkeep_nw10.md`](training_iteration_ep24_synreal_lean_passkeep_nw10.md)  
**Parent ckpt:** `runs/optim_finetune_promoted_synreal_1_5m_edge_aux_mic0m25_from_ep23/checkpoints/best.pt` (epoch **24**, sacred)  
**Config:** [`real_experiments/optim_finetune_ep24_ep35style_synreal_lean.json`](../real_experiments/optim_finetune_ep24_ep35style_synreal_lean.json) (kept for archaeology; do not relaunch as-is)  
**Run dir:** `runs/optim_finetune_ep24_ep35style_synreal_lean_from_ep24`  
**Split doc:** [`dataset_synreal_lean_ep35style.md`](dataset_synreal_lean_ep35style.md)

> **Correction (2026-11-04):** the main cause was a loader bug, not just `neg_weight`. `max_samples` took the first N sorted dirs, so the lean split got 2000 syn_1_5m clips that were **100% empty-GT negatives** (and a 100%-negative syn val slice). v1 also dropped EP35's real `passband_ripple` aux. Both are fixed; see [`training_iteration_ep24_ep35prog_synreal_lean_v2.md`](training_iteration_ep24_ep35prog_synreal_lean_v2.md).

## Failure evidence (epoch 25 sweep)

`val_distance_sweep/epoch_025.csv`, dry RT60=0.2: 0.8 / 1.0 / 1.2 m all ≈ **−98…−99 dBFS** (gate target ≈ −26 ± 3). Far kill intact. Wet kept only near-field scraps then died.

## Root cause (summary)

- SNRLP `neg_weight: 60` × LogPower silence floor (~−30 dB) → ~−1800 per empty-GT sample; dominates positive SNR on real-heavy lean batches.
- No EP24 `passband_ripple` aux (removed to mimic EP35 syn-only suppress).

Full write-up and fix recipe: [`training_iteration_ep24_synreal_lean_passkeep_nw10.md`](training_iteration_ep24_synreal_lean_passkeep_nw10.md).

## Original recipe (do not relaunch)

| Item | Value |
|------|--------|
| Loss | SNRLP, `neg_weight: 60` |
| Heads | count / distance / activity; backbone **unfrozen** |
| Aux | **no** `edge_aux`, **no** `pos_keep`, **no** elev, **AMP off** |
| Bubble | `dis_threshold: 1.5` |
| Level | `target_mic0_dbfs: -25` |
| Train / epoch | **10k** (real 7k + syn 2k + libri RT60 1k) |
| Val | **1.2k** (800 / 300 / 100) |
| Sweep val | enabled, `every_n_epochs: 5` (monitor only) |
| Epochs | 40 |

## Do not

- Resume this run’s `checkpoints/{best,last}.pt`
- Overwrite sacred EP24 / EP35 `_promoted` or zoo ONNX
- Relaunch this config without lowering `neg_weight` and restoring passband keep

## Launch (superseded — use passkeep_nw10 instead)

```powershell
cd C:\Users\darre\Sound_Bubble
$env:PYTHONPATH = "C:\Users\darre\Sound_Bubble"
python -u src/train_pt.py `
  --config real_experiments/optim_finetune_ep24_synreal_lean_passkeep_nw10.json `
  --run_dir runs/optim_finetune_ep24_synreal_lean_passkeep_nw10_from_ep24 `
  --no_wandb
```
