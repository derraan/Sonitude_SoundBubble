# Training iteration: lean syn+real + passband keep, `neg_weight=10` (from EP24)

**Date:** 2026-11-04  
**Status:** Active fix for collapsed EP35-style lean run — launched 2026-11-04  
**Parent ckpt:** `runs/optim_finetune_promoted_synreal_1_5m_edge_aux_mic0m25_from_ep23/checkpoints/best.pt` (epoch **24**, sacred)  
**Config:** [`real_experiments/optim_finetune_ep24_synreal_lean_passkeep_nw10.json`](../real_experiments/optim_finetune_ep24_synreal_lean_passkeep_nw10.json)  
**Run dir:** `runs/optim_finetune_ep24_synreal_lean_passkeep_nw10_from_ep24`  
**Split doc:** [`dataset_synreal_lean_ep35style.md`](dataset_synreal_lean_ep35style.md) (unchanged lean mix)  

> **Correction (2026-11-04):** the main cause was a loader bug, not just `neg_weight`. `max_samples` took the first N sorted dirs, so the lean split got 2000 syn_1_5m clips that were **100% empty-GT negatives** (and a 100%-negative syn val slice). v1 also dropped EP35's real `passband_ripple` aux. Both are fixed; see [`training_iteration_ep24_ep35prog_synreal_lean_v2.md`](training_iteration_ep24_ep35prog_synreal_lean_v2.md).
**Failed predecessor:** [`training_iteration_ep24_ep35style_synreal_lean.md`](training_iteration_ep24_ep35style_synreal_lean.md)

## Why this run

The EP35-style lean finetune (`optim_finetune_ep24_ep35style_synreal_lean`) from sacred EP24 **failed at epoch 25**: dry Libri RT60=0.2 passband fully collapsed (~−99 dBFS at 0.8 / 1.0 / 1.2 m). Far kill stayed intact. Do **not** resume that run’s `last.pt` / `best.pt`.

### Root cause

1. **SNRLP `neg_weight: 60` + LogPower floor.** For empty-GT (no in-bubble) clips, `LogPowerLoss` is `10·log10(∑est² + 1e-3)`, which floors near **−30 dB**. Scaled by `neg_weight=60` → ≈ **−1800** per negative sample. Positive SNR terms are ~±20. Real-heavy lean batches (70% WHAM) often contain several empty-GT samples → train step losses cluster at ≈−440 / −875 / −1310 / −1750 (1–4 silence-floor negatives in a batch of 4). Gradient pressure is “output silence everywhere.”
2. **Dropped EP24 `passband_ripple` aux.** Lean copied EP35 suppress (syn-only historically) and removed train-time passband keep / edge / suppression. EP35 could tolerate `neg_weight=60` without that aux; this real-heavy split could not.

Sacred EP24 / zoo ONNX were not overwritten.

## Recipe (delta vs failed lean)

| Item | Failed lean (`ep35style`) | This fix |
|------|---------------------------|----------|
| Loss | SNRLP `neg_weight: 60` | SNRLP **`neg_weight: 10`** |
| Aux | none | **EP24 `passband_ripple`** restored (`w_passband_keep` / `w_edge` / `w_suppression` / light ripple; `every_n_train_steps: 40`) |
| Split | real 7k + syn 2k + libri 1k | **same** |
| Bubble / level | 1.5 m, mic0 −25 dBFS | same |
| Sweep val | every 5 epochs | same (monitor) |
| Init | EP24 sacred best | **same** (fresh run_dir; do not chain collapsed lean ckpt) |
| Epochs | 40 | 40 (halt early if dry passband collapses again) |

If dry keep is still weak after ≥1 sweep: try `neg_weight` 3–5 and/or raise `w_passband_keep` (new uniquely named config; do not overwrite this one).

## Do not

- Resume `runs/optim_finetune_ep24_ep35style_synreal_lean_from_ep24/checkpoints/{best,last}.pt`
- Resume elev / poskeep / fastmem failed runs
- Overwrite sacred EP24 / EP35 `_promoted` or zoo ONNX
- Treat this as interchangeable with mic0m25 `edge_aux` contingency naming

## Launch

```powershell
cd C:\Users\darre\Sound_Bubble
$env:PYTHONPATH = "C:\Users\darre\Sound_Bubble"
python -u src/train_pt.py `
  --config real_experiments/optim_finetune_ep24_synreal_lean_passkeep_nw10.json `
  --run_dir runs/optim_finetune_ep24_synreal_lean_passkeep_nw10_from_ep24 `
  --no_wandb
```

## Success gates (after ≥1 epoch / first sweep)

Dry Libri RT60=0.2 (PT): 0.8 / 1.0 / 1.2 ≈ −26 ± 3 dBFS; far kill intact; prefer edge nearer 1.5 m; wet 0.8 m keep should not collapse vs EP24 PT. Promote / zoo export only under **new** names after gates pass.
