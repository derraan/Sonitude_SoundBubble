# Training iteration: EP35 program + lean syn+real, v2 (from EP24)

**Date:** 2026-11-04
**Status:** Config ready, smoke-tested (45 steps). Not launched.
**Config:** [`real_experiments/optim_finetune_ep24_ep35prog_synreal_lean_v2.json`](../real_experiments/optim_finetune_ep24_ep35prog_synreal_lean_v2.json)
**Run dir:** `runs/optim_finetune_ep24_ep35prog_synreal_lean_v2_from_ep24`
**Init:** sacred EP24 `runs/optim_finetune_promoted_synreal_1_5m_edge_aux_mic0m25_from_ep23/checkpoints/best.pt` (read-only)
**Replaces:** `optim_finetune_ep24_ep35style_synreal_lean` (v1, collapsed at ep25)

## What actually broke v1 (corrects the earlier root-cause note)

### 1. `max_samples` took the *first N* sorted dirs (loader bug, now fixed)

Generated corpora are stored grouped by label. `syn_1_5m/{train,val}` dirs `00000…03333` are **all** `n_in=0`, so the target is empty. The loader did `sorted(glob)[:max_samples]`:

| Split (lean caps) | Pool neg % | Neg % of what v1 actually got |
|---|---:|---:|
| syn_1_5m train, first 2000 | 33% | **100%** |
| syn_1_5m val, first 300 | 33% | **100%** |
| real WHAM train, first 7000 | 34% | 34% |
| syn_libri_rt60 train, first 1000 | 33% | 35% |

So v1 gave the model 2000 syn clips per epoch where every speaker is outside the bubble, and almost no syn in-bubble targets. Its val syn slice was also 100% negative, so `val/loss` rewarded silence. EP35 and EP24 used the **full** syn pool (10000/10000), so they never hit this.

**Fix:** `src/datasets/general_multisrc_dataset_with_perturbations.py`. A capped pool now draws a seeded random subset (`subset_seed`, default 0), sorted for determinism. With the lean caps every corpus and split comes out at 32–36% negatives. `"subset": "first"` brings back the old behaviour if you need to reproduce v1.

Also affected by the old behaviour: `real_experiments/count_distance_head_optim_1_5m.json` (syn train 3000/10000 and val 600/1000, both 100% negative).

### 2. v1 was not actually the EP35 program

The EP35 suppress config (`count_distance_head_optim_1_5m_suppress_v1.json`, git HEAD) **does** have a train-time aux: `passband_ripple` with `w_passband_ripple=0.05`, `every_n_train_steps=10`, `rt60=0.2`. v1 dropped it, based on the wrong belief that EP35 had no aux. v2 puts back the exact EP35 block.

### 3. `best.pt` competed against inherited parent epochs (fixed)

`init_ckpt` brings in the parent's `metric_values`. `on_epoch_end` then compared the new `val/loss` against EP35-lineage epochs scored on a different (syn-only) val set. Now only epochs trained in the current run compete (`_run_start_epoch`).

## Still open: real WHAM positives are silent from the start

Diagnostic: `debug/train_collapse/diag_loss_balance.py`, val clips, no augmentation:

| Ckpt | real val positives: est − gt | real/syn negatives: out dBFS |
|---|---:|---:|
| ep23 (EP35 parent, syn-only) | ≈ −64 dB | ≈ −94 |
| EP24 sacred (1 ep syn+real) | ≈ −66 dB | ≈ −95 |
| v1 ep25 (collapsed) | ≈ −70 dB | ≈ −99 |

Every checkpoint, including the EP35 lineage, already outputs near-silence on real in-bubble clips. That's why stored train SI-SDRi falls from about +2 dB (ep0–23, syn-only) to −13.9 dB at EP24. Under SNRLP, a negative at the floor is worth about −1790 (×60), while a perfect positive is worth roughly −10 to −20. Once the model is unsure about real clips, silence is close to loss-optimal.

The 45-step smoke log shows this too: batches with no negatives score about 0 dB (`debug/train_collapse/smoke_ep35prog_v2_45steps.log`).

If v2 still won't keep real in-bubble speech after the first sweep, the next lever is the loss balance (`neg_weight`, `pos_keep`). The split is no longer the problem. The `passkeep_nw10` config also benefits from the loader fix, because it uses the same lean caps.

## Launch (when asked)

```powershell
cd C:\Users\darre\Sound_Bubble
$env:PYTHONPATH = "C:\Users\darre\Sound_Bubble"
python -u src/train_pt.py `
  --config real_experiments/optim_finetune_ep24_ep35prog_synreal_lean_v2.json `
  --run_dir runs/optim_finetune_ep24_ep35prog_synreal_lean_v2_from_ep24 `
  --no_wandb
```

Gates are unchanged from v1: dry RT60=0.2 at 0.8 / 1.0 / 1.2 m ≈ −26 ± 3 dBFS, far kill intact, wet 0.8 m not collapsed relative to EP24. Promote only under new names.
