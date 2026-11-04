# Lean syn+real split (EP35-style finetune from EP24)

**Date:** 2026-11-04  
**Active config:** [`real_experiments/optim_finetune_ep24_synreal_lean_passkeep_nw10.json`](../real_experiments/optim_finetune_ep24_synreal_lean_passkeep_nw10.json)  
**Active iteration note:** [`training_iteration_ep24_synreal_lean_passkeep_nw10.md`](training_iteration_ep24_synreal_lean_passkeep_nw10.md)  
**Failed predecessor config:** [`real_experiments/optim_finetune_ep24_ep35style_synreal_lean.json`](../real_experiments/optim_finetune_ep24_ep35style_synreal_lean.json) — see [`training_iteration_ep24_ep35style_synreal_lean.md`](training_iteration_ep24_ep35style_synreal_lean.md)  
**Leveling baseline:** [`dataset_synreal_mic0m25.md`](dataset_synreal_mic0m25.md) (mic0 → −25 dBFS)

The lean **split** (caps below) is unchanged. The failed lean run used EP35 suppress (`neg_weight: 60`, no passband aux). The active fix keeps this split and changes loss/aux only.

## Why this split

| Prior recipe | Train samples / epoch | Mix |
|--------------|----------------------:|-----|
| EP35 suppress (syn only) | ~13 000 | `syn_1_5m` 10k + `syn_libri_1_5m_rt60` 3k |
| Contingency / mic0m25 edge_aux | ~30 000 | syn 10k + real WHAM 20k |
| **This lean split** | **~10 000** | **real 7k + syn 2k + libri 1k** |

Goals:

1. Match EP35’s smaller epoch budget (faster iteration on 6 GB).
2. Prefer **real** HumanMultiTargetWHAM (1.5 m bubble labels via `dis_threshold`) over syn.
3. Keep a thin syn + Libri-RT60 slice so geometry / mild wet diversity from the EP35 program are not dropped entirely.

## Train (`max_samples`)

| Corpus | Path | Cap | Share | Role |
|--------|------|----:|------:|------|
| Real WHAM omni | `datasets/HumanMultiTargetWHAM/omni/train` | **7000** | 70% | Real rooms + WHAM noise; in-bubble via `dis_threshold=1.5` |
| Syn 1.5 m | `datasets/syn_1_5m/syn_1_5m/train` | **2000** | 20% | Array geometry / bubble shape |
| Syn Libri RT60 | `datasets/syn_libri_1_5m_rt60/train` | **1000** | 10% | Mild wet syn (shrunk EP35 component) |
| **Total** | | **10000** | 100% | |

Disk pools are larger (real train 20k, syn 10k, libri 3k). Caps only limit draws per epoch.

## Val (`max_samples`)

| Corpus | Path | Cap |
|--------|------|----:|
| Real WHAM omni | `datasets/HumanMultiTargetWHAM/omni/val` | **800** |
| Syn 1.5 m | `datasets/syn_1_5m/syn_1_5m/val` | **300** |
| Syn Libri RT60 | `datasets/syn_libri_1_5m_rt60/val` | **100** |
| **Total** | | **1200** |

## Leveling (required)

Syn and real still differ by ~20 dB without alignment. Reuse **mic0 → −25 dBFS** trees and loader `target_mic0_dbfs: -25.0`.

- Syn 1.5 m + HumanMultiTargetWHAM: already covered by [`dataset_synreal_mic0m25.md`](dataset_synreal_mic0m25.md).
- Syn Libri RT60: enforce at train/val time via `target_mic0_dbfs` even if offline `level_norm` stamps are missing. Optional offline pass:

```powershell
python scripts/normalize_dataset_mic0_dbfs.py `
  --target-dbfs -25 `
  --roots `
    datasets/syn_libri_1_5m_rt60/train `
    datasets/syn_libri_1_5m_rt60/val
```

Bubble threshold **1.5 m**. Mic order **mic00…mic05**. SR **24 kHz**.

## Related (not this split)

- Large 30k syn+real: [`dataset_synreal_mic0m25.md`](dataset_synreal_mic0m25.md)
- Sacred EP24 promote: [`training_iteration_ep24_mic0m25_promote.md`](training_iteration_ep24_mic0m25_promote.md)
