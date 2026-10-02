"""
Archived stopband joint-finetune checkpoint (epoch 46 / interrupted).

Source: runs/count_distance_head_optim_1_5m_stopband_v2/checkpoints/last.pt
Note: run-dir best.pt was epoch 1 (val/loss) and is NOT this archive.

Eval snapshot (1.5 m):
- syn_test SI-SDRi ~0.5 dB vs ep93 ~6.4 dB
- distance-dBFS sweep cutoff collapsed near ~1.0 m (over-suppression)

Intended reuse: count_distance_head weights only, paired with ep93 backbone
via init_ckpt + init_head_ckpt (see count_distance_head_optim_1_5m_recover_nw70.json).
"""
