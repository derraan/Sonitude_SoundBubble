# In-bubble controllable mixing investigation

Scripts for the staged feasibility investigation layered **on top of**
`runs/optim_pretrain_sonitude_dist_count_1_5m`.

## Mix rule

```
y = b_hat + sum_k (g_k - 1) * s_hat_k
```

Default `g_k = 1` recovers the promoted bubble output exactly.

## Run order

```powershell
python -m scripts.inbubble_mixer.stage_a_oracle_mix --max-samples 40
python -m scripts.inbubble_mixer.stage_b_separation_probe --max-train 48 --max-val 16 --epochs 8
python -m scripts.inbubble_mixer.stage_c_doa_region --max-samples 80
python -m scripts.inbubble_mixer.stage_d_session_id --max-samples 40
python -m scripts.inbubble_mixer.stage_e_pi_cost
python -m scripts.inbubble_mixer.stage_f_mixer_smoke
python -m scripts.inbubble_mixer.aggregate_report
```

Dataset default: `C:\Users\darre\Sound_Bubble\datasets\syn_libri_1_5m_rt60` (samples with `mic00_voice*`).

## Realtime flag (Stage F)

```bash
python edge/inference_pipeline.py --model ... --inbubble-mixer --speaker-gains 1,1
```

Optional region control: `--region-center-deg -60 --stream-angles-deg -60,30`.
Speaker streams are read from ONNX outputs `s0`/`s1` when present; otherwise zeros (pass-through).
