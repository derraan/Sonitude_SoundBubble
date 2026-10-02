# In-bubble controllable mixing investigation

Scripts for the staged feasibility investigation layered **on top of**
`runs/optim_pretrain_sonitude_dist_count_1_5m`.

Directional improvement (Stages G–J) writes only to
`runs/optim_pretrain_sonitude_dist_count_directional_1_5m` (never renames the promote).

## Mix rule

```
y = b_hat + sum_k (g_k - 1) * s_hat_k
```

Default `g_k = 1` recovers the promoted bubble output exactly.

Steered blend (optional; `alpha=0` default):

```
y = (1 - alpha) * b_hat + alpha * z_hat
```

β gains (Stage I): `g_k = gains_from_beta(angles, theta0)` with frozen σ=0.2, ρ=8.0.

## Run order

```powershell
# Stages A–F (feasibility)
python -m scripts.inbubble_mixer.stage_a_oracle_mix --max-samples 40
python -m scripts.inbubble_mixer.stage_b_separation_probe --max-train 48 --max-val 16 --epochs 8
python -m scripts.inbubble_mixer.stage_c_doa_region --max-samples 80
python -m scripts.inbubble_mixer.stage_d_session_id --max-samples 40
python -m scripts.inbubble_mixer.stage_e_pi_cost
python -m scripts.inbubble_mixer.stage_f_mixer_smoke

# Stages G–J (DSENet β → directional run)
python -m scripts.inbubble_mixer.stage_g_beta_oracle --max-samples 40
python -m scripts.inbubble_mixer.stage_h_train_steered --steps 40
# Full overnight (optional):
# python -m src.train_pt --config real_experiments/optim_pretrain_sonitude_dist_count_directional_1_5m.json --run_dir runs/optim_pretrain_sonitude_dist_count_directional_1_5m --no_wandb
python -m scripts.inbubble_mixer.stage_i_stream_weights --max-samples 32
python -m scripts.inbubble_mixer.stage_j_pi_gate
python -m scripts.inbubble_mixer.aggregate_report
```

Dataset default: `C:\Users\darre\Sound_Bubble\datasets\syn_libri_1_5m_rt60` (samples with `mic00_voice*`).

## Realtime flag (Stage F / I)

```bash
python edge/inference_pipeline.py --model ... --inbubble-mixer --speaker-gains 1,1
```

Optional region control: `--region-center-deg -60 --stream-angles-deg -60,30`.

β look direction: `--beta-theta0-deg -60 --stream-angles-deg -60,30`.

Speaker streams are read from ONNX outputs `s0`/`s1` when present; otherwise zeros (pass-through).
