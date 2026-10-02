# In-bubble controllable mixing - investigation report

Promoted model: `runs/optim_pretrain_sonitude_dist_count_1_5m`
Directional model: `runs/optim_pretrain_sonitude_dist_count_directional_1_5m`
Mix rule: `y = b_hat + sum (g_k - 1) * s_hat_k ; y=(1-a)*b_hat + a*z_hat`

| Stage | Pass | Note |
|---|---|---|
| A Oracle mix-rule | PASS | Pass if g=1 recovers b_hat, oracle leak=0 control matches clean mix, boosting speaker0 raises s0 SI-SDR vs cutting it, and leakage degrades control. |
| B Separation probe | PASS | Pass if K=2 soft-mask probe on oracle bubble yields boost/cut control SI-SDR > 3 dB with low mix residual. Model-b_hat transfer reported separately. |
| C DOA / region | PASS | Pass if ear-pair GCC-PHAT gets L/R sign >=65% and >=40% of 2-speaker scenes are separable by a ?30? angular region. Not a full-azimuth UCA claim. |
| D Session ID | PASS | Pass if stable-embedding tracker holds IDs across pause/motion. DOA-only ablation is expected to fail under crossing ? embeddings required for Speaker 1/2 persistence. |
| E Pi cost envelope | FAIL | Pass if multi-out mean infer (77.52 ms) fits 2 hops in 16.0 ms block. Host CPU proxy ? re-measure on Pi 5 before deploy. |
| F Realtime mixer flag | PASS | Pass if g=1 is exact pass-through and non-unity gains change output. Wired behind --inbubble-mixer. |
| G Beta / product targets | PASS | Pass if product-weighted steered mixes audibly follow ?0 (speaker nearer look direction wins SI-SDR) and outside weights stay 0. |
| H Steered head train | PASS | Pass if directional run has best/last checkpoints and steered_head params train. Full SI-SDR gate vs bubble baseline requires overnight train_pt. |
| I Stream / beta gains | PASS | Pass if g=1 still recovers promote b_hat and ?-weighted gains match or beat blind softmask on s0 boost / s1 leakage when ?0 known. |
| J Cost / deploy gate | PASS | Pass if default remains promote b_hat and either steered fits the 16 ms block budget or we document laptop-only directional zoom. |

## Product blockers
- Stage E: multi-out wet path exceeds 16 ms block budget on host proxy; keep mixer default pass-through (g_k=1) until Pi 5 re-measure / lighter split.
- Stage B: soft-mask transfer onto promoted b_hat is weak; oracle-bubble split works, model-b_hat cut/boost control is not yet product-ready.
- Stage D: DOA-only IDs fail under crossing; stable speaker embeddings required for Speaker 1/2.
- Stage J: steered aux not enabled by default; ship promote b_hat until Pi 5 confirms budget (laptop-only directional zoom OK).

**Deploy recommendation:** Ship promoted bubble unchanged as default listen. Directional steered head lives under …_directional_1_5m; enable only if Stage J budget holds. Promote directional -> default only on explicit user request (copy/promote workflow).

**Overall gates:** MIXED - see blockers; default remains promote b_hat.