# In-bubble controllable mixing - investigation report

Promoted model: `runs/optim_pretrain_sonitude_dist_count_1_5m`
Mix rule: `y = b_hat + sum (g_k - 1) * s_hat_k`

| Stage | Pass | Note |
|---|---|---|
| A Oracle mix-rule | PASS | Pass if g=1 recovers b_hat, oracle leak=0 control matches clean mix, boosting speaker0 raises s0 SI-SDR vs cutting it, and leakage degrades control. |
| B Separation probe | PASS | Pass if K=2 soft-mask probe on oracle bubble yields boost/cut control SI-SDR > 3 dB with low mix residual. Model-b_hat transfer reported separately. |
| C DOA / region | PASS | Pass if ear-pair GCC-PHAT gets L/R sign >=65% and >=40% of 2-speaker scenes are separable by a ?30? angular region. Not a full-azimuth UCA claim. |
| D Session ID | PASS | Pass if stable-embedding tracker holds IDs across pause/motion. DOA-only ablation is expected to fail under crossing ? embeddings required for Speaker 1/2 persistence. |
| E Pi cost envelope | FAIL | Pass if multi-out mean infer (77.52 ms) fits 2 hops in 16.0 ms block. Host CPU proxy ? re-measure on Pi 5 before deploy. |
| F Realtime mixer flag | PASS | Pass if g=1 is exact pass-through and non-unity gains change output. Wired behind --inbubble-mixer. |

## Product blockers
- Stage E: multi-out wet path exceeds 16 ms block budget on host proxy; keep mixer default pass-through (g_k=1) until Pi 5 re-measure / lighter split.
- Stage B: soft-mask transfer onto promoted b_hat is weak; oracle-bubble split works, model-b_hat cut/boost control is not yet product-ready.
- Stage D: DOA-only IDs fail under crossing; stable speaker embeddings required for Speaker 1/2.

**Deploy recommendation:** Ship promoted bubble unchanged. Mixer flag may ship with g_k=1 default. Do not enable multi-stream wet decode on Pi until Stage E passes on-device.

**Overall gates:** MIXED - Stage E cost gate blocks multi-stream deploy.