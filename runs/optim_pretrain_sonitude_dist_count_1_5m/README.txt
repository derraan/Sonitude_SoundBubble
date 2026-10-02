Promoted Sonitude model: optim_pretrain_sonitude_dist_count_1_5m

Updated: 2026-10-02 (overnight continue-train through ep30; passband ripple metric)

This path is the promoted model (deploy / init default).

Checkpoint status:
- best.pt  — current_epoch 23 (best val/loss ≈ −572.9)
- last.pt  — current_epoch 31 (metrics logged epochs 0–30)

Lineage:
- Started as ep93 audio + ep46 heads transplant
- Jointly finetuned (suppress_v1 → copied into this slot)
- Continued training in this promoted run_dir through ~ep30

Previous transplant-only weights:
  runs/archived_promoted_pre_suppress_v1/

Epoch progress (losses + distance-dBFS graphs):
  epoch_progress/          PNG plots + metrics_by_epoch.csv/json
  epoch_progress/PROGRESS_CANVAS_LAYOUT.md   layout guardrail (keep canvas section order)
  val_distance_sweep/      per-epoch sweep CSVs

Passband ripple (RT60 0.2 s):
  Val metric:  val/sweep_passband_ripple_0_2_db
  Definition:  peak−peak out dBFS for d < 1.5 m and out > −60 dBFS
  Train loss:  distance_sweep_val.passband_ripple.w_passband_ripple (needs trainable backbone)
  Canvas:      ripple cursors on best-epoch RT60 overlay

Progress canvas (locked layout): canvases/promoted-epoch-progress.canvas.tsx
Cursor rule: .cursor/rules/promoted-epoch-progress-canvas.mdc
