# Progress canvas layout lock — `optim_pretrain_sonitude_dist_count_1_5m`

This document is the **guardrail** for the promoted-model training progress UI. Keep this layout stable across epoch updates, weight copies, and re-plots.

| Item | Path |
|------|------|
| Canvas | `canvases/promoted-epoch-progress.canvas.tsx` (workspace canvases dir) |
| Cursor rule | `.cursor/rules/promoted-epoch-progress-canvas.mdc` |
| Data / PNGs | `runs/optim_pretrain_sonitude_dist_count_1_5m/epoch_progress/` |
| Sweep CSVs | `runs/optim_pretrain_sonitude_dist_count_1_5m/val_distance_sweep/` |

## Fixed section order (do not reorder)

1. **Header** — H1 `optim_pretrain_sonitude_dist_count_1_5m — epoch progress`, short subtitle, pills (`best ep`, axis note).
2. **Stat strip** — 4 stats: Best epoch · Val SI-SDRi @ best · Val SI-SDRi @ last · Val dist MAE @ best.
3. **Training / validation losses**
   - H3 Total loss (train/val) — SVG `LineSvg`, Y ≈ −580…−400
   - H3 Audio loss (train/val) — same Y convention
4. **Separation metrics** — Val SI-SDRi + Val SNRi (optionally train later as extra series only; keep this H2).
5. **Count / distance head metrics**
   - 2×2 grid: Count accuracy · Count CE · Speaker distance MAE · Distance SmoothL1
   - Then Activity BCE full width
6. **Distance → dBFS characterization**
   - Callout: official axis convention
  - H3 Best-epoch RT60 overlay (`DistanceDbfsSvg`) — includes RT60 0.2 s **passband ripple cursors** (max/min + ΔdB inside bubble)
  - H3 Mean RT60 evolution (`DistanceDbfsSvg`)
7. **Snapshot table** — selected epochs (start / best / last logged).
8. **Footer source line** — points at `metrics_by_epoch.csv` + sweep CSVs.

## Axis conventions (locked)

### Epoch curves (`LineSvg`)

- X: epoch index (0 … N−1), dashed vertical = best epoch.
- Y: true numeric domain for that metric (loss may be largely negative).
- **Do not** use canvas `LineChart` for these if values are negative.

### Distance → dBFS (`DistanceDbfsSvg`) — matches official / ep46 overlay

- X: **source distance (m), far → near** (left 3.0 → right 0.5), same as `scripts/distance_sweep_dbfs.py` `set_xlim(max, min)`.
- Y: **signal level (dBFS)**, typically −20 (top) … −95 (bottom).
- Always show: input mic0 ≈ −25 dBFS, bubble radius vertical at **1.5 m**, RT60 overlay curves (or mean-RT60 evolution).
- Stopband = far / low; passband = near / high.
- **Passband ripple cursors** (best-epoch RT60 0.2 s only): horizontal max/min lines + ΔdB label for active passband points (`d < bubble` and level > −60 dBFS). Lower ripple is better.

## Allowed updates

- Replace embedded series with newer `metric_values` / sweep CSV numbers.
- Bump best/last epoch pills, Stat values, and passband-ripple pill / cursor Δ.
- Regenerate PNGs via `scripts/plot_promoted_epoch_progress.py` and official-style RT60 overlays into `epoch_progress/`.
- Add an **H3 under an existing H2** if a closely related metric appears.
- Keep ripple cursors on the best-epoch overlay when refreshing (recompute from RT60 0.2 series).

## Forbidden

- Reordering top-level H2 sections or replacing the distance characterization with epoch-on-X multi-distance charts as the primary dBFS view.
- Positive-only fake axes (e.g. 0–30 “dBFS”) for suppression curves.
- Promoting experimental run dirs as a second progress canvas layout; keep this canvas as the promoted progress surface.
- Removing train/val loss or head distance-accuracy blocks when refreshing after a finetune weight copy.

## Refresh checklist

1. Export metrics from promoted `checkpoints/last.pt` → `epoch_progress/metrics_by_epoch.{csv,json}`.
2. Ensure `val_distance_sweep/epoch_*.csv` present; regenerate RT60 overlay PNGs with official axes.
3. Update canvas **data constants only**; preserve section order and SVG helpers.
4. Spot-check: loss Y shows large negatives; dBFS Y shows ~−90 stop and ~−25…−30 pass.
