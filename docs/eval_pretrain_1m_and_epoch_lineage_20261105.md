# 1 m pretrain on the real array + true epoch counts (EP35, v2 ep25, v2 ep37)

**Date:** 2026-11-05
**Related:**
- `docs/eval_v2_ep37_20261105.md`
- `docs/eval_v2_ep25_vs_ep35_20261104.md`
- `docs/training_status_v2_and_v3_improvements_20261105.md` (§4 Pi model table, A9)

## Summary

1. **The 1 m pretrain is not a working 1 m bubble on the real array.**
   - On the bench it removes about 12–17 dB from speech at 0.5, 1.0 *and* 1.5 m, so it does not reject a talker outside its bubble.
   - It is already selectable on the Pi dashboard (`optim_pretrain_1m_20261105.onnx`). Treat that option as **experimental / not for listening tests**.
2. **Epoch labels are cumulative, 0-indexed counters**, not "epochs of this run".
   - v2 "ep25" is 26 epochs in total, of which only 1 is v2 training.
   - v2 "ep37" is 37 in total (12 of v2).
   - EP35 is 36 epochs of the synthetic-only pretrain.
   - The 1 m pretrain is 81.
3. **The v2 run stopped early because its console window was closed.**
   - It was meant to run to epoch 40; the last finished epoch is 36.
   - The log ends with `forrtl: error (200): program aborting due to window-CLOSE`.

## 1. 1 m pretrain: real-array bench replay

- **Model:** `runs/_export_pretrain_1m_2m_20261105/optim_pretrain_1m.onnx`, copied unchanged.
  - Trained on `syn_1m` only, `dis_threshold` 1.0.
  - No distance or count head.
- **Method:** `debug/eval_models_on_bench.py` with the models dict reduced to this model and `--bubble-radius 1.0`.
  - Same protocol as the v2/EP35 evals: fresh state, C++ offline, each take's recorded gains, mic gains and channel map, limiter and gate off.
  - All 49 bench takes were replayed in about 44 s.
  - EP35 / v2 columns come from `runs/_eval_v2_ep37_20261105/bench_replay.json` (same protocol; deterministic from a fresh state).
- **Outputs:** `runs/_eval_pretrain_1m_20261105/` (`bench_replay.json`, `bench_replay.log`, `README.md`).

For a 1 m bubble, 0.5 m is **inside**, 1.0 m is **at the edge**, and 1.5 m is **outside** (it should be suppressed).

| Set | d (m) | n | 1 m | EP35 | v2 ep25 | v2 ep37 |
|---|---:|---:|---:|---:|---:|---:|
| ESS azimuth | 1.0 | 14 | 29.3 | 45.8 | 41.7 | 35.5 |
| ESS elevation | 1.0 | 6 | 37.9 | 45.5 | 41.1 | 27.1 |
| Speech run 1 | 1.0 | 15 | 22.3 | 59.8 | 58.7 | 63.7 |
| Speech run 2 | 0.5 | 5 | 16.6 | 65.2 | 55.2 | 66.0 |
| Speech run 2 | 1.0 | 6 | 11.7 | 54.5 | 44.7 | 55.8 |
| Speech run 2 | 1.5 | 3 | **11.9** | 69.6 | 66.6 | 68.0 |

*(Median suppression, dB.)*

**Median keep per band**, 1 m model (dB; bands 100–300 / 300–1k / 1–2k / 2–4k / 4–6k / 6–8k / 8–11k Hz):

| Set | 100–300 | 300–1k | 1–2k | 2–4k | 4–6k | 6–8k | 8–11k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Speech 0.5 m | −20.1 | −16.1 | −12.5 | −11.4 | −13.6 | −6.1 | −3.3 |
| Speech 1.0 m (run 2) | −14.3 | −13.4 | −6.7 | −3.6 | −1.4 | −1.4 | −2.3 |
| Speech 1.5 m | −23.1 | −13.5 | −9.7 | **+1.0** | **+2.0** | **+3.4** | **+7.6** |

**ESS azimuth @ 1 m** (suppression dB):

| Azimuth | 0° | 30° | 60° | 90° | 120° | 150° | 180° | 210° | 240° | 270° | 300° | 330° |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 m model | 13–15 | 32 | 46 | 46 | 42 | 22 | 14 | 26 | 36 | 32 | 26 | 17 |

### Observations

- **No distance discrimination on the array.**
  - Speech at 0.5 / 1.0 / 1.5 m comes out within about 5 dB of each other (16.6 / 11.7 / 11.9).
  - At 1.5 m, outside the bubble, the 2–11 kHz bands are *boosted* by 1–8 dB.
- **It acts like a mild low-cut filter**, not a separator: the lows are cut and the highs mostly pass.
- **The direction response is strongly lopsided:** the front, rear and about 330° stay open (13–17 dB), while the sides (60–120°) close (42–46 dB). This is the same asymmetry pattern as the 1.5 m models.
- **Synthetic data says otherwise** (`training_status_v2_and_v3_improvements_20261105.md` §4):
  - The dry edge is about 0.9–1.0 m.
  - Far rejection is only about 35–43 dB (EP35 about 82 dB).
  - With reverb it is already −9…−11 dB at 0.5 m.

  The real-array behaviour above therefore looks like a **synthetic→real gap**: no real-array data in training, and the old `syn_1m` set.
- **Why the low numbers aren't a win:** the 1 m model's low suppression on in-bubble takes is not an improvement over EP35/v2, because it suppresses the out-of-bubble 1.5 m take just as little.

### Caveats

- Only 3 takes at 1.5 m and **no 2–3 m bench recordings**, so out-of-bubble rejection on the array is barely sampled. Recording 2–3 m takes would settle it for this model and for v2.
- The bench takes were captured with EP35-tuned gains.
- Fresh-state offline replay ≠ live (see recurrent-state notes in `polar_sweep_1m_ep35_20261104.md`).

**Recommendation:** keep the 1 m (and 2 m) pretrain off the default Pi path. Use plan **A9** (fine-tune 1 m / 2 m from the v3 1.5 m recipe with real-array data) for a real radius family.

## 2. True epoch counts

**How the counter works:**
- `src/hl_modules/distance_based_hl_module.py` keeps 0-indexed epochs.
- With a `.pt` `init_ckpt` it **continues the counter and metric history** from the parent checkpoint (`[init_ckpt] continuing from current_epoch=…`).
- `best.pt` is written inside `on_epoch_end` *before* the increment, so its `current_epoch` is the epoch it was saved in.
- `last.pt` is written after the increment, so its `current_epoch` is the *next* epoch to run.

| Model | Lineage | Own-run epochs | **Total epochs** | Evidence |
|---|---|---|---:|---|
| 1 m pretrain | trained from scratch on `syn_1m` (`init_ckpt` none, run limit 150) | 0–80 | **81** | `best.pt` `current_epoch` 80, `metric_values` 0..80 |
| EP35 | trained from scratch, `optim_pretrain_sonitude_dist_count_1_5m` (`init_ckpt` none, run limit 40), synthetic only | 0–35 | **36** | zoo `…_ep35.runtime.json` `trained_epochs: 36`. The original `best.pt` is lost; the copies under `Sound_Bubble_optim_distance/runs/…/checkpoints/` don't unpickle. |
| v2 ep25 (`best.pt`) | pretrain epochs 0–23 → epoch 24 (mic0m25 edge_aux fine-tune, the sacred EP24) → v2 | 25 | **26** | `current_epoch` 25, `metric_values` 0..25; v2 log: "epoch 24 already logged; starting next epoch 25" |
| v2 ep37 (`last.pt`) | same as ep25 | 25–36 (12) | **37** | `current_epoch` 37 (= next to run), `metric_values` 0..36 |

**What the counts mean:**
- **Same root, different branches:**
  - EP35 and both v2 checkpoints share the same pretrain root (epoch 23 of `optim_pretrain_sonitude_dist_count_1_5m`, promoted as `runs/_promoted/optim_pretrain_sonitude_dist_count_1_5m_best.pt`).
  - EP35 spent epochs 24–35 on the synthetic-only recipe.
  - v2 spent epoch 24 on the mic0m25 edge_aux syn+real fine-tune and epochs 25–36 on the v2 syn+real recipe.
- **Similar budgets:** all three 1.5 m models have similar total budgets (26–37 epochs). Their differences come from recipe, not training time.
- **v2 ep25 is EP24 + 1 epoch.** Read the ep25-vs-EP35 comparison with that in mind.
- **The 1 m pretrain has the most training (81 epochs)**, so its bench result is not under-training.

## 3. v2 run status

- **The run is stopped, not finished:**
  - No trainer process is running.
  - `last.pt` was last written 2026-11-05 07:14 at `current_epoch` 37.
  - The run was meant to reach epoch 40 (see `training_status_v2_and_v3_improvements_20261105.md` §1).
- **Cause:** the log `runs/optim_finetune_ep24_ep35prog_synreal_lean_v2_from_ep24.log` (UTF-16) ends mid-epoch with `forrtl: error (200): program aborting due to window-CLOSE`. The console window hosting the trainer was closed (the Intel Fortran runtime in MKL/NumPy aborts on console close). It was not a training error.
- **To finish the run:**
  - Relaunching with the same `--run_dir` resumes from `last.pt` (epoch 37).
  - Run it detached (e.g. `Start-Process -WindowStyle Hidden` or a scheduled task) so closing a terminal does not kill it.
