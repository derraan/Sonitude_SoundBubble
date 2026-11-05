# v2 training status (epoch 37) and v3 improvements

**Date:** 2026-11-05
**Run:** `runs/optim_finetune_ep24_ep35prog_synreal_lean_v2_from_ep24` (still training; capped at 40 epochs)
**Related:**
- `docs/eval_v2_ep37_20261105.md`: epoch 37 vs epoch 25 vs EP35, synthetic + bench
- `docs/eval_v2_ep25_vs_ep35_20261104.md`
- `docs/training_iteration_v3_design_20261104.md`: the v3 design; this doc adds to §3 and §4

## 1. Will more epochs help? Short answer: a little, not for the real-array problems

### 1.1 Training curve (from `last.pt` `metric_values`)

The metric history includes the parent run: epochs 0–23 are the parent, and v2 itself starts at epoch 24.

| Epoch | val SI-SDRi (dB) | val SI-SDRi, 1 speaker | train SI-SDRi | Note |
|---:|---:|---:|---:|---|
| 20–23 | +1.1 … +1.7 | +1.2 … +1.9 | +1.8 … +2.0 | parent run |
| 24 | **−15.4** | −15.3 | −13.9 | v2 starts: new loss/data mix |
| 25 | −15.1 | −16.1 | −15.2 | `best.pt` (lowest `val/loss`) |
| 28 | −6.9 | −8.9 | −5.5 | |
| 32 | −0.5 | −1.6 | −2.1 | |
| 36 | **−0.2** | −1.1 | −1.1 | still below the parent run |

**Scheduler:** `ReduceLROnPlateau` on `val/loss` (patience 5, factor 0.5). The learning rate is already halved to 2.5e-5, with 5 bad epochs in a row, so the next halving is due. About 3 epochs remain before the cap.

### 1.2 What epochs 25 → 37 changed (`docs/eval_v2_ep37_20261105.md`)

- **Better:**
  - synthetic edge: sharper, close to 1.5 m
  - reverberant keep up to 1.2 m
  - ESS on the real array: median 35.5 / 27.1 dB vs 41.7 / 41.1 at epoch 25
  - steep elevations and the rear now open
- **Not better:**
  - real-array speech: median +1.4 dB worse than epoch 25
  - the speech residue lost its high end (4–11 kHz keep −48…−54 vs −27…−29 dB)
  - 30°/60° azimuth still closed; 1.5 m still closed
- **Selection metric:** `best.pt` stayed at epoch 25, even though epoch 37 is better on synthetic and ESS. `val/loss` (SNRLP, negative floor about −1790 per empty clip) rewards suppression, not keep.

### 1.3 Conclusion

- More epochs of this recipe recover SI-SDRi and improve stationary, broadband behaviour.
- They won't fix the real-array speech problems, which are data and protocol problems (P1–P3 below).
- **Plan:**
  - let the run reach epoch 40 and evaluate the final epoch with `runs/_eval_v2_ep37_20261105/`
  - select by bench and speech results, not `best.pt`
  - only extend (by at most about 10 epochs) if val SI-SDRi is still clearly rising at epoch 39
  - spend GPU time on v3

## 2. Why the model leans on mic 0 (new finding, 2026-11-05)

Mic 0 is the left ear (`edge/MIC_INDEX_CANON.md` §1). On this headset it is **USB device ch5**, which is one of the two mics that read low (−12.9 dB, wideband).

The code makes it the reference in two places:

| Where | What | Effect |
|---|---|---|
| `src/datasets/general_multisrc_dataset_with_perturbations.py:80` | `reference_channels` defaults to `[0]`: the target is the in-bubble speech **as heard at mic 0** | The model learns to filter M0 and uses the other mics only as spatial cues. Any change to M0's level, spectrum or reverb moves "the talker". |
| same file, lines 212–221, `target_mic0_dbfs: -25` | Each clip is scaled so **mic 0** sits at −25 dBFS | M0's absolute level, and every other mic's level relative to M0, become distance cues. A quiet M0 reads as "far", so the model suppresses. |

**Why mic 0 and not another mic:** no design reason. It's the upstream default (`[0]`), and the later mic-0 normalization reused the same index.

**Consequences on the real array:**
- **Uncorrected, M0 is the quietest mic.** That is the worst possible mic to be low, which explains why the wideband mic-gain correction mattered so much.
- **Hypothesis: head shadow on a one-sided reference.** Talkers on the side away from M0 arrive attenuated and with more reverb, which looks "far". This would fit the 30° vs 330° imbalance on an array that measured symmetric.
  - Not yet confirmed: the bench angle convention vs the wearer's left/right.
  - Test: replay the azimuth takes with mirrored roles.

## 3. v3 improvements (additions to `training_iteration_v3_design_20261104.md` §3)

| # | Change | Fixes | Notes |
|---|---|---|---|
| A1 | **Six-mic level normalization:** level each clip by the mean RMS of all six mics (replacing `target_mic0_dbfs`), and keep ChannelGain ±2 dB jitter | M0 level dependence, low-mic sensitivity | Cheap and safe; set the same −25 dBFS target on the array mean |
| A2 | **Random-ear reference:** per clip, the target is either ear (p = 0.5), with channel order and azimuth labels mirrored to match | Side preference; P3 azimuth imbalance | Keeps the "at the ear" sound; at runtime, output the matching ear or keep the left ear (the model becomes symmetric either way) |
| A3 | **Optional binaural output (2 targets, both ears)** | True stereo, symmetric | Larger output head; needs a deeper finetune; evaluate after A2 |
| A4 | **Mirror augmentation** (already §3.3) + **low-mic augmentation** (−12 ± 1.5 dB wideband on 1–3 channels, p = 0.15, already §3.4), **including M0** | Robust to the actual hardware fault | Make sure M0 is in the augmentation pool |
| A5 | **Stateful long-sequence training** (§3.1, E2/E3): 30–60 s streams, truncated BPTT, quiet → speech schedules | P1: closes up after quiet | Highest priority for live use |
| A6 | **Own-array corpus** (§3.2, E0): distance ladder 0.5–3 m, 2–3 m out-of-bubble set, azimuth/elevation grid | P2: real distance cues; 1.5 m closed | Record on the Pi with `record_session.sh` or the dashboard's Dataset recording (raw_8ch) |
| A7 | **Checkpoint selection on the bench and speech**, not `val/loss` (§3.5) | P4 | Epoch 25 vs 37 shows that `val/loss` picks the wrong epoch |
| A8 | **Per-band keep hinge** (§3.5) | P5: dull or bright residue | Epoch 37 lost the 4–11 kHz keep; add a high-band term |
| A9 | **Radius family:** finetune 1 m and 2 m from the v3 1.5 m recipe instead of the old synthetic pretrains | Real-array 1 m / 2 m options | The current 1 m / 2 m are synthetic-only (see §4) |

**Updated experiment order:**
1. E0: recordings
2. E2/E3: stateful + A1 + A2 + A4, init v2 epoch 37 or epoch 25, picked on the bench
3. E4: + own-array data
4. E5: loss/selection A/B (A7, A8)
5. A9: radius family
6. A3: optional binaural

## 4. Models deployed for A/B on the Pi (dashboard Settings → Bubble size / Model)

| Dashboard label | Radius | File on the Pi (`~/sb_rt_dev/edge/zoo/`) | What it is |
|---|---|---|---|
| **V1 · EP35** | 1.5 m | `optim_sonitude_dist_count_1_5m_ep35.onnx` | Production default |
| **V2 · EP25** | 1.5 m | `optim_v2_ep25_eval_20261104.onnx` | v2 epoch 25 (eval export, sha `707bd865…`) |
| **V3 · EP37** | 1.5 m | `optim_v2_ep37_eval_20261105.onnx` | v2 epoch 37 (eval export, sha `9a51cdc7…`) |
| Pretrain 1 m | 1.0 m | `optim_pretrain_1m_20261105.onnx` | `runs/optim_pretrain_1m` epoch 80, synthetic only, val SI-SDRi 7.45 dB |
| Pretrain 2 m | 2.0 m | `optim_pretrain_2m_20261105.onnx` | `runs/optim_pretrain_2m` epoch 48, synthetic only, val SI-SDRi 4.63 dB |

The dashboard's V3 label here is just a model slot (v2 epoch 37). It is not the v3 design iteration above.

**Synthetic sweep of the 1 m / 2 m pretrains** (`runs/_export_pretrain_1m_2m_20261105/`, output dBFS, input −25, dry / RT60 1.0):

| Distance | 0.5 | 0.8 | 0.9 | 1.0 | 1.2 | 1.5 | 1.8 | 2.0 | 2.2 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 m | −27/−34 | −34/−66 | −40/−64 | **−63**/−66 | −64/−64 | −69/−67 | −57/−67 | −61/−60 | −67/−56 | −67/−65 | −67/−64 |
| 2 m | −26/−28 | −26/−29 | −26/−30 | −26/−35 | −27/−31 | −26/−31 | −30/−34 | **−36**/−47 | −47/−47 | −58/−56 | −58/−59 |

- **Edges:** in the right place (1 m at about 0.9–1.0 m; 2 m at about 2.0–2.2 m).
- **Far rejection:** only about 32–40 dB, against about 80 dB for EP35. These models also keep less in reverb (1 m at RT60 1.0 already −47 dBFS at 0.7 m).
- **No real-data finetune:** they've never seen real WHAM data or mic-0 normalization, and have no distance/count heads (no distance telemetry in the dashboard).
- Treat them as **demo options**; 1.5 m stays recommended.
