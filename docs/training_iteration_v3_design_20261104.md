# Design iteration v3: state-robust, real-array-calibrated bubble model

**Date:** 2026-11-04
**Status:** **DESIGN, not started.** No configs launched, no checkpoints written.
**Starting point:** v2 epoch 25 (`runs/optim_finetune_ep24_ep35prog_synreal_lean_v2_from_ep24/checkpoints/best.pt`; eval copy in `runs/_eval_v2_ep25_vs_ep35_20261104/`).
**Evidence base:** `docs/FINDINGS_20261104.md`, `docs/eval_v2_ep25_vs_ep35_20261104.md`, `docs/speech_elevation_tests_ep35_20261104.md`, `docs/polar_sweep_1m_ep35_20261104.md`, `docs/mic_low_state_investigation_20261104.md`
**Guardrails:** never overwrite EP24/EP35 sacred artifacts or `runs/_promoted/*`; new run dirs and new names only (`.cursor/rules/promoted-ep24-sacred.mdc`, `local-checkpoint-files.mdc`). The protected polar run is read-only (`protected-polar-sweep-1m-20261104.mdc`).

## 1. Where we are (v2 epoch 25 vs EP35)

| Property | EP35 | v2 epoch 25 | Target for v3 |
|---|---|---|---|
| Synthetic dry passband 0.5–1.2 m | ≈ −25…−26 dBFS (input −25) | same | same |
| Synthetic edge (−6 dB point) | 1.4–1.5 m | 1.4–1.5 m (≈ 5 dB softer at the edge) | 1.5 m ± 0.1 |
| Synthetic reverberant keep at 0.8 m, RT60 1.0 | −77 dBFS | −34 dBFS | ≥ −32 dBFS up to 1.2 m |
| Bench in-bubble suppression, fresh state (median) | 46–66 dB | 41–59 dB | **≤ 15 dB** |
| Azimuth asymmetry 30° vs 330° (ESS, 1 m) | 74.9 vs 28.3 dB | 72.7 vs 27.6 dB | ≤ 6 dB apart |
| Below array (el −60°, 1 m) | 78–79 dB | 62–64 dB | ≤ 20 dB |
| Real-array bubble edge (0° az/el) | ≈ 0.5–1 m | not yet measured | 1.5 m ± 0.25 |
| Live: after quiet periods | **locks closed** (69–77 dB on every pose) | not tested live | no lock: first-utterance keep within 6 dB of fresh state |
| Real val SI-SDRi | — | ≈ −15 dB | > 0 dB |
| Far-talker rejection on the real array (2–3 m) | untested | untested | ≥ 30 dB more than 1 m |

v2 is a consistent improvement (better in 31/49 bench rows, median −2.9 dB), but it doesn't address the three root problems below.

## 2. Root problems this iteration must fix

| # | Problem | Evidence | Why training (not runtime) |
|---|---|---|---|
| P1 | **Recurrent-state lock-in.** Output depends on history; quiet periods drive the model closed and speech doesn't reopen it | Same 0° sweep 37–73 dB by history; identical 90° takes 14 dB apart; live speech closed at every pose 26 s after a restart; fresh-state replay of the same audio opens | Training clips are **4.5 s crops** (`sig_len=4.5`) from fresh state, so the model never sees long streams, quiet → speech transitions or carried state |
| P2 | **Real-array distance cues don't match training.** Effective edge ≈ 0.5–1 m; 1.5 m closed even fresh | Fresh-state distance curve 32 → 54 → 70 dB (0.5/1.0/1.5 m); level drops only 0.8 dB from 1 to 1.5 m, so the model relies on spatial/DRR cues | Syn rooms + a different array geometry (HumanMultiTargetWHAM) dominate training; there are **no recordings from this array** in training |
| P3 | **Direction bias.** Lopsided azimuth and rejection below the array | 30° vs 330° 37 dB apart; −60° el closed even fresh; mic array verified symmetric | Training positives are almost all near ear height; the array/head shadow in the real rig isn't represented |
| P4 | **Checkpoint selection rewards suppression** | `best.pt` picked by `val/loss`, which the SNRLP negative floor (≈ −1790 per empty clip) dominates; `val/si_sdr_i` ≈ −15 dB at the "best" epoch | The selection metric must measure keep *and* reject on real-like data |
| P5 | **Bright residue / spectral tilt** | Open-phase keep: 8–11 kHz −1…−6 dB vs 100–300 Hz −21…−43 dB | Loss is SNR-only and fullband; no per-band keep constraint |

The mic hardware offset (dev 2/5 −12.6/−12.9 dB, dev 4 toggling) is handled at runtime by wideband-only mic gains. v3 adds robustness to it in training (§3.4) but doesn't rely on it.

## 3. Design changes

### 3.1 Stateful long-sequence training (P1, primary)
- **Long streams:** train on **30–60 s** sequences built by concatenating 4.5–5 s clips. Carry the recurrent state across segments with truncated BPTT (backprop every 4–5 s, state detached but not reset). The streaming model already exposes state buffers (ONNX `state_*`), and the training path must use the same streaming step.
- **Quiet → speech schedules:** inside each long stream, insert **silence / room-noise gaps of 2–60 s** (real bench room tails, WHAM noise, digital silence) followed by **in-bubble speech** with a full-keep target, plus out-of-bubble speech with a suppress target. The model must learn to reopen after quiet.
- **Per-segment loss weighting:** weight the first 1–2 s after a quiet → speech transition ×2, so recovery speed is optimized explicitly.
- **Cost:** about 6–10× the step time per sample. Mitigate with batch 2 + grad accumulation 2, and `grad_ckpt` (quality-neutral). **Keep AMP off** (the fastmem/bf16 regressions are documented).

### 3.2 Real data from *this* array (P2, P3)
- **Record a training corpus on the bench** with the Sonitude headset on the HATS, using the existing testbench (raw 8-channel taps are model-independent):
  - **Distances:** 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0 m.
  - **Azimuths:** 0–330° in 30° steps at ≥ 3 distances; elevations −60…+60° at 1 m and 2 m.
  - **Talkers:** LibriSpeech loops (testbench speech campaign) through the loudspeaker; ≥ 3 rooms/positions if possible.
  - **Targets:** for in-bubble sources the target is the clean utterance convolved with a short measured direct-path response (from the ESS takes at the same pose), or the close-mic reference.
  - **Mixing:** mix in-bubble and out-of-bubble recordings offline into multi-talker scenes with real room noise.
- **Use real rooms in the sim:** fit room simulation (RT60, DRR, array geometry incl. head shadow) to the bench measurements, and regenerate the syn corpus with the measured array geometry.
- **Rebalance:** about 40% own-array real scenes, 40% syn with the measured geometry, 20% HumanMultiTargetWHAM (kept for diversity, down-weighted because its array differs).

### 3.3 Direction coverage (P3)
- Add in-bubble positives at **elevations −60…+60°** and the full **360° azimuth**, uniformly sampled, in both syn and real scenes.
- **Mirror augmentation:** swap channels left↔right (map `5,4,3,2,1,0` ↔ `0,1,2,3,4,5`) with mirrored azimuth labels, so the model can't learn a side preference.
- **Eval gate:** 30°/330° and 60°/300° within 6 dB on the bench ESS replay.

### 3.4 Mic-level robustness
- Keep **ChannelGain ±2 dB** (natural tolerance). Add, with p = 0.15, a **per-channel wideband step of −12 ± 1.5 dB on 1–3 random channels**, mimicking the low-state modules, without changing the target.
- Never apply per-pose level normalization; it erases interaural level differences, which the model uses (shown at +60°).

### 3.5 Loss and checkpoint selection (P4, P5)
- **Loss:**
  - SNRLP with `neg_weight` reduced to **10–20**, plus the existing **positive keep hinge** (`pos_keep_loss`, weight 0.5, slack 6 dB).
  - Add a **per-band keep hinge** (5 bands 100 Hz–8 kHz, slack 6 dB) on positives, against the bright-residue tilt.
  - Keep the EP35 `passband_ripple` aux.
- **Checkpoint selection:** replace `val/loss` with a composite on a **held-out own-array bench set**: `score = median in-bubble keep (dB) − 0.5 × median out-of-bubble keep (dB) − asymmetry penalty`. Also log `val/si_sdr_i` on real data, and select only epochs that pass the hard gates (§5).
- **Training-time monitors:** keep the synthetic distance sweep every epoch, and add the **bench replay harness** (`debug/eval_models_on_bench.py`-style, fresh state + long-history protocol) every 2–5 epochs on a fixed subset.

### 3.6 Runtime mitigation (in parallel, not a substitute)
- **Interim:** state refresh with two staggered model instances crossfaded every N seconds, or a state reset on voice-activity onset. Prototype in `soundbubble-rt` behind a flag, and evaluate on the live bench with EP35/v2 to quantify how much of P1 runtime alone can recover.

## 4. Experiment plan (ordered)

| Step | What | Output | Depends on |
|---|---|---|---|
| E0 | Record the own-array corpus (§3.2) + the **out-of-bubble 2–3 m** set + a distance ladder at 0° | `measure/` runs + Drive + manifest | bench time |
| E1 | Evaluate **v2 later epochs** (last.pt / final) with the current harness (synthetic + bench fresh) + **live after restart** | Updated `eval_v2_*` doc | v2 run finishes |
| E2 | Stateful long-sequence trainer (§3.1): data loader for long streams, TBPTT step, quiet → speech schedule; unit-test state carry | `src/` changes + tests | — |
| E3 | v3a: E2 trainer + current data (no own-array data yet), init v2 best | `runs/optim_v3a_stateful_from_v2ep25` | E2 |
| E4 | v3b: + own-array real scenes + measured-geometry syn + direction coverage + mic-step aug (§3.2–3.4) | `runs/optim_v3b_ownarray_from_v3a` | E0, E3 |
| E5 | Loss / selection changes (§3.5) as an A/B on v3b | v3b-loss vs v3b | E4 |
| E6 | Runtime state refresh A/B (§3.6) on the best v3 | live bench | E3+ |
| E7 | Promote under a new name only after gates pass (ONNX export, parity, zoo copy `*_v3*`) | `_promoted`, `edge/zoo/*_v3*` | gates |

## 5. Success gates (all on the bench unless noted)

| Gate | Threshold |
|---|---|
| In-bubble keep, fresh state (speech, 0.5–1.25 m, all poses) | median suppression ≤ 15 dB, no pose > 30 dB |
| **Live** in-bubble keep after ≥ 60 s of quiet | first utterance within 6 dB of fresh-state keep; no closing trend > 6 dB over 20 s |
| Out-of-bubble rejection (2–3 m) | ≥ 30 dB more suppression than 1 m at the same pose |
| Real-array edge (0° az/el) | −6 dB point at 1.5 ± 0.25 m |
| Azimuth symmetry | mirror pairs within 6 dB |
| Below array (−60° el, 1 m) | ≤ 20 dB suppression |
| Spectral balance (open speech) | 100–300 Hz and 8–11 kHz keep within 10 dB of the 1–4 kHz keep |
| Synthetic regression | dry passband 0.5–1.2 m −26 ± 3 dBFS; 2–3 m ≤ −90 dBFS |
| Runtime | 0 underruns live at hops_per_io 1; Pi real-time factor unchanged (same architecture) |

## 6. Risks and open questions

- **Training cost:** long-sequence TBPTT on a 6 GB RTX 4050 may need batch 1–2 and accumulation; AMP stays off unless an fp32 A/B proves it neutral.
- **Target quality for own-array recordings:** loudspeaker playback isn't a human talker, so the direct-path target estimate needs validation (compare with a close-mic reference).
- **Hardware drift:** dev 4 toggling can contaminate recordings. Run `debug/mic_state_check.py` before every recording session and store its JSON with the run.
- **Data/array mismatch:** if the HATS rig differs from on-head use, the edge calibration may shift; plan a small on-head validation set.
- **Selection metric gaming:** the composite must combine keep and reject, otherwise it collapses to "keep everything" (the opposite failure of today).

## 7. Not in scope for v3

- Architecture changes (TF-GridNet size, hop) unless E3/E4 show a capacity limit.
- Firmware changes for the mic low state (owner-gated; tracked in the mic investigation doc).
- Changing the 1.5 m bubble radius.
