# Windows + Pico — REW absolute mic/DAC FR day

Hardware-only frequency response for the Class B Pico (mics + onboard DAC).
Uses the **existing REW ESS (sine sweep) + timing-reference WAV** — do not
regenerate a new sweep in REW for later Python / Algorithm trials.

Neural / ONNX path stays off. Suppression (`dry_ref` vs `wet_out`) and
**wet/suppression FR** (`plots/suppression_fr.*`) are Algorithm-tab / tap
metrics — not a REW job. REW answers absolute mic/DAC honesty; Algorithm FR
answers “what does the suppressed RT output look like vs frequency?” Use the
**same** Setup ESS for both so campaigns stay comparable.

## Preconditions

1. Pico on the Windows PC; Pi `sonitude-soundbubble` stopped (USB single-owner).
2. Class B device opens at 24 kHz (see [`edge/cpp/WINDOWS_BUILD.md`](../../../edge/cpp/WINDOWS_BUILD.md)).
3. Operator-owned stimulus WAV path recorded in Setup → **Stimulus WAV**
   (same file every comparable trial). Prefer a **24 kHz** stereo ESS with
   acoustic timing ref on **Left** (`*_refL`); Setup → Play channel **Left**
   (or Both). Matching capture SR avoids host resample skew for Python
   `metrics.timing_ref` / E2E absolute legs (REW still uses its own dual-ch
   timing-ref mode independently).
4. Fixed geometry for absolute FR: on-axis, fixed distance, quiet room.

## REW procedure

1. Open **REW** (external). This testbench’s **REW FR** tab is checklist-only.
2. Configure dual-channel transfer function with **timing reference** using the
   shared ESS + timing-ref stimulus (the same file selected in Setup).
3. Route playback to the **Pico DAC** (Class B out) and capture the mic under
   test from the Pico UAC2 inputs (or the dual-ch loop you use for TF).
4. Measure absolute **mic** FR and **DAC** FR at the fixed geometry. Keep
   levels out of clip; note gain / Windows mixer state in your lab notes.
5. Export **IR / FR** (and coherence if useful) for optional Python overlays
   later. Prefer files that can sit beside a trial under
   `measure/<run_id>/<trial_id>/` or a sibling `rew_exports/` folder — the GUI
   does not drive REW.
6. Record in notes (or a manual trial folder): `stimulus_path`, distance,
   `azimuth_deg` / `elevation_deg`, host=`windows_testbench`,
   `acoustic_profile=windows_pico`.

## What REW is for vs not for

| Goal | Use REW? |
| ---- | -------- |
| Absolute mic / DAC FR, IR, coherence, delay, THD | **Yes** |
| Mic element level matching (Setup ESS / clap) | No → Levels tab / Python probe |
| Raw array polar az/el | No → play same stimulus + multi-ch capture |
| Internal suppression + wet FR (dry vs wet) | No → Algorithm (auto `suppression_fr`) |
| FR polar vs azimuth/elevation | No → Algorithm trials + offline `analyze_run` |
| Product FR through earcups / ear-rig (path 2) | No → Part B on `pi_hats` (Windows first, then Pi); see [`campaign_grids.md`](campaign_grids.md) |

## After the day

- Keep the stimulus WAV unchanged for Levels / Polar / Algorithm days.
- Optional: copy exported FR/IR into a Library trial folder and register paths
  in `trial.json` notes so offline overlays know which REW export belongs to
  which geometry.
- On `controlled_ess` Levels / Polar / Algorithm / E2E takes, Python also
  detects the ESS timing-ref marker → `plots/timing_ref.*` + `metrics.timing_ref`
  (absolute stim→mic / dry / wet). That does **not** replace REW absolute FR,
  and it does **not** replace mic↔mic `relative_delay_ms` for array TDOA.
- Live Pi→Windows PCM is out of scope; REW runs locally on Windows with the
  Pico attached.
