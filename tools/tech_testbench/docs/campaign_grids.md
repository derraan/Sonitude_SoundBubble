# Campaign grids (discrete static takes)

Operator recipes for the tech testbench after the “mostly work” mods.
**Motion is stop → move → retake.** Second talker for 2-speaker in-context is
**external** (BT / other player); tag it in Algorithm context fields.

Launch: `python -m tools.tech_testbench` from repo root.

**Host order:** finish Part B path-2 smoke on **Windows** (`pi_hats` +
`local_windows`), then repeat on the **Pi** (`pi_hats` + `pi_rig`). Training /
zoo export can continue in parallel — smoke with the current `edge/zoo`; when a
newer epoch wins, add a new zoo stem (do not overwrite older `*_epNN*` files).

## Path 2 vs path 1 (ear-rig)

| Path | Where | Ear-rig meaning |
| ---- | ----- | --------------- |
| **2 — live** | Polar (ONNX on) / Algorithm | Same take: monitors play stimulus, Pico RT drives earcups, ear-rig records acoustic wet |
| **1 — offline** | Parity tab / `tools.measure parity` | Replay a saved `raw_8ch` → digital `wet_out`; ear-rig is not re-recorded unless you run a second acoustic pass |

Use path 2 for any claim about suppression **at the ear**.

## What gets logged (per trial)

Every capture writes under `measure/<run_id>/<trial_id>/`:

| Source | Fields / files |
| --- | --- |
| Auto (pipeline / measure) | WAVs: taps `raw_8ch`, `mapped_6ch`, `dry_ref`, `wet_out`; ear-rig `hats_*` whenever `hats_in_device` is set; `stimulus_used.wav` |
| Auto (analysis) | `metrics.json` (suppression, levels mode, `relative_delay_ms` on ESS levels/polar); `plots/*` |
| Keyed on Setup | devices, SR, channel_map, gains, runtime/zoo/bubble, stimulus path + repeat, campaign_kind |
| Keyed on Algorithm / Polar | `distance_m`, `azimuth_deg`, `elevation_deg`, `motion_direction`, `fixed_source_m`, `moving_source_m`, `notes` |
| Keyed on Levels (Part A) | `hw_char_mode` (`per_capsule` \| `headphone_array`), `capsule_id`, `usb_slot` |
| Durable index | `trial.json` (geometry + devices + artifacts); Library reloads this — UI never deletes trials |

For 2-speaker in-context: start the external speaker yourself; check **Tag fixed/external 2nd source** and put a short note (device + file). Measure only drives one stimulus device.

## Shared Setup checklist

1. Acoustic profile + host:
   - Part A (levels / REW / raw array): `windows_pico` + `local_windows`.
   - Part B path-2 (product + ear-rig): `pi_hats` on **either** host — `local_windows` first, then `pi_rig`.
2. Devices (headphone rig):
   - Stimulus out = studio monitor 2ch
   - Pipeline in = Pico 6ch mics
   - Pipeline out = Pico 2ch → earcups
   - HATS in = ear-rig 2ch
   - Stimulus out ≠ pipeline out (PortAudio exclusive).
3. Paste `--channel-map` after you fill `edge/MIC_INDEX_CANON.md` §4 yourself (after headphone seating is known). Until §4 is filled, smoke devices only — do not trust az/el or distance geometry.
4. Campaign:
   - **controlled_ess** — one shared REW ESS + timing-ref WAV; `stimulus_repeat = 1`. Prefer a **24 kHz** stereo ESS (`*_refL`); Setup → Play channel **Left** (or Both) so the timing-ref leg is audible. Python writes `metrics.timing_ref` (absolute stim→stream lags + IR align offsets) and keeps `relative_delay_ms` as mic↔mic TDOA. Timing-ref analysis does **not** run for speech.
   - **in_context_speech** — speech loop WAV; raise `stimulus_repeat` so the take is long enough. No timing-ref path.
5. Measure root under `measure/` (default). Toolbar **New run** per campaign day.

### Timing-ref vs relative delay (controlled_ess)

| Metric | Meaning | Source |
| --- | --- | --- |
| `relative_delay_ms` | Mic↔mic TDOA (array geometry / SSL check) | GCC between capture channels |
| `timing_ref` / `stim_to_raw_ms` | Absolute stim→stream lag (e2e, IR align) | GCC of ESS timing-ref leg vs each stream |

Prefer matched **24 kHz** ESS + Class B capture. If stimulus SR ≠ capture SR, analysis still runs but sets `sample_rate_mismatch` / warning — do not trust absolute ms claims until rates match.

---

## Part A — Hardware characterization (no jig)

There is **no multi-capsule jig**. Do two sub-campaigns. Prefer a **New run** for each.

### A1 — Per-capsule (one mic at a time on USB slot 0)

Goal: intrinsic level / FR / phase of each capsule at a fixed acoustic position.

1. Remove all capsules from the headphone.
2. Wire **one** capsule into **USB / I2S seat 0** only (same seat every time).
3. Place that mic at the fixed test point (1 m from monitor acoustic centre, same stand/aim).
4. Setup: `windows_pico`, shared ESS, stimulus = monitor, pipeline in = Pico.
5. Levels → mode **per_capsule**, set **Capsule ID** (role or paint mark), **USB slot = 0**.
6. **Play stimulus + calibrate** (and REW for absolute FR/phase if needed).
7. Swap the next capsule onto seat 0; change Capsule ID; repeat until all six are done.

Logged: `hw_char_mode=per_capsule`, `capsule_id`, `usb_slot=0`, levels WAV + `relative_delay_ms` (trivial with one live channel).

### A2 — Headphone array @ 1 m (coplanar)

Goal: installed array response with correct seating.

1. Reseat every capsule into its **correct headphone slot** (then update MIC_INDEX_CANON §4 / channel map if not already).
2. Mount headphone so the **vertical mic arc is coplanar with the speaker baffle**, **1 m** from acoustic centre, on-axis.
3. Levels → mode **headphone_array** (Capsule ID optional / blank). Distance tag 1.0 m on Polar / Algorithm if used.
4. **Play stimulus + calibrate** for relative levels across seats; REW for absolute FR/phase.
5. Optional Polar az/el (ONNX off) only if you rotate the whole headphone (not a jig plate).

Offline delay only:

```powershell
python -m tools.tech_testbench.app.analysis --trial measure\<run>\<trial> --mic-delay-only
```

---

## Part B — Proto + ear rig (path 2 / `pi_hats`)

Live pipeline + simultaneous ear-rig. Same recipe on Windows then Pi.

1. Setup:
   - Profile `pi_hats` (HATS / ear-rig required; picker also works on `windows_pico`).
   - Host `local_windows` for the first smoke; switch to `pi_rig` after Windows is green.
   - Stimulus out = studio monitor / speakers.
   - Pipeline in/out = Pico (mics → earcups).
   - HATS in = ear-rig interface.
   - Devices may sit on separate USB connections; captures are start-of-take synced, not bit-clock locked.
2. Zoo: current `edge/zoo` + bubble radius from the runtime contract. When promoting a better epoch, copy to a **new** zoo stem; leave prior epoch files intact.
3. **Az/el characterization — Polar tab** (preferred): enable ONNX, load
   - **Part B az** — 0…330 @ 30°, d=1.0 m
   - **Part B el** — ±60 @ 30°, az=0, d=1.0 m
   Apply pose → rotate → **Capture polar**. Expects `mapped_6ch` / `dry_ref` /
   `wet_out` / `hats_*` / `stimulus_used`; run analysis builds fullband polars
   for mics, dry, wet, hats L/R, and stimulus (+ suppression).
4. **Distance / campaign — Algorithm tab**: campaign `controlled_ess`, load
   **Part B distances** — 0.5 / 1.0 / 2.0 m (and in-context queues). Apply pose
   → **Start algorithm trial**. Taps + `hats_*.wav` land in TrialStore.
5. Smoke checklist before geometry claims: earcups play wet, `hats_*` non-silent,
   taps flush, `trial.json` lists all four device indices.

CLI equivalent (Windows example — replace device indices):

```powershell
python -m tools.measure trial `
  --acoustic-profile pi_hats --host local_windows `
  --stimulus D:\stim\rew_ess_timing.wav `
  --stimulus-out-device N `
  --pipeline-in-device M --pipeline-out-device P `
  --hats-in-device H `
  --runtime cpp --zoo-dir edge/zoo --bubble-radius 1.5 `
  --distance-m 1.5 --azimuth-deg 0 --elevation-deg 0
```

---

## In-context — 1 speaker

1. Setup: campaign `in_context_speech`, speech WAV, set `stimulus_repeat`.
2. Algorithm: motion **into_bubble** or **out_of_bubble** (reverses distance presets).
3. Load **In-context distances** (0.5…2.0) or **In-context elev × dist**.
4. Leave “fixed 2nd source” unchecked. Notes optional.

## In-context — 2 speakers (external fixed)

1. Same as 1-speaker for the **moving** source (measure stimulus device).
2. Start prerecorded speech on the **external** speaker at the fixed distance.
3. Check **Tag fixed/external 2nd source distance**, set Fixed source (e.g. 0.5 or 2.0 m).
4. Notes: e.g. `BT fixed @ 0.5 m playing utterance_loop.wav`.
5. Run into_bubble and out_of_bubble distance queues separately (New run or clear queue + flip direction).

Not sample-locked dual PortAudio — product/in-context labeling only.

---

## Out of scope

- Continuous walk-while-recording
- Dual synced PortAudio stimulus players
- Automating REW
- Agents editing `MIC_INDEX_CANON.md`
