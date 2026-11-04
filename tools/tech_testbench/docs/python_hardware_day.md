# Windows + Pico — Python hardware day (levels + Polar)

Hardware characterization on the Class B Pico. Reuse the **same REW ESS +
timing-ref stimulus WAV** as the REW FR day ([`rew_fr_day.md`](rew_fr_day.md)).
Absolute FR stays in REW; this day is levels + array polar.

**Profiles**

| Goal | Profile | Host |
| ---- | ------- | ---- |
| Levels / raw Polar (no ear-rig) | `windows_pico` | `local_windows` |
| Polar or Algorithm with ear-rig (path 2) | `pi_hats` | `local_windows` first, then `pi_rig` |

Persist every take under `measure/<run_id>/<trial_id>/` via the tech testbench
**TrialStore** (Levels / **Polar** tabs) or the same folder layout by hand.

See also path-1 vs path-2 and I/O roles in [`README.md`](README.md).

## Preconditions

1. Pico on Windows; Pi unit stopped; Class B opens at 24 kHz
   ([`edge/cpp/WINDOWS_BUILD.md`](../../../edge/cpp/WINDOWS_BUILD.md)).
2. Stimulus WAV path known (Setup → Stimulus WAV) — prefer studio **monitor**
   as stimulus out (not Pico DAC) whenever earcups need the Pico 2ch out.
3. Tech testbench GUI optional but preferred for durable trial folders:

```powershell
cd C:\Users\darre\Sound_Bubble
python -m tools.tech_testbench
```

## Mic element levels

Tool: [`tools/mic_calibration_probe.py`](../../mic_calibration_probe.py).
Uses the **same Setup ESS** as Polar / Algorithm / REW when in sweep mode.

### Via GUI (recommended)

1. Setup → profile `windows_pico`, measure root, Class B as pipeline in,
   stimulus WAV + stimulus out (monitor) + pipeline in.
2. Toolbar **New run** (or let the first trial create a run).
3. **Levels** → **Play stimulus + calibrate** (preferred). Clap-only remains as
   **Clap probe (no play)** if you need a quick check without DAC playback.
4. Trial folder gets `mic_calibration_probe.ndjson`, `calibration.json`,
   `levels_8ch.wav`, and `stimulus_used.wav`. Hand `calibration.json` to an
   agent to commit `k_mic_cal_gain[6]` / [`docs/MIC_FX_CALIBRATION.md`](../../../docs/MIC_FX_CALIBRATION.md).

### Via CLI

```powershell
cd C:\Users\darre\Sound_Bubble
# Shared ESS sweep (preferred)
python tools\mic_calibration_probe.py levels_day `
  --stimulus path\to\ess_timingref.wav --out-device N --in-device M `
  --log measure\manual_levels\mic_calibration_probe.ndjson `
  --calibration-json measure\manual_levels\calibration.json `
  --wav-out measure\manual_levels\levels_8ch.wav

# Clap-only
python tools\mic_calibration_probe.py levels_day --secs 5 `
  --log measure\manual_levels\mic_calibration_probe.ndjson
```

The first positional is a run label written into the NDJSON (`run_id`), not a
subcommand. Default rate is 24 kHz / 8ch / 6 active — matches Class B.

Suggested `k_mic_cal_gain[]` is printed and written to `calibration.json`;
flashing cal gains into Pico firmware stays outside this UI.

## Polar (azimuth / elevation)

**Polar** tab owns az/el characterization. One pose queue can capture:

| Signal | WAV | When |
| ------ | --- | ---- |
| Stimulus (file copy) | `stimulus_used.wav` | always |
| Pico mics | `raw_polar_8ch.wav` or `raw_8ch.wav` / `mapped_6ch.wav` | always / ONNX on |
| Dry / wet | `dry_ref.wav` / `wet_out.wav` | ONNX on |
| Ear-rig stereo | `hats_l.wav` / `hats_r.wav` / `hats_stereo.wav` | Setup `hats_in_device` set |

Streams are **start-of-take time synced** on the host (arm HATS → pipeline/mics
→ settle → stimulus → stop together). USB clocks stay independent — not
bit-clock locked. See `sync` in `trial.json`.

### Geometry convention

| Sweep | Vary | Hold |
| ----- | ---- | ---- |
| Azimuth | `azimuth_deg` | `elevation_deg` (usually 0°) |
| Elevation | `elevation_deg` | `azimuth_deg` (usually 0° / look) |

Prefer `azimuth_deg` / `elevation_deg` in `trial.json` (not a single
`angle_deg`). Load Part B az / el presets from the pose queue.

### Via GUI

1. Setup:
   - ONNX off (raw array): `windows_pico`; stimulus = monitor; pipeline in = Pico.
   - ONNX on + ear-rig (path 2): `pi_hats`; stimulus = monitor; pipeline in/out =
     Pico; HATS in = ear-rig. Paste channel-map only after you fill
     `edge/MIC_INDEX_CANON.md` §4.
2. **Polar** → enable ONNX if you want dry/wet taps; set az/el or load pose
   queue → **Capture polar** (or Prepare folder only).
3. On success the tab refreshes `measure/<run_id>/analysis/` with fullband
   polars for mics / dry / wet / hats L+R / stimulus (+ suppression when dry+wet).

CLI (ONNX off):

```powershell
python -m tools.measure raw-polar `
  --stimulus path\to\ess_timingref.wav `
  --stimulus-out-device N --pipeline-in-device M `
  --azimuth-deg 30 --elevation-deg 0
```

## Persistence layout

```text
measure/<run_id>/
  run.json
  analysis/                   # polar_* aggregates after capture / CLI analyze
  <trial_id>/                 # kind=levels | raw_polar | polar
    trial.json                # includes sync.* timestamps when captured
    metrics.json              # fullband mic/dry/wet/hats/stimulus + FR
    plots/
    *.wav                     # mics / taps / hats_* / stimulus_used
    mic_calibration_probe.ndjson   # levels
    calibration.json
    stimulus_used.wav
```

UI never deletes operator data. Copy WAVs in; do not move sole copies from
other tools without a backup.

## Out of scope this day

- Live Pi→Windows PCM — out of scope; use local Pico on Windows.
- Regenerating the ESS in REW for Python trials — reuse the existing file.
- Bit-clock / ASIO aggregate lock across USB interfaces.
