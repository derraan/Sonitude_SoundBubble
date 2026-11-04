# Sound Bubble measure core

Host-agnostic capture + offline analysis for technical tests. The unified
PySide6 shell (`tools/tech_testbench`) calls this package; you can also run the
CLI directly.

**Live Pi→Windows raw PCM streaming is out of scope.** Tailscale carries
compact metrics / trial summaries only (`/telemetry` + `telemetry.jsonl`).
Move tap / HATS WAVs with `scp` when needed.

Operator day index: [`tools/tech_testbench/docs/README.md`](../tech_testbench/docs/README.md).

## Path 2 vs path 1

| Path | Entry | Result |
| ---- | ----- | ------ |
| **2 — live** | `trial` / Polar with ONNX | Stimulus + RT + ear-rig in one take → acoustic wet at `hats_*` plus digital taps |
| **1 — offline** | `parity` / `--input-wav` | Replay saved `raw_8ch` → digital wet only |

Bring-up order: **Windows path-2 smoke**, then **Pi**. Zoo training can lag —
use the current `edge/zoo`; promote better epochs as **new** stems (do not
overwrite `*_epNN*` in place).

## Device roles (headphone + ear-rig)

| Hardware | Flag / field |
| -------- | ------------ |
| Studio monitor 2ch | `--stimulus-out-device` |
| Pico 6ch mics | `--pipeline-in-device` |
| Pico 2ch → earcups | `--pipeline-out-device` |
| Ear-rig 2ch | `--hats-in-device` |

`stimulus_out` ≠ `pipeline_out` (PortAudio exclusive open).

## Acoustic profiles

| Profile | Typical use | Stimulus out | Pipeline | Ear-rig |
| ------- | ----------- | ------------ | -------- | ------- |
| `windows_pico` | Part A levels / REW / raw array | Monitor (or Pico DAC for REW-only DAC FR) | Pico in/(out) | optional (`hats_*` when set) |
| `pi_hats` | **Part B path-2** on Windows or Pi | Monitor / speakers | Pico in → earcups | required (`hats_*.wav`) |

GUI HATS picker is enabled on both profiles; measure core records ear-rig
whenever `hats_in_device` is set. HATS open prefers 48 kHz then falls back to
the interface default (e.g. QUAD-CAPTURE WDM-KS at 44.1 kHz).

Do not compare `windows_pico` tap curves to `pi_hats` ear-rig curves without an
explicit cross-profile note. Filter analysis by `acoustic_profile` in `trial.json`.

### HATS clock / SR assumptions

Pico UAC2 and the external interface are **independent clocks**. Prefer matching
rates (e.g. 48 kHz HATS with 24/48 kHz Class B) and keep trials short. Measure
core arms HATS (when `hats_in_device` is set) → starts ONNX RT / Pico capture →
settle → plays stimulus → stops/flushes. Alignment is **start-of-take** (see
`sync` in `trial.json`), not bit-clock / sample-locked.

## CLI

```powershell
# Smoke (no devices) — verifies cpp|python argv
python -m tools.measure smoke --runtime python

# Path 2 on Windows — monitor + Pico I/O + ear-rig (replace device indices)
python -m tools.measure trial `
  --acoustic-profile pi_hats --host local_windows `
  --stimulus D:\stim\rew_ess_timing.wav `
  --stimulus-out-device 5 `
  --pipeline-in-device 3 --pipeline-out-device 4 `
  --hats-in-device 6 `
  --runtime cpp --zoo-dir edge/zoo --bubble-radius 1.5 `
  --distance-m 1.5 --azimuth-deg 0 --elevation-deg 0

# Path 2 on Pi — same roles, host pi_rig
python -m tools.measure trial `
  --acoustic-profile pi_hats --host pi_rig `
  --stimulus /home/sonitude/stim/rew_ess_timing.wav `
  --stimulus-out-device 2 `
  --pipeline-in-device 0 --pipeline-out-device 0 `
  --hats-in-device 4 `
  --runtime python --zoo-dir edge/zoo --bubble-radius 1.5

# windows_pico algorithm without ear-rig (Part A / array-only)
python -m tools.measure trial `
  --acoustic-profile windows_pico `
  --stimulus D:\stim\rew_ess_timing.wav `
  --stimulus-out-device 5 `
  --pipeline-in-device 3 --pipeline-out-device 4 `
  --runtime cpp --zoo-dir edge/zoo --bubble-radius 1.5 `
  --distance-m 1.5 --azimuth-deg 0 --elevation-deg 0

# Raw polar (hardware only; no ONNX)
python -m tools.measure raw-polar `
  --stimulus D:\stim\rew_ess_timing.wav `
  --stimulus-out-device 4 --pipeline-in-device 3 `
  --azimuth-deg 30 --elevation-deg 0

# Path 1 — offline C++ vs Python parity (before trusting acoustic curves)
python -m tools.measure parity --raw-8ch path\to\raw_8ch.wav --measure-root measure

# Offline suppression / polar / FR (writes metrics.json + plots/ + run analysis/)
python -m tools.measure analyze --measure-root measure --run-id run_...
```

Trial layout (also written by the UI TrialStore):

```text
measure/<run_id>/<trial_id>/
  trial.json
  metrics.json
  plots/
  raw_8ch.wav  mapped_6ch.wav  dry_ref.wav  wet_out.wav   # from --tap-dir
  hats_l.wav / hats_r.wav / hats_stereo.wav               # when hats_in_device set
  stimulus_used.wav
measure/<run_id>/telemetry.jsonl   # compact scalars only — never WAVs
```

## Windows `soundbubble-rt` build (Class B testbench)

```powershell
cmake -S edge/cpp -B edge/cpp/build/win -G Ninja `
  -DSOUNDBUBBLE_WITH_ORT=ON -DSOUNDBUBBLE_WITH_PORTAUDIO=ON `
  -DSOUNDBUBBLE_WITH_SOXR=ON -DSOUNDBUBBLE_BUILD_TESTS=OFF
cmake --build edge/cpp/build/win --target soundbubble-rt
```

soxr/ORT via vcpkg or FetchContent. CMake falls back to a linear SRC if libsoxr
is absent — use real soxr for FR work.

Device probe:

```powershell
.\edge\cpp\build\win\soundbubble-rt.exe --list-devices
.\edge\cpp\build\win\soundbubble-rt.exe --preflight --input-device N --output-device N `
  --input-sr 24000 --output-sr 24000
```

Stop the Pi unit before moving the Pico to the PC:
`ssh sonitude@172.20.10.8` then `sudo systemctl stop sonitude-soundbubble`.

## Windows+Pico REW FR day (hardware only)

1. Use the **existing** REW ESS + timing-reference stimulus WAV (do not regenerate
   a new sweep for Python / measure trials).
2. Open REW; dual-channel transfer function with timing reference.
3. Measure absolute mic / DAC FR at fixed geometry (on-axis, fixed distance).
4. Export IR/FR if you want optional Python overlays later
   (`tools.measure analyze --rew-fr-json …` notes the path only).
5. No neural path on this day — REW answers “is the hardware honest?”

## Windows+Pico Python hardware day (levels + Polar)

1. Mic levels: Levels tab **Play stimulus + calibrate** (or
   `tools/mic_calibration_probe.py --stimulus …`) using the **same** ESS WAV;
   emits `calibration.json` for firmware `k_mic_cal_gain[]`. Clap mode remains.
2. Polar az/el (tech testbench **Polar** tab): play the **same** stimulus WAV;
   capture Pico mics; optional ONNX taps (`raw,mapped,dry,wet`); ear-rig
   `hats_*` whenever `hats_in_device` is set; always archive `stimulus_used.wav`.
   Start-of-take time sync across USB interfaces (not bit-clock locked). After
   capture, run analysis writes fullband polars for mics / dry / wet / hats /
   stimulus under `measure/<run_id>/analysis/`.
3. Algorithm day (path 2): UI Algorithm tab or `python -m tools.measure trial`
   with `pi_hats`, selectable runtime + gains + `--tap-dir`. Auto-writes
   `plots/suppression_fr.*` from `dry_ref`/`wet_out`.

Full procedures: [`tools/tech_testbench/docs/`](../tech_testbench/docs/).

## Tailscale telemetry (metrics only)

Controller already exposes bearer-token `/telemetry` (C++ + Python). Measure
core appends compact JSON lines to `measure/<run_id>/telemetry.jsonl`
(geometry, gains, exit codes — **no audio**).

```bash
curl -H "Authorization: Bearer $TOKEN" http://<pi-tailscale-name>:PORT/telemetry
```

Pull full WAV trees on demand:

```text
scp -r sonitude@172.20.10.8:~/sb_rt_dev/measure/<run_id> D:\measures\<run_id>
```

## Selectable ONNX RT

Same argv family for both runtimes (`--input-gain-db`, `--output-gain-db`,
`--mix-dry`, limiter flags, `--tap-dir` / `--tap-select` / caps).
`wet_out` is **pre-output_gain**. Default tap select is `dry,wet` unless the
measure core arms all four for algorithm trials.

Runtime toggle: Linux `SIGUSR1` arm/disarm, `SIGUSR2` flush; MIX-UART
`TAP on|off|toggle|flush`.
