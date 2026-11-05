# Sound Bubble tech testbench (GUI)

Simple **PySide6** operator shell for Windows+Pico and Pi HATS measure days.
Inspired by the Sonitude Audio Algorithm Test Bench (`_pr48_pcm5101a/testbench`):
tabs + device pickers + subprocess/CLI backends — **no second audio graph in the UI**.

**Bring-up order:** Windows path-2 smoke first, then Pi. Operator day index:
[`docs/README.md`](docs/README.md).

## Run

From repo root (preferred):

```powershell
cd C:\Users\darre\Sound_Bubble
python -m venv tools\tech_testbench\.venv
.\tools\tech_testbench\.venv\Scripts\Activate.ps1
pip install -r tools\tech_testbench\requirements.txt
python -m tools.tech_testbench
```

Or from the package folder:

```powershell
cd C:\Users\darre\Sound_Bubble\tools\tech_testbench
.\.venv\Scripts\Activate.ps1   # if created as above
pip install -r requirements.txt
python -m app.main
```

## Device roles (headphone + ear-rig)

| Hardware | Setup field |
| -------- | ----------- |
| Studio monitor 2ch | Stimulus out |
| Pico 6ch mics | Pipeline in |
| Pico 2ch → earcups | Pipeline out |
| Ear-rig 2ch | HATS in |

Stimulus out ≠ pipeline out. For live RT + ear-rig (path 2), pick profile
**`pi_hats`** (HATS required) even on a Windows host; host target
`local_windows` then `pi_rig` when you move the same recipe to the Pi.
On **`windows_pico`**, HATS / ear-rig is optional — set the device when you
want `hats_*.wav` alongside array / tap takes.

## Tabs

| Tab | Status |
| --- | --- |
| Setup | Acoustic profile, devices, gains, runtime, model/zoo, campaign (ESS vs speech), stimulus repeat, channel map, measure root |
| Levels | Play Setup stimulus + calibrate (clap secondary); `calibration.json` + NDJSON + WAVs + `relative_delay_ms` (ESS) |
| Polar | Az/el pose queue; optional ONNX (`raw/mapped/dry/wet`); ear-rig `hats_*` when device set; `stimulus_used.wav`; start-of-take sync; auto run analysis |
| Algorithm | **Path 2:** ONNX RT + taps + ear-rig while stimulus plays; campaign / distance / in-context; `plots/suppression_fr.*` |
| Parity | **Path 1:** offline cpp vs python wet diff via `tools.measure.parity` on a saved `raw_8ch` |
| REW FR | Checklist + shared-stimulus reminder; absolute mic/DAC FR in REW |
| Library | Browse/reload past runs: WAVs, `plots/`, `trial.json`, metrics |

Backends: [`tools/measure/README.md`](../measure/README.md) (Windows build, REW day, hardware day, telemetry; **no live Pi→Windows PCM**).

Toolbar **New run** starts a fresh `measure/<run_id>/`; trials append until the next New run.
Status bar shows the active run id. Captures refresh the Library list automatically.

Discrete campaign recipes (Part A / B / in-context 1- & 2-spk): [`docs/campaign_grids.md`](docs/campaign_grids.md).

## Persistence (reloadable)

Every capture writes under **Measure root** (default `Sound_Bubble/measure/`):

```text
measure/<run_id>/
  run.json
  <trial_id>/
    trial.json          # geometry, gains, devices, acoustic_profile, artifact index
    metrics.json        # scalars (suppression_dB, etc.)
    plots/              # .png/.svg plus .csv/.json series so plots can be redrawn
    *.wav               # taps, HATS, levels, etc. (never deleted by the UI)
```

`TrialStore` (`app/storage/trial_store.py`) is the **only** writer for WAVs /
plots / metrics / `trial.json`. The Library tab lists trials, previews plot
images, and opens folders — same idea as the Sonitude testbench `ResultStore`.

## Offline analysis + parity CLI

```powershell
# Per-trial / filtered run: suppression, polar az/el, FR series + PNGs
python -m tools.tech_testbench.app.analysis --measure-root measure --run-id run_...
python -m tools.tech_testbench.app.analysis --trial measure\run_x\algorithm_...
python tools\tech_testbench\scripts\analyze_offline.py --measure-root measure --acoustic-profile windows_pico

# Parity: same raw_8ch through cpp + python offline (identical gains)
python -m tools.measure parity --raw-8ch path\to\raw_8ch.wav --measure-root measure
python tools\tech_testbench\scripts\run_parity.py --raw-8ch path\to\raw_8ch.wav
```

Per trial, Algorithm finish (and offline analysis) writes `metrics.json` plus
`plots/suppression_fr.{csv,json,png}` (and `hats_fr.*` when HATS WAVs exist).
Aggregates land under `measure/<run_id>/analysis/`:

- Scalar suppression: `suppression_vs_distance.*`, `polar_azimuth.*`, `polar_elevation.*`
- Fullband level polars: `polar_{azimuth|elevation}_{mics,dry,wet,hats,stimulus}.*`
- FR polar: `polar_azimuth_fr.*` / bands (suppression) plus `polar_*_{dry,wet,mics,hats_l,hats_r,stimulus}_fr.*`

Parity trials store `wet_cpp.wav`, `wet_python.wav`, `wet_diff.wav`, and
`plots/parity_diff.{csv,json}`.

## Operator day docs

| Doc | Contents |
| --- | -------- |
| [`docs/README.md`](docs/README.md) | Index, path-1 vs path-2, I/O roles, Windows→Pi order |
| [`docs/campaign_grids.md`](docs/campaign_grids.md) | Part A / **Part B path-2** / in-context pose queues + 2-spk external tagging |
| [`../../edge/cpp/WINDOWS_BUILD.md`](../../edge/cpp/WINDOWS_BUILD.md) | Windows CMake + vcpkg soxr/ORT; Class B `--list-devices` / `--preflight` |
| [`docs/rew_fr_day.md`](docs/rew_fr_day.md) | REW absolute mic/DAC FR with shared ESS + timing-ref |
| [`docs/python_hardware_day.md`](docs/python_hardware_day.md) | Levels probe + Polar (ONNX off/on) with the same stimulus WAV |

## Out of scope

- **Live Pi→Windows raw PCM** streaming — not planned. Move `measure/<run_id>/`
  trees with `scp`. Tailscale carries metrics/summaries only
  (`/telemetry` + `telemetry.jsonl`); see [`docs/tailscale_telemetry.md`](docs/tailscale_telemetry.md).

Plan reference (read-only): `.cursor/plans/tech_test_capture_f6d8a2ad.plan.md`.
