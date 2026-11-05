# Tech testbench docs

Operator procedures for the Sound Bubble measure harness UI
([`../README.md`](../README.md)).

**Host order:** bring up path-2 live trials on **Windows first**, then repeat on
the **Pi**. Do not wait for a final zoo export to smoke devices — use the
current `edge/zoo` artifact; when a better epoch wins, add a **new** zoo stem
(never overwrite `*_ep35*` in place).

| Doc | Day |
| --- | --- |
| [`../../../edge/cpp/WINDOWS_BUILD.md`](../../../edge/cpp/WINDOWS_BUILD.md) | Windows `soundbubble-rt` CMake + vcpkg soxr/ORT; Class B probe |
| [`campaign_grids.md`](campaign_grids.md) | Part A hardware, **Part B path-2** (live RT + ear-rig), in-context grids |
| [`rew_fr_day.md`](rew_fr_day.md) | Windows+Pico REW absolute mic/DAC FR (shared ESS + timing ref) |
| [`python_hardware_day.md`](python_hardware_day.md) | Windows+Pico levels + Polar (mics / dry / wet / ear-rig / stimulus) |
| [`tailscale_telemetry.md`](tailscale_telemetry.md) | Poll Pi `/telemetry` over Tailscale (metrics / trial summaries only) |
| [`../../measure/README.md`](../../measure/README.md) | CLI backends, profiles, device roles, offline parity |

## Two ways to get “wet”

| Path | What happens | Ear-rig hears |
| ---- | ------------ | ------------- |
| **2 — live (preferred for product FR)** | Stimulus on monitors + ONNX RT playing Pico → earcups + ear-rig record in the same take | Acoustic wet at the ear |
| **1 — capture then offline** | Record Pico (and optional ear-rig of the room) → later `parity` / `--input-wav` replay | Digital `wet_out` only unless you do a second play-through |

Path 2 is Part B / Algorithm / Polar-with-ONNX. Path 1 is Parity and offline
analysis on a saved `raw_8ch.wav`.

## Current headphone I/O (both hosts)

| Hardware | Setup role |
| -------- | ---------- |
| Studio monitor 2ch out | `stimulus_out_device` |
| Pico 6ch mic in | `pipeline_in_device` |
| Pico 2ch out → earcups | `pipeline_out_device` |
| Ear-rig 2ch in | `hats_in_device` |

`stimulus_out` must differ from `pipeline_out` (PortAudio exclusive open).

## Out of scope

**Live Pi→Windows raw PCM** (UDP/ZMQ/Tailscale audio bridge) is intentionally
out of scope. Tailscale carries compact metrics / trial summaries via the
controller `/telemetry` endpoint; tap WAVs and HATS audio move via `scp` of
`measure/<run_id>/` trees only.

Agents must not edit [`edge/MIC_INDEX_CANON.md`](../../../edge/MIC_INDEX_CANON.md)
— owner fills §4 after a tap probe, then pastes that `--channel-map` into Setup.
