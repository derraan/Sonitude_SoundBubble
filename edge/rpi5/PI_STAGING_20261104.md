# Pi staging 2026-11-04

**Pi access:** `ssh sonitude@172.20.10.8` only. From Windows use `C:\Windows\System32\OpenSSH\ssh.exe`; its agent holds the key, while Git Bash ssh doesn't.

## What was on the Pi (unchanged)

- **Service:** `sonitude-soundbubble` has been `failed` since Oct 31, with `PortAudioError: Error querying device 0`. The Pico wasn't attached, so the Python device preflight in `deploy_soundbubble.sh` failed. Only HDMI audio cards are present now.
- **Binary:** `/home/sonitude/sb_rt_dev/edge/cpp/build/pi/soundbubble-rt`, built Oct 31. It has none of today's runtime fixes, no `--mic-gains`, and no tap recorder.
- **Live config:** `/etc/sonitude-soundbubble.env` has `CHANNEL_MAP=0,1,2,3,4,5`, which **conflicts with the owner canon** `5,4,3,2,1,0` (`edge/MIC_INDEX_CANON.md` §4). It also sets `INPUT_GAIN_DB=35` and has no mic gains.

## Staged (new folder, no sudo, live service untouched): `~/sb_rt_stage_20261104/`

| Path | What |
|---|---|
| `edge/cpp/build/pi/soundbubble-rt` | Today's runtime, built natively (Release, ORT 1.20.1). sha256 `cacae585…` |
| `parity/` | Parity check against the Windows bench build on the speech take `run_20261104_075009_b48fe6/algorithm_20261104_075217_34de`. `dry_ref` is bit-identical; `wet_out` differs by −54 dB relative. About 3.8× real time offline. |
| `zoo_eval/v2_ep25.onnx` | v2 epoch 25 eval export, sha256 `707bd865…`. Eval only, not a zoo artifact. Runs on the Pi. |
| `run/record_session.sh <label> [s]` | Live session recording: `raw_8ch`, `mapped_6ch`, `dry_ref`, `wet_out`, `session.json` (argv, binary and model sha) and `pipeline.log`, written to a new folder `~/sb_recordings/<stamp>_<label>/`. Refuses to start while the service holds the Pico. |
| `run/pi_mic_check.py` | Direct 6-mic check on the Pi, using the same wideband low-state rule as `debug/mic_state_check.py`. Prints relative trims. |
| `run/deploy_soundbubble.sh` | The Pi's current script, plus the C++-only `MIC_GAINS`, `WET_GATE_DBFS` and optional `TAP_DIR` for always-on recording. |
| `run/sonitude-soundbubble.env.20261104` | Proposed env: the new binary, `CHANNEL_MAP=5,4,3,2,1,0`, bench gains (`INPUT_GAIN_DB=15`, `MIC_GAINS=1.514,1.514,6.457,1.514,1.514,6.683`). |

Windows side: `edge/rpi5/pull_pi_recordings.ps1` copies `~/sb_recordings` into a new `measure/pi_<stamp>/`.

## Owner steps to adopt (sudo, interactive)

```bash
sudo cp /etc/sonitude-soundbubble.env /etc/sonitude-soundbubble.env.bak_20261104
sudo cp ~/sb_rt_stage_20261104/run/sonitude-soundbubble.env.20261104 /etc/sonitude-soundbubble.env
cp ~/sb_rt_dev/run/deploy_soundbubble.sh ~/sb_rt_dev/run/deploy_soundbubble.sh.bak_20261104
cp ~/sb_rt_stage_20261104/run/deploy_soundbubble.sh ~/sb_rt_dev/run/
# plug the Pico into the Pi, then:
sudo systemctl reset-failed sonitude-soundbubble && sudo systemctl restart sonitude-soundbubble
```

Recording: `sudo systemctl stop sonitude-soundbubble`, then `~/sb_rt_stage_20261104/run/pi_mic_check.py` (venv python), then `~/sb_rt_stage_20261104/run/record_session.sh <label> 300`.

**Open:** `OUTPUT_GAIN_DB=15` is kept from the Pi. The bench used 10 through a different playback path, so check the level by ear on the first run.

## Update 2026-11-05: dashboard wiring

- **Controller** (`soundbubble-ctl`, source `edge/cpp/src/controller.cpp`):
  - new config key `mic_gains`: comma-separated linear trims, device order, 1–8 values each in (0, 100]
  - passed as `--mic-gains` to the cpp runtime only
  - test: `ControllerConfig.MicGainsArgvCppOnlyAndValidated`
- **Pi binaries:** the controller service's `~/sb_rt_dev/edge/cpp/build/pi/soundbubble-ctl` and `soundbubble-rt` were replaced with the 2026-11-04 builds. The old binaries are kept as `*.bak_20261031`. The controller restarted itself (systemd `Restart=always`, no sudo).
- **Dashboard Settings page** (`edge/dashboard/src/pages/SettingsPage.tsx`, rebuilt into `dist/`):
  - **Bubble size:** 1 m / 1.5 m (recommended) / 2 m
  - **Model:** V1 · EP35, V2 · EP25, V3 · EP37 at 1.5 m; Pretrain 1 m / 2 m
  - **mic_gains** and **channel_map** fields, defaulting to the bench calibration and canon `5,4,3,2,1,0`
  - input gain default 15 dB
  - **Dataset recording:** the tap section already existed; its default now includes `raw` (raw_8ch). WAVs are written on the Pi under `~/sb_rt_dev/measure/<tap_dir>`.
- **Models on the Pi:** listed in `docs/training_status_v2_and_v3_improvements_20261105.md` §4.
- **Production vs dashboard:**
  - "Restart with config" stops the production unit and runs an owned child with the dashboard settings. This needs the one-time sudoers rule `edge/rpi5/install_soundbubble_ctl_sudoers.sh`.
  - The production unit's own settings are still in `/etc/sonitude-soundbubble.env`.

## Update 2026-11-05: recordings go to the external SSD

- **Drive:** Samsung Portable SSD T5, 465.8 GB, exFAT, UUID `B424-9B8F`. It also holds the owner's personal files: **write only inside `sonitude-recordings/`**.
- **Mount:** owner-installed fstab entry
  `UUID=B424-9B8F /mnt/sonitude-drive exfat defaults,nofail,uid=1000,gid=1000,umask=022,x-systemd.device-timeout=10 0 0`
  - the mount point is `chattr +i`, so nothing can be written to the SD card if the drive is missing
  - files are owned by `sonitude`, so no sudo is needed
- **Recording root:** `/mnt/sonitude-drive/sonitude-recordings/`

  | Subfolder | Written by |
  |---|---|
  | `dashboard/` | Dashboard "Dataset recording" (default tap_dir `measure/dashboard`) |
  | `sessions/` | `run/record_session.sh <label> [s]` (refuses if the drive isn't mounted) |
  | `mic_checks/` | `run/pi_mic_check.py` (refuses if the drive isn't mounted) |
  | `service/` | Optional `TAP_DIR` in the service env |

- **Dashboard path:** `~/sb_rt_dev/measure` is a symlink to the recording root. The controller validates tap paths after following links, so no service change was needed.
- **Write test:** an offline replay with all four taps wrote to `/dev/sda1` (30 MB), then the test folder was removed.
- **Laptop:** `edge/rpi5/pull_pi_recordings.ps1 [-Name <folder>]` copies from the drive into a new `measure/pi_<stamp>/`.

## Update 2026-11-05: "Dataset recording" = audio + logs

With Dataset recording on, the controller (`edge/cpp/src/controller.cpp`) creates a new folder per Restart:

`/mnt/sonitude-drive/sonitude-recordings/<tap_dir>/<YYYYmmdd_HHMMSS>/`

| File | Content |
|---|---|
| `raw_8ch.wav`, `mapped_6ch.wav`, `dry_ref.wav`, `wet_out.wav` | Audio taps (selected streams; `.partN` after a Flush) |
| `session.json` | Start time, full config (model, radius, gains, mic gains, channel map), runtime argv, model size |
| `pipeline.log` | Every runtime stdout/stderr line with a UTC timestamp |
| `telemetry.jsonl` | One JSON line per parsed `[profile]` (latency, underruns, drift) and monitor update (channel levels, output level, speaker count/distance) |
| `session_end.json` | End time and runtime exit code (written on Stop, Restart or child exit) |

- **Controller test:** `ControllerSession.DatasetRecordingWritesSessionLogAndTelemetry`. It is POSIX-only and runs a fake runtime, so it runs in Linux CI and skips on Windows and on a Pi while production is active.
- **Pi:** controller rebuilt and deployed; the previous binary is kept as `soundbubble-ctl.bak_20261105_mic_gains`.
