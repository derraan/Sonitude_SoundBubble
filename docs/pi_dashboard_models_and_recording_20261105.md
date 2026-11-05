# Pi + dashboard: model selection, bubble size and dataset recording

**Date:** 2026-11-05
**Scope:** what runs on the Pi, how the dashboard selects models, and how recording works (audio + logs to the external SSD).
**Detailed change log:** `edge/rpi5/PI_STAGING_20261104.md`. **Model evaluations:** `docs/eval_v2_ep37_20261105.md`, `docs/training_status_v2_and_v3_improvements_20261105.md`.

## 1. System overview

| Part | Where | Notes |
|---|---|---|
| **Production unit** `sonitude-soundbubble` | Pi, systemd | Settings in `/etc/sonitude-soundbubble.env` (owner/sudo). Runtime: `~/sb_rt_stage_20261104/edge/cpp/build/pi/soundbubble-rt` (2026-11-04 build). Current model: V3 · EP37 (`optim_v2_ep37_eval_20261105.onnx`). Backup env: `/etc/sonitude-soundbubble.env.bak_ep35`. |
| **Controller** `soundbubble-controller` | Pi, systemd, Tailscale `100.125.176.100:8000` | `~/sb_rt_dev/edge/cpp/build/pi/soundbubble-ctl` (2026-11-05). Launches the owned runtime `~/sb_rt_dev/edge/cpp/build/pi/soundbubble-rt` (2026-11-04 build). Old binaries kept as `*.bak_20261031` and `soundbubble-ctl.bak_20261105_mic_gains`. |
| **Dashboard** | Laptop: `edge/dashboard`, `npm start` → http://127.0.0.1:8787 | The proxy injects the bearer token from `.env.local`. Serves `dist/` (rebuild with `npm run build`). |
| **Recording drive** | Pi: `/mnt/sonitude-drive` (Samsung T5, exFAT, UUID `B424-9B8F`) | Recordings **only** go in `sonitude-recordings/`; the rest of the drive holds the owner's personal files. |

**Pi access:**
- Use `ssh sonitude@172.20.10.8` only (repo rule). From Windows, use `C:\Windows\System32\OpenSSH\ssh.exe`; its agent holds the key.
- `sudo` stays with the owner.

**Production vs dashboard:**
- "Restart with config" stops the production unit (passwordless rule `edge/rpi5/sudoers-sonitude-soundbubble-ctl`) and runs an **owned** runtime with the dashboard settings.
- Production settings in `/etc` are unchanged by the dashboard.

## 2. Dashboard Settings page

**Code:** `edge/dashboard/src/pages/SettingsPage.tsx`

### 2.1 Bubble size and model

| Bubble size | Model button | File (`~/sb_rt_dev/edge/zoo/`) | Use |
|---|---|---|---|
| **1.5 m (recommended)** | **V1 · EP35** | `optim_sonitude_dist_count_1_5m_ep35.onnx` | Production default; deepest far rejection; darker speech |
| 1.5 m | **V2 · EP25** | `optim_v2_ep25_eval_20261104.onnx` | Keeps more speech (close, ±60° elevation, behind); brighter |
| 1.5 m | **V3 · EP37** | `optim_v2_ep37_eval_20261105.onnx` | Best on ESS/synthetic sweeps, sharper 1.5 m edge; duller speech |
| 1 m | Pretrain 1 m | `optim_pretrain_1m_20261105.onnx` | Synthetic-only demo (edge about 0.9–1.0 m, about 35–40 dB rejection) |
| 2 m | Pretrain 2 m | `optim_pretrain_2m_20261105.onnx` | Synthetic-only demo (edge about 2.0–2.2 m, about 32 dB rejection) |

- **Picking a model:** choosing a bubble size picks that size's default model. The V1–V3 buttons appear only at 1.5 m.
- **Telemetry:** the 1 m / 2 m pretrains have no distance/count heads, so the dashboard shows no distance telemetry for them.
- **Naming:** "V3" here is just a dashboard slot (v2 epoch 37). It is not the v3 design iteration (`docs/training_iteration_v3_design_20261104.md`).

### 2.2 Levels and mapping (sent on every restart)

| Field | Default | Meaning |
|---|---|---|
| `input_gain_db` | 15 | Applied **on top of** mic gains |
| `mic_gains` | `1.514,1.514,6.457,1.514,1.514,6.683` | Linear, by **device** channel (ch2/ch5 corrected +12.6/+12.9 dB, wideband only). Empty = off. cpp runtime only. |
| `channel_map` | `5,4,3,2,1,0` | Owner canon, `edge/MIC_INDEX_CANON.md` §4 (model input 0 = left ear = USB ch5) |
| `output_gain_db` | 15 | |

**Total gain per mic at defaults:** about +18.6 dB on ch0/1/3/4 and about +31.2/+31.5 dB on ch2/ch5. Raise `input_gain_db` (not the mic gains) to change the overall level without changing the per-mic balance.

### 2.3 Dataset recording (audio + logs)

**Dataset recording = On**, then **Restart with config**, creates a new folder per restart:

```
/mnt/sonitude-drive/sonitude-recordings/dashboard/<YYYYmmdd_HHMMSS>/
  raw_8ch.wav        device order, before input gain and mic gains (model-independent; keep for datasets)
  mapped_6ch.wav     model input order after gains
  dry_ref.wav        reference (mic 0 path)
  wet_out.wav        model output, before output gain
  session.json       start time, full config (model, radius, gains, mic gains, channel map), runtime argv, model size
  pipeline.log       every runtime stdout/stderr line, UTC-timestamped
  telemetry.jsonl    one JSON line per [profile] (latency, underruns, drift) and monitor update (levels, output, speakers/distance)
  session_end.json   end time + exit code (on Stop, Restart or runtime exit)
```

- **Streams:** default is `raw,mapped,dry,wet`. Deselect streams to save space; at 24 kHz float, raw_8ch is about 2.8 GB/hour.
- **Limits:** `tap_max_seconds` (default 120) and `tap_max_mb` cap a recording; 0 = unlimited. Set `tap_max_seconds` to the session length you want.
- **Arm / Disarm / Flush:** pause and resume recording, or close the current WAV segment. After a Flush, new segments are written as `*.partN.wav` in the same folder.
- **Path:** `tap_dir` defaults to `measure/dashboard`. On the Pi, `~/sb_rt_dev/measure` is a symlink to `/mnt/sonitude-drive/sonitude-recordings`, and the controller only accepts paths inside it.

## 3. Recording without the dashboard (Pi shell)

| Command (in `~/sb_rt_stage_20261104/run/`) | Writes to | Notes |
|---|---|---|
| `./record_session.sh <label> [seconds]` | `sonitude-recordings/sessions/<stamp>_<label>/` | Uses the production env (`/etc/sonitude-soundbubble.env`): WAVs + `session.json` (binary/model sha256) + `pipeline.log`. The production unit must be stopped first. |
| `~/Sonitude/Sound_Bubble/.venv/bin/python pi_mic_check.py` | `sonitude-recordings/mic_checks/<stamp>.wav/.json` | Direct 6-mic low-state check; play a broadband stimulus. Prints relative trims. |
| `TAP_DIR=/mnt/sonitude-drive/sonitude-recordings/service` in the service env | `sonitude-recordings/service/<stamp>/` | Optional always-on recording for the production unit (`deploy_soundbubble.sh`) |

Both scripts refuse to run if the SSD isn't mounted.

## 4. Getting the data to the laptop

```powershell
.\edge\rpi5\pull_pi_recordings.ps1                         # everything in sonitude-recordings
.\edge\rpi5\pull_pi_recordings.ps1 -Name dashboard/20261105_101500
```

- Copies into a new `measure/pi_<stamp>/` folder; never overwrites.
- Measurement data stays out of git. Add finished sets to the Drive folder and manifest (`docs/data/MEASURE_DATA_20261104.md`).

## 5. Drive safety

- **Auto-mount:** fstab entry `UUID=B424-9B8F /mnt/sonitude-drive exfat defaults,nofail,uid=1000,gid=1000,umask=022,x-systemd.device-timeout=10 0 0`. The Pi still boots without the drive, and files are owned by `sonitude`, so no sudo is needed.
- **SD-card guard:** `/mnt/sonitude-drive` is `chattr +i` while unmounted, so a missing drive makes recording fail instead of filling the SD card. Not yet tested with the drive unplugged.
- **Write area:** agents and scripts write **only** inside `sonitude-recordings/`. Never delete or modify anything else on the drive.
- **Before unplugging:** stop recording (Stop/Restart without recording), then the owner runs `sudo umount /mnt/sonitude-drive`.

## 6. Verification status (2026-11-05)

| Check | Result |
|---|---|
| Pi runtime parity vs the Windows bench build (1 m speech take, EP35) | dry bit-identical, wet −54 dB relative; about 3.8× real time |
| All five dashboard models load and run on the Pi (offline) | OK, about 2.0–2.4 ms per inference step |
| Write to SSD via the dashboard path (all four taps, offline replay) | OK, 30 MB on `/dev/sda1`; test folder removed |
| C++ tests (Windows) | 40/40 pass, incl. `ControllerConfig.MicGainsArgvCppOnlyAndValidated` |
| `ControllerSession.DatasetRecordingWritesSessionLogAndTelemetry` | POSIX only: runs in Linux CI; skipped on Windows and on a Pi with production active |
| **Live dashboard session (owned restart with recording on)** | **Not yet run.** It stops the production unit; run from the dashboard and check the session folder |
| Unplugged-drive guard | Not yet tested |

## 7. Rollback

| What | How |
|---|---|
| Production model | `sudo cp /etc/sonitude-soundbubble.env.bak_ep35 /etc/sonitude-soundbubble.env && sudo systemctl restart sonitude-soundbubble` |
| Production env (pre-2026-11-04) | `/etc/sonitude-soundbubble.env.bak_20261104` |
| Controller | `cp ~/sb_rt_dev/edge/cpp/build/pi/soundbubble-ctl.bak_20261105_mic_gains` (or `.bak_20261031`) over `soundbubble-ctl`, then `pkill -x soundbubble-ctl` (systemd restarts it) |
| Owned runtime | Same, with `soundbubble-rt.bak_20261031` |
| Deploy script | `~/sb_rt_dev/run/deploy_soundbubble.sh.bak_20261104` |
| Dashboard recording path | `rm ~/sb_rt_dev/measure` (symlink only; data on the SSD stays) |
