# Pi bring-up for soundbubble-rt

Windows testbench build + Class B probe:
[`WINDOWS_BUILD.md`](WINDOWS_BUILD.md).

One systemd unit (`sonitude-soundbubble`) runs either Python or C++. Flip with
`SOUNDBUBBLE_RUNTIME`; do not add a second audio service (both would fight the Pico).

SSH is only `ssh sonitude@172.20.10.8`. Do not use `soundpi.local` or another
account. Type `sudo` passwords yourself in that session.

Repo on the Pi: `/home/sonitude/sb_rt_dev`. Unit entrypoint:
`/home/sonitude/sb_rt_dev/run/deploy_soundbubble.sh` (copy of `edge/rpi5/deploy_soundbubble.sh`).

## Binary

Until a `.deb` is installed, use the tree build:

```text
/home/sonitude/sb_rt_dev/edge/cpp/build/pi/soundbubble-rt
```

Rebuild on the Pi after syncing `edge/cpp`:

```bash
cd /home/sonitude/sb_rt_dev
cmake -S edge/cpp -B edge/cpp/build/pi -G Ninja \
  -DSOUNDBUBBLE_WITH_ORT=ON -DSOUNDBUBBLE_WITH_SHERPA=OFF \
  -DSOUNDBUBBLE_WITH_PORTAUDIO=ON -DSOUNDBUBBLE_WITH_SOXR=ON \
  -DSOUNDBUBBLE_BUILD_TESTS=OFF
cmake --build edge/cpp/build/pi --target soundbubble-rt --parallel
```

Controller only (does not relink the running `soundbubble-rt` binary):

```bash
cmake --build edge/cpp/build/pi --target soundbubble-ctl --parallel
```

`soundbubble-ctl` is `/home/sonitude/sb_rt_dev/edge/cpp/build/pi/soundbubble-ctl`.
The unit file is `edge/cpp/packaging/soundbubble-controller.service` (user
`sonitude`, bind `100.125.176.100:8000`). If that address is not on the box yet,
the process retries `bind` for about 30 seconds and exits; systemd restarts it
until `StartLimitBurst`, then gives up so boot is not stuck. Installing the unit
and `/etc/soundbubble-controller.env` needs `sudo`.

Optional packaged install (puts the binary at `/opt/soundbubble/bin/soundbubble-rt`):

```powershell
.\edge\rpi5\deploy_cpp_deb.ps1 -DebPath .\soundbubble-rt_0.1.0_arm64.deb
```

## Switch runtime (same unit)

From Windows (TTY so you can type the sudo password):

```powershell
.\edge\rpi5\set_pi_runtime.ps1 status
.\edge\rpi5\set_pi_runtime.ps1 cpp -RtBin /home/sonitude/sb_rt_dev/edge/cpp/build/pi/soundbubble-rt
.\edge\rpi5\set_pi_runtime.ps1 python
```

On the Pi:

```bash
bash /home/sonitude/sb_rt_dev/edge/rpi5/sb_runtime.sh status
sudo bash /home/sonitude/sb_rt_dev/edge/rpi5/sb_runtime.sh cpp \
  --rt-bin /home/sonitude/sb_rt_dev/edge/cpp/build/pi/soundbubble-rt
sudo bash /home/sonitude/sb_rt_dev/edge/rpi5/sb_runtime.sh python
```

That upserts `/etc/sonitude-soundbubble.env` and restarts `sonitude-soundbubble`.
If `/opt/soundbubble/bin/soundbubble-rt` exists, omit `--rt-bin` / `-RtBin`.

## Verify

Run journal commands **on the Pi** (do not nest another `ssh` while already logged in):

```bash
systemctl is-active sonitude-soundbubble
journalctl -u sonitude-soundbubble -n 80 --no-pager
```

Expect:

- `[deploy] starting neural Sound Bubble pipeline runtime=cpp` (or `python`)
- `[audio] streams started`
- repeating `[profile] … steps=2 …`

Ignore PortAudio enumeration noise on start: `Unknown PCM surround*`, `iec958`,
`dmix`, `asym`, and `jack server is not running`. Those are probes of virtual /
optional backends. They do **not** mean the Pico is missing. A real device problem
looks like `[audio-config-error]`, `Invalid device`, or no `[profile]` lines.

Confirm the Pico is the open device:

```bash
arecord -l
# card … INMP441 8ch + DAC 2ch
ps -o pid,cmd -p "$(systemctl show -p MainPID --value sonitude-soundbubble)"
```

## Soak and keep cpp

1. `python` baseline: `set_pi_runtime.ps1 python`, then on the Pi
   `bash edge/cpp/packaging/pi_soak.sh --minutes 5`. Keep the log.
2. Switch to `cpp` (commands above). Check `infer_mean` is still ONNX compute
   (~2 ms/hop on this Pi build); the C++ win is non-infer path and scheduling.
3. Pass vs Python: lower non-infer time, `steps=2` more steadily, shed / dry-fill
   not worse, `underrun_frames` not climbing without bound.
4. `bash edge/cpp/packaging/pi_soak.sh --minutes 30` with the unit still active.
5. Listening check; USB mix knob works on **python or cpp** when
   `--mix-uart-port /dev/ttyACM0` is passed (deploy sets this from
   `MIX_UART_PORT`). Verify with `soundbubble-mix-listen /dev/ttyACM0`.
6. Leave `SOUNDBUBBLE_RUNTIME=cpp` as the kept default. Roll back with
   `set_pi_runtime.ps1 python`.

`systemctl stop` sends SIGINT; a clean stop should exit 0.
Device / worker failure must exit non-zero so `Restart=on-failure` recovers.
