# Sound Bubble dashboard: setup for new users

The dashboard is a web UI that runs on **your laptop** and controls the Sound Bubble pipeline on a **Raspberry Pi 5** over **Tailscale**.

```
 Browser ──► laptop proxy (127.0.0.1:8787) ──Tailscale──► Pi controller soundbubble-ctl (:8000) ──► soundbubble-rt + Pico
             adds the bearer token                         starts/stops the runtime, telemetry, recording
```

- **Token:** the controller requires a bearer token. The laptop proxy adds it server-side, so it never reaches the browser.
- **Network:** the controller listens only on the Pi's Tailscale address. Only devices in your tailnet can reach it.

You need three things working, in this order:
1. Tailscale on both machines (§1)
2. The controller on the Pi (§2)
3. The dashboard on the laptop (§3)

---

## 1. Tailscale (both machines)

### 1.1 Laptop
1. Install Tailscale from https://tailscale.com/download and sign in.
2. **Joining someone else's tailnet** (e.g. the project owner's): ask the owner to invite you, or to share the Pi device with you (admin console → Machines → Pi → Share). Accept the invite with the same account you signed in with.
3. Check it works with `tailscale status`. You should see your laptop and, once §1.2 is done, the Pi.

### 1.2 Raspberry Pi (owner, once per Pi)
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up            # opens a login URL; sign in to the tailnet
tailscale ip -4              # note this 100.x.y.z address: it's the controller address
```
- In the admin console (https://login.tailscale.com/admin/machines):
  - **disable key expiry** for the Pi, so it doesn't drop off after 180 days
  - optionally enable **MagicDNS** to reach it by name
- **Optional, recommended:** add an ACL so only the intended users can reach port 8000 on the Pi:
  ```json
  { "action": "accept", "src": ["user@example.com"], "dst": ["<pi-tailscale-name>:8000"] }
  ```

### 1.3 Check
From the laptop, `tailscale ping <pi-tailscale-ip>` should get a reply.

---

## 2. Pi: controller service

**Prerequisites:**
- the runtime is built on the Pi (`edge/cpp/PI_BRINGUP.md`)
- the Pico audio interface is attached

The project Pi uses this layout; adjust the paths if yours differ:

| Item | Path on the project Pi |
|---|---|
| Repo root (`SOUNDBUBBLE_ROOT`) | `/home/sonitude/sb_rt_dev` (must contain `edge/zoo/*.onnx`; the dashboard models are in git) |
| Runtime | `<root>/edge/cpp/build/pi/soundbubble-rt` |
| Controller | `<root>/edge/cpp/build/pi/soundbubble-ctl` |
| Recording drive | `/mnt/sonitude-drive/sonitude-recordings` (see `docs/pi_dashboard_models_and_recording_20261105.md`) |

### 2.1 Build
```bash
cd <root>/edge/cpp
cmake -S . -B build/pi -G Ninja -DCMAKE_BUILD_TYPE=Release -DSOUNDBUBBLE_BUILD_TESTS=OFF
cmake --build build/pi -j3 --target soundbubble-rt soundbubble-ctl
```

### 2.2 Token (shared secret)
```bash
TOKEN=$(openssl rand -hex 32); echo "$TOKEN"     # keep this; the laptop needs it (§3.2)
echo "SOUND_BUBBLE_CONTROLLER_TOKEN=$TOKEN" | sudo tee /etc/soundbubble-controller.env >/dev/null
sudo chmod 600 /etc/soundbubble-controller.env
```
**Never commit the token.** Share it with users privately; a password manager works.

### 2.3 Service
Use **`edge/cpp/packaging/soundbubble-controller.service`** (C++ controller). Don't use `edge/rpi5/soundbubble-controller.service`; it's the legacy Python controller.
```bash
sudo cp <root>/edge/cpp/packaging/soundbubble-controller.service /etc/systemd/system/
# Edit it: set User/Group, paths, and --host to YOUR Pi's `tailscale ip -4` address.
sudo systemctl daemon-reload
sudo systemctl enable --now soundbubble-controller
systemctl status soundbubble-controller --no-pager
```

**Let the dashboard take over the Pico from the production pipeline.** "Restart with config" stops the `sonitude-soundbubble` unit, which needs one passwordless sudo rule:
```bash
sudo bash <root>/edge/rpi5/install_soundbubble_ctl_sudoers.sh
```

**Recording (optional):** mount the external drive and link `<root>/measure` to `/mnt/sonitude-drive/sonitude-recordings`, as described in `docs/pi_dashboard_models_and_recording_20261105.md` §5. Without it, recordings go to `<root>/measure` on the SD card.

### 2.4 Check (on the Pi or the laptop)
```bash
curl http://<pi-tailscale-ip>:8000/health                                       # no token needed
curl -H "Authorization: Bearer $TOKEN" http://<pi-tailscale-ip>:8000/telemetry   # needs the token
```

---

## 3. Laptop: dashboard

### 3.1 Install
Needs **Node.js 20+** (https://nodejs.org).
```bash
git clone https://github.com/derraan/Sonitude_SoundBubble.git
cd Sonitude_SoundBubble/edge/dashboard
npm ci
npm run build
```
Until PR #16 is merged, check out `feat/pi-dashboard-models-recording` first.

### 3.2 Configure
```bash
cp .env.local.example .env.local
```
Edit `.env.local` (it's git-ignored):
```
PI_URL=http://<pi-tailscale-ip>:8000
PI_TOKEN=<token from §2.2>
POLL_MS=500
HOST=127.0.0.1
PORT=8787
LOCAL_MEASURE_DIR=          # optional: local folder with pulled recordings for the Trials page
```

### 3.3 Run
```bash
npm start            # serves dist/ and proxies /api to the Pi
```
Open **http://127.0.0.1:8787**.
- **Connected:** the banner shows the link as OK, and the Bubble page shows live levels.
- **Observe mode:** the production unit is running. Go to Settings and press **Restart with config** to take over with the chosen model, gains and recording settings.
- **Development:** run `npm run proxy` in one terminal and `npm run dev` (Vite, hot reload) in another.

---

## 4. Using it

- **Settings:**
  - **Bubble size:** 1 m / 1.5 m (recommended) / 2 m
  - **Model:** V1 · EP35, V2 · EP25, V3 · EP37
  - **Gains:** input gain (applied on top of mic gains), `mic_gains`, `channel_map`
  - **Dataset recording:** audio + logs into a timestamped folder
- **Details:** `docs/pi_dashboard_models_and_recording_20261105.md` (what each model is, what gets recorded, rollback).

---

## 5. Troubleshooting

| Symptom | Check / fix |
|---|---|
| Banner shows link down / `ECONNREFUSED` / timeout | `tailscale status` on both machines; `tailscale ping <pi-ip>`; `systemctl status soundbubble-controller` on the Pi; `PI_URL` uses the Pi's **Tailscale** IP and port 8000 |
| `400` `missing or invalid bearer token` | `PI_TOKEN` in `.env.local` must equal `SOUND_BUBBLE_CONTROLLER_TOKEN` on the Pi; restart `npm start` after editing |
| `{"error":"not allowlisted"}` | You called an API path the proxy doesn't forward; use the UI, or the paths in `server/proxy.mjs` |
| Controller won't start: bind error | `--host` in the unit must be the Pi's current `tailscale ip -4`; the unit waits about 30 s for Tailscale at boot |
| Restart fails: "failed to stop sonitude-soundbubble" | Install the sudoers rule (`edge/rpi5/install_soundbubble_ctl_sudoers.sh`) |
| Restart fails: "model does not exist" | The model files must exist under `<root>/edge/zoo/` on the Pi (`git pull` there) |
| Restart fails: "tap_dir must be inside …" | Recording path must be under `measure/` (the default `measure/dashboard` is fine) |
| No sound / audio device error in logs | Pico attached? On the Pi, `soundbubble-rt --list-devices` should show device 0 = `INMP441 8ch + DAC 2ch` |
| Page loads but is blank or stale after an update | `npm run build` again, then hard-refresh the browser |
| SSH to the project Pi | `ssh sonitude@172.20.10.8` only (repo rule) |
