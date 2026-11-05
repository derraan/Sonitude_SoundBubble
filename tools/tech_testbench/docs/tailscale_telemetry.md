# Tailscale telemetry (metrics only)

Compact trial summaries and live profile/monitor scalars cross Tailscale via the
**existing** bearer-token controller. There is no second network service and
**never** WAV / PCM payloads on this channel. Full tap / HATS trees still move
with `scp` of `measure/<run_id>/`.

## Endpoints

| Method | Path | Auth | Body |
| ------ | ---- | ---- | ---- |
| `GET` | `/health` | none | `{ "ok": true }` |
| `GET` | `/telemetry` | Bearer | live `last_profile` / `last_monitor`, `taps`, recent `trials_tail` |
| `GET` | `/trials?run_id=&since=` | Bearer | JSONL records from `measure/<run_id>/telemetry.jsonl` |

Same bind + token as `/status` / `/logs` (`SOUND_BUBBLE_CONTROLLER_HOST` /
`PORT` / `TOKEN`). Measure runs append one JSON object per trial to
`measure/<run_id>/telemetry.jsonl`; the controller serves the latest run’s tail.

## Auth header

```text
Authorization: Bearer <SOUND_BUBBLE_CONTROLLER_TOKEN>
```

## Windows poll (PowerShell)

Replace the Tailscale name/IP and token. HTTP on port `8000` is the default
controller listen (TLS only if you terminate it yourself).

```powershell
$Pi = "100.x.y.z"   # or Tailscale MagicDNS name
$Token = $env:SOUND_BUBBLE_CONTROLLER_TOKEN

curl.exe -sS -H "Authorization: Bearer $Token" `
  "http://${Pi}:8000/telemetry"

curl.exe -sS -H "Authorization: Bearer $Token" `
  "http://${Pi}:8000/trials?run_id=20260401_algo"
```

Equivalent bash:

```bash
curl -H "Authorization: Bearer $SOUND_BUBBLE_CONTROLLER_TOKEN" \
  "http://<pi-tailscale-name>:8000/telemetry"
```

## Emit side

- Measure core / testbench parity append compact records (host, runtime,
  geometry, gains, `suppression_db` when known, `trial_path`, …) — no audio.
- While the pipeline is owned or observed, `/telemetry` also mirrors the latest
  `--profile` / `--monitor` scalars parsed from runtime stdout.
