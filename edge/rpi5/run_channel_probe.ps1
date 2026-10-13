param(
  [string]$PiHost = "172.20.10.8",
  [string]$PiUser = "sonitude",
  [string]$PiRepo = "/home/sonitude/sb_rt_dev",
  [string]$PiVenv = "/home/sonitude/Sonitude/Sound_Bubble/.venv",
  [ValidateSet("list-devices", "tap", "clap")]
  [string]$Mode = "clap",
  [int]$InputDeviceIdx = 1,
  [int]$InputSR = 48000,
  [switch]$SyncCode
)

# Syncs edge/ (optional) and runs edge.tools.channel_probe on the Pi for B5/B6 triage.
# tap and clap are interactive (need a TTY). Example:
#   .\edge\rpi5\run_channel_probe.ps1 -SyncCode -Mode clap
#   .\edge\rpi5\run_channel_probe.ps1 -Mode tap
# Apply the printed --channel-map via:
#   .\edge\rpi5\run_pi_live_pipeline.ps1 -ChannelMap 0,1,2,3,4,5 -SyncCode

$ErrorActionPreference = "Stop"
$Pi = "$PiUser@$PiHost"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function Invoke-Pi([string]$Cmd) { ssh -o BatchMode=yes -o ConnectTimeout=8 $Pi $Cmd }

Write-Host "[1/3] Check SSH ($Pi)"
try {
  Invoke-Pi "echo connected" | Out-Host
} catch {
  Write-Host "SSH failed ($Pi). Connect the hotspot / update -PiHost, then re-run." -ForegroundColor Red
  exit 1
}
if ($LASTEXITCODE -ne 0) {
  Write-Host "SSH failed ($Pi). Connect the hotspot / update -PiHost, then re-run." -ForegroundColor Red
  exit 1
}

$Svc = Invoke-Pi "systemctl is-active sonitude.service 2>/dev/null || true"
if ($Svc -eq "active") {
  Write-Host "sonitude.service holds the Pico. Stop it: sudo systemctl stop sonitude.service" -ForegroundColor Red
  exit 4
}

if ($SyncCode) {
  Write-Host "[2/3] Sync edge/ -> ${Pi}:$PiRepo"
  $Tgz = Join-Path $env:TEMP "sb_edge_probe_sync.tgz"
  tar -czf $Tgz -C $RepoRoot --exclude="__pycache__" --exclude="_archive_test" edge
  scp -q -o BatchMode=yes $Tgz "${Pi}:/tmp/sb_edge_probe_sync.tgz"
  Invoke-Pi "mkdir -p $PiRepo && tar -xzf /tmp/sb_edge_probe_sync.tgz -C $PiRepo && rm /tmp/sb_edge_probe_sync.tgz"
} else {
  Write-Host "[2/3] Skipping sync (pass -SyncCode to push edge/ first)"
}

Write-Host "[3/3] Run channel_probe mode=$Mode"
$Extra = ""
if ($Mode -eq "clap") { $Extra = "--repeats 3" }
if ($Mode -eq "tap") { $Extra = "" }
$Cmd = "cd $PiRepo && $PiVenv/bin/python -m edge.tools.channel_probe --device $InputDeviceIdx --sr $InputSR $Mode $Extra"
Write-Host $Cmd -ForegroundColor DarkGray
ssh -t $Pi "bash -lc `"$Cmd`""
Write-Host "Done. If tap printed a map, pass it as -ChannelMap to run_pi_live_pipeline.ps1."
Write-Host "If clap FAIL (>1 ms lag), check Pico CDC ovr/ovr_bytes and re-flash a committed UF2."
