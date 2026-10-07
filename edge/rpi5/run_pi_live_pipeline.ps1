param(
  [string]$PiHost   = "172.20.10.8",
  [string]$PiUser   = "sonitude",
  # Code + zoo are synced here; the Sonitude checkout (~/Sonitude) is left untouched.
  [string]$PiRepo   = "/home/sonitude/sb_rt_dev",
  [string]$PiVenv   = "/home/sonitude/Sonitude/Sound_Bubble/.venv",

  # baseline = pre-plan launcher values ("before" numbers)
  # defaults = only device/model flags; everything else is inference_pipeline.py's defaults
  # lowlat   = Phase 2 step 7 settings (needs 48 kHz Pico firmware)
  [ValidateSet("baseline", "defaults", "lowlat")]
  [string]$Preset   = "lowlat",

  [string]$ZooDir       = "edge/zoo",
  [double]$BubbleRadius = 1.5,
  # Explicit model inside the synced repo, e.g. edge/zoo/optim_pretrain_1_5m.onnx; overrides the zoo.
  [string]$Model        = "",

  # Device IDs on Pi (from `python edge/inference_pipeline.py --list-devices`). Pick the real
  # hw:*,* devices, NOT 'sysdefault' / 'default' / 'dmix' - those xrun under real signal.
  [int]$InputDeviceIdx  = 1,
  [int]$OutputDeviceIdx = 0,
  [int]$InputChannels   = 8,
  [int]$OutputChannels  = 2,
  [string]$ChannelMap   = "0,1,2,3,4,5",
  [double]$MixDry       = 0.0,

  # Per-run overrides of the preset (0 / "" = use preset value).
  [int]$InputSR           = 0,
  [int]$HopsPerIo         = 0,
  [int]$IntraOpThreads    = 0,
  [int]$MaxStepsPerBlock  = 0,
  [double]$TargetAheadSec = 0,
  [double]$MaxAheadSec    = 0,
  [string]$QueuePolicy    = "",
  [string]$OverloadPolicy = "",

  # SCHED_FIFO via `chrt`. Works without sudo when /etc/security/limits.d grants rtprio.
  [int]$RtPrio = 50,
  [int]$DurationSec = 0,     # 0 = run until Ctrl+C
  [switch]$SyncCode,         # tar edge/ (incl. zoo) from this checkout to $PiRepo first
  [switch]$RequirePreemptRt
)

$ErrorActionPreference = "Stop"
$Pi = "$PiUser@$PiHost"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

$Presets = @{
  baseline = @{ InputSR = 44100; HopsPerIo = 8; IntraOpThreads = 3; MaxStepsPerBlock = 0; TargetAheadSec = 0; MaxAheadSec = 0; QueuePolicy = ""; OverloadPolicy = "" }
  defaults = @{ InputSR = 0;     HopsPerIo = 0; IntraOpThreads = 0; MaxStepsPerBlock = 0; TargetAheadSec = 0; MaxAheadSec = 0; QueuePolicy = ""; OverloadPolicy = "" }
  lowlat   = @{ InputSR = 48000; HopsPerIo = 1; IntraOpThreads = 2; MaxStepsPerBlock = 1; TargetAheadSec = 0.012; MaxAheadSec = 0.030; QueuePolicy = "adaptive"; OverloadPolicy = "silence_fill" }
}
$P = $Presets[$Preset].Clone()
foreach ($k in @($P.Keys)) {
  $v = Get-Variable -Name $k -ValueOnly
  if ($v -and "$v" -ne "0") { $P[$k] = $v }
}

function Invoke-Pi([string]$Cmd) { ssh -o BatchMode=yes $Pi $Cmd }

Write-Host "[1/5] Check SSH ($Pi)"
Invoke-Pi "echo connected" | Out-Host
if ($LASTEXITCODE -ne 0) { Write-Host "SSH failed. If the hotspot changed the Pi's IP: arp -a | findstr 2c-cf-67" -ForegroundColor Red; exit 1 }

Write-Host "[1b/5] Preflight: audio owner, kernel, governor, rtprio"
$Svc = Invoke-Pi "systemctl is-active sonitude.service 2>/dev/null || true"
if ($Svc -eq "active") {
  Write-Host "sonitude.service is running and holds the Pico/DAC. Stop it: sudo systemctl stop sonitude.service" -ForegroundColor Red
  exit 4
}
$Cfg = Invoke-Pi "grep -c '^CONFIG_PREEMPT_RT=y' /boot/config-`$(uname -r) 2>/dev/null || true"
if ($Cfg -ne "1") {
  $Msg = "[warn] PREEMPT_RT kernel not detected (CONFIG_PREEMPT only). Watch max/p99 in --profile."
  if ($RequirePreemptRt) { Write-Host $Msg -ForegroundColor Red; exit 2 }
  Write-Host $Msg -ForegroundColor Yellow
}
$Gov = Invoke-Pi "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown"
if ($Gov -ne "performance") { Write-Host "CPU governor is '$Gov' (expected 'performance')." -ForegroundColor Red; exit 3 }
$RtLimit = Invoke-Pi "ulimit -r"
if ($RtPrio -gt 0 -and ($RtLimit -ne "unlimited") -and ([int]$RtLimit -lt $RtPrio)) {
  Write-Host "ulimit -r is $RtLimit; chrt -f $RtPrio will fail. Add 'sonitude - rtprio 95' to /etc/security/limits.d/ and log in again." -ForegroundColor Red
  exit 5
}

if ($SyncCode) {
  Write-Host "[2/5] Sync edge/ (incl. zoo) -> ${Pi}:$PiRepo"
  $Tgz = Join-Path $env:TEMP "sb_edge_sync.tgz"
  tar -czf $Tgz -C $RepoRoot --exclude="__pycache__" --exclude="_archive_test" edge
  scp -q -o BatchMode=yes $Tgz "${Pi}:/tmp/sb_edge_sync.tgz"
  Invoke-Pi "mkdir -p $PiRepo/logs && tar -xzf /tmp/sb_edge_sync.tgz -C $PiRepo && rm /tmp/sb_edge_sync.tgz"
} else {
  Write-Host "[2/5] Skipping code sync (pass -SyncCode to push edge/ first)"
}

Write-Host "[2b/5] Preflight: Pi venv has runtime deps (no torch needed)"
$Preflight = "$PiVenv/bin/python -c `"import importlib.util,sys; m=[n for n in ('numpy','scipy','onnxruntime','sounddevice','soundfile') if importlib.util.find_spec(n) is None]; print('missing: '+', '.join(m) if m else 'preflight ok'); sys.exit(1 if m else 0)`""
Invoke-Pi $Preflight | Out-Host
if ($LASTEXITCODE -ne 0) { Write-Host "Fix: $PiVenv/bin/pip install -r edge/requirements_edge.txt" -ForegroundColor Red; exit 1 }

$Flags = [System.Collections.Generic.List[string]]::new()
if ($Model) { $Flags.Add("--model $Model") } else { $Flags.Add("--zoo-dir $ZooDir --bubble-radius $BubbleRadius") }
$Flags.Add("--input-device $InputDeviceIdx --output-device $OutputDeviceIdx")
$Flags.Add("--input-channels $InputChannels --output-channels $OutputChannels --channel-map $ChannelMap")
$Flags.Add("--mix-dry $MixDry")
if ($P.InputSR)          { $Flags.Add("--input-sr $($P.InputSR)") }
if ($P.HopsPerIo)        { $Flags.Add("--hops-per-io $($P.HopsPerIo)") }
if ($P.IntraOpThreads)   { $Flags.Add("--intra-op-threads $($P.IntraOpThreads) --inter-op-threads 1") }
if ($P.MaxStepsPerBlock) { $Flags.Add("--max-steps-per-block $($P.MaxStepsPerBlock)") }
if ($P.TargetAheadSec)   { $Flags.Add("--target-output-ahead-sec $($P.TargetAheadSec)") }
if ($P.MaxAheadSec)      { $Flags.Add("--max-output-ahead-sec $($P.MaxAheadSec)") }
if ($P.QueuePolicy)      { $Flags.Add("--queue-policy $($P.QueuePolicy)") }
if ($P.OverloadPolicy)   { $Flags.Add("--overload-policy $($P.OverloadPolicy)") }
$Flags.Add("--profile")

$RtPrefix = if ($RtPrio -gt 0) { "chrt -f $RtPrio " } else { "" }
$TimeoutPrefix = if ($DurationSec -gt 0) { "timeout --signal=INT $DurationSec " } else { "" }
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RemoteLog = "$PiRepo/logs/live_${Preset}_$Timestamp.log"

Write-Host "[3/5] Start live pipeline (preset=$Preset; CTRL+C to stop)"
$Cmd = "cd $PiRepo && echo '[info] preset=$Preset' | tee -a $RemoteLog && $TimeoutPrefix$RtPrefix$PiVenv/bin/python -m edge.inference_pipeline $($Flags -join ' ') 2>&1 | tee -a $RemoteLog"
Write-Host $Cmd -ForegroundColor DarkGray
ssh -t $Pi "bash -lc `"$Cmd`""

Write-Host "[4/5] Pipeline stopped"
Write-Host "[5/5] Log saved on Pi: $RemoteLog"
Write-Host "To view: ssh $Pi `"tail -n 200 $RemoteLog`""
