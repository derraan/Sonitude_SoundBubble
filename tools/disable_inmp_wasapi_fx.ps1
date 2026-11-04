# Disable Windows Audio enhancements / Voice Clarity (VocaEffectPack) on INMP capture
# endpoints so WASAPI can see native 8ch @ 24 kHz instead of APO 2ch @ 48 kHz.
# Run elevated (Administrator). Safe to re-run.
#Requires -RunAsAdministrator
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# PKEY_AudioEndpoint_Disable_SysFx
$DisableSysFx = "{1da5d803-d492-4edd-8c23-e0c0ffee7f0e},5"
$NameProp = "{b3f8fa53-0004-438e-9003-51a46e139bfc},6"
$FxMode = "{d04e05a6-594b-4fb6-a80d-01af5eed7d1d},0"
$NullGuid = "{00000000-0000-0000-0000-000000000000}"

$root = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Capture"
$changed = 0
Get-ChildItem $root | ForEach-Object {
    $propPath = Join-Path $_.PSPath "Properties"
    $fxPath = Join-Path $_.PSPath "FxProperties"
    if (-not (Test-Path $propPath)) { return }
    $props = Get-ItemProperty $propPath
    $name = $props.$NameProp
    if ($null -eq $name -or $name -notmatch "INMP441") { return }

    Write-Host "Device: $name ($($_.PSChildName))"
    if (-not (Test-Path $fxPath)) {
        New-Item -Path $fxPath | Out-Null
    }
    # 1 = disable system effects (Sound control panel "Disable all enhancements")
    New-ItemProperty -Path $fxPath -Name $DisableSysFx -PropertyType DWord -Value 1 -Force | Out-Null
    # Clear FX mode / chain pointers so VocaEffectPack is not applied
    New-ItemProperty -Path $fxPath -Name $FxMode -PropertyType String -Value $NullGuid -Force | Out-Null
    Write-Host "  set Disable_SysFx=1, cleared FX mode"
    $script:changed++
}

Write-Host "Updated $changed INMP capture endpoint(s)."
Write-Host "Unplug/replug the Pico (or reboot audio service), then reopen Audacity with WASAPI @ 24000 / 8ch."
