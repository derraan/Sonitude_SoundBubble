# Pull Pi recordings into a NEW local folder under measure/ (never overwrites).
#   .\edge\rpi5\pull_pi_recordings.ps1            # everything in /mnt/sonitude-drive/sonitude-recordings
#   .\edge\rpi5\pull_pi_recordings.ps1 -Name 20261105_101500_cafe_walk
param([string]$Name = "")
$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Dest = Join-Path $Repo ("measure\pi_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
if (Test-Path $Dest) { throw "exists: $Dest" }
New-Item -ItemType Directory $Dest | Out-Null
$Remote = if ($Name) { "/mnt/sonitude-drive/sonitude-recordings/$Name" } else { "/mnt/sonitude-drive/sonitude-recordings/*" }
& "C:\Windows\System32\OpenSSH\scp.exe" -r "sonitude@172.20.10.8:$Remote" $Dest
Get-ChildItem -Recurse $Dest | Measure-Object Length -Sum | ForEach-Object { "{0} files, {1:N1} MB -> {2}" -f $_.Count, ($_.Sum / 1MB), $Dest }
