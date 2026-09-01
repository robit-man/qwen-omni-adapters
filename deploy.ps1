param(
  [ValidateSet("Task", "Service")]
  [string]$Mode = "Task"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $PSScriptRoot).Path
& (Join-Path $RepoRoot "scripts\bootstrap.ps1")
& (Join-Path $RepoRoot "services\windows\install.ps1") -Mode $Mode
& (Join-Path $RepoRoot ".venv\Scripts\qwen-omni-daemon.exe") status
