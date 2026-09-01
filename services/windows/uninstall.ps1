param(
  [ValidateSet("Task", "Service")]
  [string]$Mode = "Task"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if ($Mode -eq "Task") {
  Stop-ScheduledTask -TaskName "QwenOmniAdapters" -ErrorAction SilentlyContinue
  Unregister-ScheduledTask -TaskName "QwenOmniAdapters" -Confirm:$false -ErrorAction SilentlyContinue
} else {
  & $Python -m qwen_omni_adapters.windows_service stop
  & $Python -m qwen_omni_adapters.windows_service remove
}
Write-Host "Removed QwenOmniAdapters. Model tags and runtime data were retained."
