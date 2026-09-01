param(
  [ValidateSet("Task", "Service")]
  [string]$Mode = "Task",
  [switch]$NoStart
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Daemon = Join-Path $RepoRoot ".venv\Scripts\qwen-omni-daemon.exe"
if (-not (Test-Path $Daemon)) {
  throw "Run .\scripts\bootstrap.ps1 before installing the service."
}

[Environment]::SetEnvironmentVariable("OMNI_REPO_ROOT", $RepoRoot, "User")

if ($Mode -eq "Task") {
  $Action = New-ScheduledTaskAction -Execute $Daemon -Argument "serve" -WorkingDirectory $RepoRoot
  $Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
  $Settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
  Register-ScheduledTask -TaskName "QwenOmniAdapters" -Action $Action -Trigger $Trigger -Settings $Settings -Description "Qwen Omni runtime and dashboard" -Force | Out-Null
  if (-not $NoStart) { Start-ScheduledTask -TaskName "QwenOmniAdapters" }
  Write-Host "Installed QwenOmniAdapters as a per-user managed startup task."
} else {
  $Identity = Get-Credential -Message "Windows account that owns the Ollama models and can access the Ollama service"
  & $Python -m qwen_omni_adapters.windows_service --username $Identity.UserName --password $Identity.GetNetworkCredential().Password --startup auto install
  if ($LASTEXITCODE -ne 0) { throw "pywin32 service installation failed" }
  if (-not $NoStart) { & $Python -m qwen_omni_adapters.windows_service start }
  Write-Host "Installed the QwenOmniAdapters Windows Service."
}
Write-Host "Status: .\.venv\Scripts\qwen-omni-daemon.exe status"
