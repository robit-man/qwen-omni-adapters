param(
  [switch]$SkipLlama,
  [switch]$SkipModels,
  [switch]$SkipLaya,
  [switch]$Prepare
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Venv = if ($env:OMNI_VENV) { $env:OMNI_VENV } else { Join-Path $RepoRoot ".venv" }
$Model = if ($env:OMNI_MODEL) { $env:OMNI_MODEL } else { "robit/qwen3.8-27b-e03-obliterated-omni:q4km" }
$LanguageModel = if ($env:OMNI_LANGUAGE_MODEL) { $env:OMNI_LANGUAGE_MODEL } else { "robit/qwen3.8-27b-obliterated-e03:27b" }

py -3 -m venv $Venv
$Python = Join-Path $Venv "Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel
& $Python -m pip install -e "$RepoRoot[dev]"

if (-not $SkipLaya) {
  $LayaVenv = if ($env:OMNI_LAYA_VENV) { $env:OMNI_LAYA_VENV } else { Join-Path $RepoRoot ".laya-venv" }
  py -3 -m venv $LayaVenv
  $LayaPython = Join-Path $LayaVenv "Scripts\python.exe"
  & $LayaPython -m pip install --upgrade pip setuptools wheel
  & $LayaPython -m pip install "laya==0.3.4" "httpx>=0.27,<1" "PyYAML>=6,<7"
  & $LayaPython -c "from laya import Router; print('Laya resident runtime installed')"
}

if (-not $SkipLlama) { & (Join-Path $PSScriptRoot "build_llama_cpp.ps1") }
if (-not $SkipModels) {
  ollama show $Model *> $null
  if ($LASTEXITCODE -ne 0) { ollama pull $Model }
  ollama show $LanguageModel *> $null
  if ($LASTEXITCODE -ne 0) { ollama pull $LanguageModel }
  & $Python -m qwen_omni_adapters resolve $Model | Out-Null
}
if ($Prepare) {
  $Cache = if ($env:OMNI_COMPONENT_CACHE) { $env:OMNI_COMPONENT_CACHE } else { Join-Path $RepoRoot "runtime-data\components" }
  & $Python -m qwen_omni_adapters prepare $Model --out $Cache --overwrite | Out-Null
}
Write-Host "Bootstrap complete. Run: .\.venv\Scripts\qwen-omni-daemon.exe serve"
