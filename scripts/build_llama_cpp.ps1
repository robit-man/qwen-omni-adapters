param(
  [string]$SourceDir = "",
  [int]$BuildJobs = [Environment]::ProcessorCount
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $SourceDir) { $SourceDir = Join-Path $RepoRoot "vendor\llama.cpp" }
$PinnedCommit = "458681e1d5d4a29a1463c4732e03226cf384b997"
$PatchFile = Join-Path $RepoRoot "patches\llama.cpp-qwen3tts-pcm-stream.patch"
$PersistentPatchFile = Join-Path $RepoRoot "patches\llama.cpp-qwen3tts-persistent.patch"
$BuildDir = Join-Path $SourceDir "build"

if (-not (Test-Path (Join-Path $SourceDir ".git"))) {
  git clone https://github.com/ggml-org/llama.cpp.git $SourceDir
  git -C $SourceDir checkout --detach $PinnedCommit
}
$Current = (git -C $SourceDir rev-parse HEAD).Trim()
if ($Current -ne $PinnedCommit) {
  throw "llama.cpp must be at pinned commit $PinnedCommit; found $Current"
}

foreach ($Patch in @(
  @{ File = $PatchFile; Label = "Qwen3-TTS PCM stream" },
  @{ File = $PersistentPatchFile; Label = "Qwen3-TTS persistent worker" }
)) {
  git -C $SourceDir apply --reverse --check $Patch.File 2>$null
  if ($LASTEXITCODE -ne 0) {
    git -C $SourceDir apply --check $Patch.File
    if ($LASTEXITCODE -ne 0) { throw "$($Patch.Label) patch does not apply cleanly" }
    git -C $SourceDir apply $Patch.File
  }
}

cmake -S $SourceDir -B $BuildDir -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build $BuildDir --config Release --target llama-server llama-tts --parallel $BuildJobs
$Tts = Join-Path $BuildDir "bin\Release\llama-tts.exe"
if (-not (Test-Path $Tts)) { $Tts = Join-Path $BuildDir "bin\llama-tts.exe" }
if (-not (Test-Path $Tts)) { throw "llama-tts.exe was not built" }
& $Tts --help 2>&1 | Select-String -SimpleMatch "--tts-stream-frames" | Out-Null
& $Tts --help 2>&1 | Select-String -SimpleMatch "--tts-persistent" | Out-Null
Write-Host "Built patched llama-server and llama-tts in $BuildDir"
