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
$StreamStatePatchFile = Join-Path $RepoRoot "patches\llama.cpp-qwen3tts-stream-state.patch"
$JsonPatternPatchFile = Join-Path $RepoRoot "patches\llama.cpp-json-pattern-safety.patch"
$JsonStringBoundsPatchFile = Join-Path $RepoRoot "patches\llama.cpp-json-string-bounds.patch"
$ConditioningCachePatchFile = Join-Path $RepoRoot "patches\llama.cpp-qwen3tts-conditioning-cache.patch"
$TokenRowsPatchFile = Join-Path $RepoRoot "patches\llama.cpp-qwen3tts-token-rows.patch"
$IndexedViewsPatchFile = Join-Path $RepoRoot "patches\llama.cpp-indexed-tts-views.patch"
$ReferenceEncoderPatchFile = Join-Path $RepoRoot "patches\llama.cpp-qwen3tts-reference-encoder.patch"
$ReferenceIgpuPatchFile = Join-Path $RepoRoot "patches\llama.cpp-qwen3tts-reference-integrated-gpu.patch"
$ReferencePromptPatchFile = Join-Path $RepoRoot "patches\llama.cpp-qwen3tts-reference-prompt.patch"
$UtteranceStatePatchFile = Join-Path $RepoRoot "patches\llama.cpp-qwen3tts-utterance-state.patch"
$BuildDir = Join-Path $SourceDir "build"

if (-not (Test-Path (Join-Path $SourceDir ".git"))) {
  git clone https://github.com/ggml-org/llama.cpp.git $SourceDir
  git -C $SourceDir checkout --detach $PinnedCommit
}
$Current = (git -C $SourceDir rev-parse HEAD).Trim()
if ($Current -ne $PinnedCommit) {
  throw "llama.cpp must be at pinned commit $PinnedCommit; found $Current"
}

function Test-PatchMarker([string]$File, [string]$Marker) {
  $Path = Join-Path $SourceDir $File
  return (Test-Path $Path) -and [bool](Select-String -Path $Path -SimpleMatch -Pattern $Marker -Quiet)
}

function Apply-NativePatch([string]$File, [string]$Label) {
  git -C $SourceDir apply --reverse --check $File 2>$null
  if ($LASTEXITCODE -ne 0) {
    git -C $SourceDir apply --check $File
    if ($LASTEXITCODE -ne 0) { throw "$Label patch does not apply cleanly" }
    git -C $SourceDir apply $File
    if ($LASTEXITCODE -ne 0) { throw "$Label patch failed" }
  }
}

# Overlapping incremental patches cannot individually reverse-apply once their
# successors change the same context. Match each composed feature before apply.
if (-not ((Test-PatchMarker "common/arg.cpp" '"--tts-stream-frames"') -and
          (Test-PatchMarker "common/arg.cpp" '"--tts-persistent"') -and
          (Test-PatchMarker "tools/mtmd/clip.cpp" 'state_out describes this exact output boundary'))) {
  Apply-NativePatch $PatchFile "Qwen3-TTS PCM stream"
  Apply-NativePatch $PersistentPatchFile "Qwen3-TTS persistent worker"
  Apply-NativePatch $StreamStatePatchFile "Qwen3-TTS streaming decoder state"
}
Apply-NativePatch $JsonPatternPatchFile "JSON pattern string safety"
Apply-NativePatch $JsonStringBoundsPatchFile "JSON string length bounds"
if (-not ((Test-PatchMarker "common/arg.cpp" '"--tts-no-conditioning-cache"') -and
          (Test-PatchMarker "tools/mtmd/mtmd-helper-gen.cpp" 'mtmd_helper_gen_audio_cache_init(') -and
          (Test-PatchMarker "tools/tts/tts.cpp" 'conditioning_cache.get()'))) {
  Apply-NativePatch $ConditioningCachePatchFile "Qwen3-TTS immutable conditioning cache"
}
if (-not ((Test-PatchMarker "common/arg.cpp" '"--tts-conditioning-cache-rows"') -and
          (Test-PatchMarker "src/llama-model.cpp" 'uint32_t llama_model_get_tok_embd_row(') -and
          (Test-PatchMarker "tools/mtmd/mtmd-helper-gen.cpp" 'conditioning->token_rows.find(t)'))) {
  Apply-NativePatch $TokenRowsPatchFile "Qwen3-TTS bounded token rows"
}
if ((Test-PatchMarker "ggml/include/gguf.h" 'gguf_select_tensor_view(') -and
    (Test-PatchMarker "src/llama-model-loader.cpp" 'indexed_view = true;') -and
    (Test-PatchMarker "tools/mtmd/clip.cpp" 'robit.ollama-audio-bridge.v2')) {
    Write-Host "Indexed TTS source views patch is already applied"
} else {
    Apply-NativePatch $IndexedViewsPatchFile "Indexed TTS source views"
}
if ((Test-PatchMarker "tools/tts/reference-encode.cpp" 'params.reference_encoder = true;') -and
    (Test-PatchMarker "tools/mtmd/clip-model.h" 'ref_semantic_book') -and
    (Test-PatchMarker "tools/mtmd/models/qwen3tts-refenc.cpp" 'reference_downsample')) {
    Write-Host "Qwen3-TTS reference acoustic encoder patch is already applied"
} else {
    Apply-NativePatch $ReferenceEncoderPatchFile "Qwen3-TTS reference acoustic encoder"
}
Apply-NativePatch $ReferenceIgpuPatchFile "Qwen3-TTS integrated GPU residency"
if ((Test-PatchMarker "tools/mtmd/mtmd-helper-gen.cpp" 'reference_decoder_state') -and
    (Test-PatchMarker "tools/tts/tts.cpp" 'robit.qwen3tts.reference-prompt.v1')) {
    Write-Host "Qwen3-TTS full reference prompt patch is already applied"
} else {
    Apply-NativePatch $ReferencePromptPatchFile "Qwen3-TTS full reference prompt"
}
if ((Test-PatchMarker "common/arg.cpp" '"--tts-utterance-controls"') -and
    (Test-PatchMarker "tools/tts/tts.cpp" 'robit.qwen3tts.utterance-control.v1') -and
    (Test-PatchMarker "tools/mtmd/mtmd-helper-gen.cpp" 'set_waveform_state')) {
    Write-Host "Qwen3-TTS utterance decoder state patch is already applied"
} else {
    Apply-NativePatch $UtteranceStatePatchFile "Qwen3-TTS utterance decoder state"
}

cmake -S $SourceDir -B $BuildDir -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build $BuildDir --config Release --target llama-server llama-tts llama-tts-reference-encode --parallel $BuildJobs
$Tts = Join-Path $BuildDir "bin\Release\llama-tts.exe"
if (-not (Test-Path $Tts)) { $Tts = Join-Path $BuildDir "bin\llama-tts.exe" }
if (-not (Test-Path $Tts)) { throw "llama-tts.exe was not built" }
& $Tts --help 2>&1 | Select-String -SimpleMatch "--tts-stream-frames" | Out-Null
& $Tts --help 2>&1 | Select-String -SimpleMatch "--tts-persistent" | Out-Null
& $Tts --help 2>&1 | Select-String -SimpleMatch "--tts-no-conditioning-cache" | Out-Null
& $Tts --help 2>&1 | Select-String -SimpleMatch "--tts-conditioning-cache-rows" | Out-Null
& $Tts --help 2>&1 | Select-String -SimpleMatch "--tts-reference-prompt-file" | Out-Null
& $Tts --help 2>&1 | Select-String -SimpleMatch "--tts-utterance-controls" | Out-Null
Write-Host "Built patched llama-server and llama-tts in $BuildDir"
