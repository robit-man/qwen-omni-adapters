# B4b1 — frozen reference acoustic encoder inside the shared bundle

The speech path still uses its original speaker embedding. Full reference-text
and acoustic-prompt conditioning requires a reference tokenizer that was absent
from the shipped GGUF inventory. This milestone supplies that frozen component;
native prompt construction, reference codec-state priming, cache integration and
public voice fields remain unfinished B4 work.

## Source and execution

The build input is Qwen's Apache-2.0
[Qwen3-TTS-Tokenizer-12Hz](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/tree/7dd38ad4e9bad454aae9cd937d0cd577604fe229),
revision `7dd38ad4e9bad454aae9cd937d0cd577604fe229`. Its safetensors SHA-256 is
`836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`.
The export checks that exact digest before importing build-only Torch. The
encoder maps the frozen SEANet, eight transformer layers and the sixteen valid
semantic/acoustic codebooks into 129 F32 GGUF tensors, 190080704 bytes including
metadata. Unused quantizers and decoder weights are not duplicated.

`llama.cpp-qwen3tts-reference-encoder.patch` adds a native graph and a bounded
private-pipe tool, `llama-tts-reference-encode`. It consumes 0.5–30 seconds of
normalized 24 kHz mono audio and emits sixteen integer acoustic codes per frame.
Loaded weights must reside on a GPU; the caller separately proves actual PID
residency before sending the admission handshake. There is no inference CPU
fallback, language-model stack, runtime model download or additional service.
The CPU fixture builder is an offline numerical oracle only.

Initial code agreement was 45%/53% with the reused half-precision convolution
path. Full-precision convolution raised it to 100%/95%. The remaining mismatch
came from cuBLAS TF32 rounding. The encoder child explicitly disables TF32 before
backend initialization; both complete references then matched the frozen oracle:
**304 + 2608 = 2912 codes**, byte-for-byte after integer conversion. That precision
choice is confined to the encoder process. Existing reasoning and speech
processes retain their settings. Only the two recorded references are qualified
by this comparison; no universal/perceptual fidelity claim follows.

## One weight umbrella

Both local candidate tags carry one shared 1672479648-byte sidecar, digest
`7838f27ad60ed5181c60e9b6e7263df944933f07099602aa70a5e96c8640f388`:

- `robit/ornith-1.5-omni:edge-v2-reference-candidate-20261030`
- `robit/qwen3.8-27b-e03-omni:edge-v2-reference-candidate-20261030`

The corresponding language/projector layers remain separately bound to each
backbone. `tts_reference_encoder` uses `s.t.e.*` within that same sidecar; the
model and codec retain `s.t.m.*`/`s.t.p.*`. Indexed preparation creates three
source aliases with zero reconstructed payload. The native reference tool
selects its own metadata and original tensor offsets directly from the installed
blob. The same 2912-code comparison passed through that installed indexed view.

The larger sidecar is an explicit storage cost of adding full reference encoding.
It does not change existing A2 measurements into claims about this richer package.
No weights are committed, and these candidate tags have not been published.

## Reproduction

Use a build environment with this project installed (or `PYTHONPATH=src`) and
Torch, safetensors, Transformers, soxr and GGUF tooling. Supply the pinned local
weights/configuration; these commands do not download them:

```bash
python runtime/export_tts_reference_encoder.py \
  --source-safetensors SOURCE/model.safetensors --out BUILD/reference-encoder.gguf
python runtime/qualify_tts_reference_encoder.py fixtures \
  --source-weights SOURCE/model.safetensors --source-config SOURCE/config.json \
  --reference portal/voices/female_voice.wav --reference portal/voices/default_voice.wav \
  --directory BUILD/reference-fixtures
.venv/bin/qwen-omni pack-tts --tts TTS/model.gguf --tts-projector TTS/codec.gguf \
  --tts-reference-encoder BUILD/reference-encoder.gguf --out BUILD/shared-sidecar.gguf
```

Run `runtime/qualify_tts_reference_encoder.py compare --help` for the bounded
GPU comparison. Discrete CUDA requires scoped broker admission with a fresh
readiness filename. Tegra requires a positive memory floor. The comparison exits
unsuccessfully on any code mismatch, cleans its private outputs and always
reports `full_voice_cloning_qualified:false`.

## Regression scope

All twelve seeded existing-TTS PCM results match the B2 native control, including
A/B/A for both references; continuity gates passed. Native libraries were checked
against each binary's own directory. The Linux builder applies the ninth patch,
builds the reference tool and repeats without rebuilding unchanged objects.
Windows build integration is present; no Windows execution is claimed.

`./scripts/validate.sh` passed **6217 tests** and all remaining gates. The source,
installed payload, native comparison and existing-TTS receipts accompany this
document. Orin builds use a private additive GGUF-base library with the existing
same-pin CUDA kernels; that device's encoder comparison is tracked separately.
The Orin follow-up passed all **2912 codes** against the same frozen fixtures,
using the exact same installed-sidecar digest. Its CUDA backend reports `IGPU`;
an incremental native patch accepts both GPU types while still rejecting CPU
weights. The Python caller additionally required `nvgpu` and `nvmap` handles.
Available system memory stayed above 10408 MiB against a 2048 MiB guard. The
existing primary and call harness stayed healthy. The discrete CUDA comparison
also passed again after that admission change.

The final follow-up source gate passed **6218 tests**. One earlier run passed
6217 and hit the inherited process-group timing test in
`tests/test_passive_focus_guard.py`; all 60 tests in that file passed unchanged
on rerun, followed by the complete successful gate. No cleanup assertion or
unrelated controller code was changed. Encoder startup failures now retain a
bounded native diagnostic in the comparison receipt.

Full voice conditioning, trained heads, complete release qualification and
Hugging Face/Ollama publication remain pending.
