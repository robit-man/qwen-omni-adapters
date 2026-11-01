# Native Ollama audio integration

This experimental overlay adds native ASR and Qwen3-TTS Base execution to
**Ollama 0.40.13, commit04088e7c9d41cba5d33bafbf788d87a465c79d5b**.
It uses Ollama's own `/v1/audio/transcriptions` and `/v1/audio/speech` handlers
and scheduler. Generation runs in native C++ workers. No Python inference server,
second language model, additional neural download or replacement TTS is used.

This is a **patched Ollama build**. Unmodified Ollama's audio endpoint support
does not load our custom Qwen3-TTS sidecar or enforce the trained audio framing.
The existing adapter and its request contract remain available. Edge v2 weights
are not published: environmental/temporal and natural-speech parity remain failed.

## Verified mechanism

Each scoped A100 test used one private Ornith or Qwen tag with the existing indexed TTS
sidecar and its exact-bound conditional FFN adapter. Two actual native ASR calls
returned the expected speech and empty silence transcript. Five streamed speech
calls per backbone matched the existing C++ runtime **byte for byte**, covering female/male,
A→B→A and request-local WAV plus transcript/acoustic conditioning. One primary
PID remained resident; all generation/reference workers were proven on GPU0
through the central accelerator module. The isolated server exited and its lease
released normally. No audio files or public weights were created.

See the exact [Ornith receipt](../../docs/upgrades/native-ollama-ornith-mechanism-20261101.json)
and [Qwen receipt](../../docs/upgrades/native-ollama-qwen-mechanism-20261101.json).
This gate does not prove Orin compatibility, complete WAV handling,
cancellation, language coverage, perception quality, speaker similarity, all
sampling/profile controls, or performance improvements. Those remain release work.

Build `0.40.13-robit-omni.2` also passes the bounded native sampling gate on
Ornith/A100. Default controls, explicit zero filters and the maximum supported
seed produce exactly the existing runtime's PCM; random seed produces valid
PCM. Invalid controls return HTTP 400 before a model is scheduled. Bounds are
temperature 0–2, top_k 0–1000, top_p 0–1 and seed −1–2147483647, including explicit
zeros. Twelve cases derived from the current runtime parser, nonfinite-number
checks and pre-scheduling handler tests pass. See the
[sampling receipt](../../docs/upgrades/native-ollama-sampling-parity-20261101.json).
This does not replace the pending full native or assigned-target matrix.

Build `0.40.13-robit-omni.3` adds a bounded RAM cache for full reference
conditioning. The key includes reference content, transcript, bundle/TTS/encoder
identity, normalization version, language and conditioning algorithm. Each
loaded configuration retains at most eight entries and 1 MiB of JSON payload,
with ten-minute expiry. Each request still uses fresh generation state; request
files are removed after synthesis and cached conditioning is never written as a
persistent profile.

The matched Ornith/A100 gate compared ten requests per build. All ten PCM
outputs matched the uncached build exactly, including changed references,
transcripts and language. Six cache hits reduced median first audio from
2.186 s to 1.486 s (32.0%); reference encoder launches fell from ten to four.
Both private servers and all CUDA workers exited normally. This is a bounded
mechanism and latency result, not full voice-quality, Orin or Qwen qualification.
See the [conditioning-cache receipt](../../docs/upgrades/native-ollama-conditioning-cache-20261101.json).

## Build and prepare

Bootstrap the normal standalone checkout and build its pinned native engines.
The optional Ollama build pins and verifies the official source archive, applies
only this overlay, runs Go runner/server tests and keeps build/module caches in
the selected build directory:

```bash
.venv/bin/python scripts/build_native_ollama.py
```

Use `--cuda-backend cuda_v12` for a JetPack6 build. The existing `go` command must
be available; Go downloads the pinned1.26.10 toolchain when needed. No shared
Ollama binary or service is overwritten. The build's CUDA visibility is disabled.

After a qualified indexed tag is installed locally, prepare its exact binding:

```bash
.venv/bin/python scripts/prepare_native_ollama.py \
  --model '<installed-indexed-tag>' \
  --models-dir '<local-Ollama-model-store>' \
  --config-dir '<private-native-configuration-directory>'
```

This reuses the existing sidecar validator, GGUF/conditional-adapter checks,
native prefill ABI and exact training prompt. The digest-addressed configuration
hashes the model, projector, sidecar, optional audio adapter, native binaries,
libraries, default reference voices and accelerator proof helper. No weights
are materialized or copied. Creation is exclusive; an installed binding is not
overwritten. The native runtime revalidates artifact identities.

Run `runtime-data/native-ollama/ollama-native serve` only through the host's
normal scoped CUDA admission, with `OLLAMA_OMNI_CONFIG_DIR` pointing to that
configuration directory and `OLLAMA_MODELS` to the matching model store. Bind to
loopback and preserve the existing authenticated front door. On broker hosts,
reserve the complete primary plus auxiliary budget, register the supervising
process and verify actual workers. Never start this as an anonymous CUDA process
or as an additional reasoning runner beside an existing deployment.

The scheduler reserves8GiB of auxiliary headroom per native Omni runner, separately
from actual resident-memory reports. This is conservative admission policy,
not measured resident memory or a qualified32GB profile. Native workers inherit
exactly the admitted GPU UUID; the central accelerator proof rejects CPU fallback.

## Native requests

ASR accepts Ollama's native multipart request with `model` and WAV `file`.
It splits at30seconds and uses the exact trained prefix/suffix, explicit
`cache_prompt:false` and acoustic-only adapter scale1. Only the tagged transcript
becomes ASR text. Malformed/truncated evidence fails rather than becoming a
fabricated transcript. Translation and custom extraction prompts are rejected.
Ordinary text/vision/reasoning/tool requests explicitly use adapter scale0.

Speech accepts the standard native `model`, `input`, `voice` and
`response_format` (`pcm` or `wav`). The default references are `female` and `male`.
The optional **native-only** `omni` object carries `speaker_audio` (the existing
base64 WAV envelope), `reference_text`, `language`, `temperature`, `top_k`,
`top_p`, `seed` and `max_frames`. This does not alter adapter-v1's schema.

```json
{
  "model": "<installed-indexed-tag>",
  "input": "Hello there.",
  "voice": "female",
  "response_format": "pcm",
  "omni": {
    "speaker_audio": {
      "mime_type": "audio/wav",
      "encoding": "base64",
      "data": "<reference-WAV-base64>"
    },
    "reference_text": "The words spoken in the reference.",
    "language": "en",
    "seed": 42
  }
}
```

Reference bytes and derived acoustic codes live in a private temporary directory
and are removed after the request. Full reference conditioning uses the same
bundled native reference encoder and Base decoder; it is not reduced to a speaker
embedding. Request cancellation terminates the native child. Immutable reference
conditioning may be reused from the bounded per-configuration RAM cache.
Complete profile import, CLI voice listing, broader error/status handling and
the full release matrix still require work.
