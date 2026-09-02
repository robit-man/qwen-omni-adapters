# Qwen Omni Adapters

Standalone runtime, protocol, and deployment tooling for the logical Ollama
model:

```text
robit/qwen3.8-27b-e03-obliterated-omni:q4km
```

The repository turns that one Ollama tag into one authenticated, Ollama-shaped
API for text, tools, optional thinking, images, audio/ASR, environmental sound
analysis, video understanding, and Qwen3-TTS speech. It also includes the
phone-first validation portal used to exercise microphone, camera, voice clone,
streamed playback, call mode, and concurrent isolated sessions.

## Start here

On the broker-managed GPU host:

```bash
git clone https://github.com/robit-man/qwen-omni-adapters.git
cd qwen-omni-adapters
./deploy.sh
```

The first run creates `.venv`, installs the Python package, clones a pinned
llama.cpp revision, applies the Qwen3-TTS PCM streaming and resident-worker
patches, builds the two
CUDA binaries, pulls missing Ollama tags, validates the attached sidecar,
materializes its disposable runtime views, starts the services, runs local
smoke gates, and prints an authenticated Cloudflare Quick Tunnel URL.

For a staged installation:

```bash
./scripts/bootstrap.sh
.venv/bin/qwen-omni doctor --deployment
./portal/start.sh --daemon
./portal/start.sh --status
./portal/start.sh --stop
```

Platform service installs are also one command after cloning:

```bash
./deploy-macos.sh                    # macOS Metal + launchd
```

```powershell
.\deploy.ps1                         # Windows CUDA + managed user task
.\deploy.ps1 -Mode Service           # true pywin32 Windows Service
```

Do not expose the URL including its `#access=...` fragment publicly. The
fragment is the portal credential.

## What the model tag contains

The release is one logical Ollama model, not one graph that stock Ollama can
execute end to end:

```text
logical Ollama tag
├── standard model/projector/template layers
│   └── Qwen3.8 text, native image vision, tools, optional thinking
└── application/vnd.robit.ollama.omni.bundle.v1+gguf
    ├── Qwen3-Omni comprehension model + projector
    └── Qwen3-TTS model + codec/projector
```

Stock Ollama handles the standard layers. This adapter resolves the custom
sidecar layer, reconstructs byte-preserving executable component views, and
runs audio/video comprehension and TTS with the pinned llama.cpp build. The
public request remains Ollama-shaped and names the one logical tag.

This is intentionally a semantic router. Qwen3.8, Qwen3-Omni, and Qwen3-TTS do
not share compatible hidden-state interfaces, so the implementation does not
pretend that their tensors can be spliced into a directly executable graph.

## Capability map

| Capability | Runtime owner | Available through this adapter |
|---|---|---|
| Text and Markdown | Qwen3.8 through Ollama | Yes |
| Structured tools | Qwen3.8 renderer/parser | Yes |
| Thinking | Native Ollama `think` boolean, off by default in portal | Yes |
| Image understanding | Qwen3.8 or Omni comprehension path | Yes |
| Speech transcription | Qwen3-Omni comprehension | Yes |
| Environmental audio interpretation | Qwen3-Omni tagged observation | Yes |
| Video understanding | Qwen3-Omni bounded `input_video` | Yes |
| Silent video and animated GIF | FFmpeg probe/normalization → Qwen3-Omni | Yes |
| PDF/DOCX/text retrieval | Session-isolated portal extraction/index | Yes |
| Spoken response | Qwen3-TTS, 24 kHz mono PCM16 | Yes |
| Voice reference cloning | Qwen3-TTS Base speaker embedding path | Yes |
| Live-call turns | Adaptive VAD + streamed text/PCM extension | Yes |
| Video generation | No component is shipped | No |

## Request example

Adapter v1 is `robit.ollama.omni-adapter.v1` and its portable route requires
`stream:false`:

```json
{
  "model": "robit/qwen3.8-27b-e03-obliterated-omni:q4km",
  "messages": [{
    "role": "user",
    "content": "What happened, and answer aloud.",
    "audios": [{
      "mime_type": "audio/wav",
      "encoding": "base64",
      "data": "<16 kHz mono PCM16 RIFF/WAVE>"
    }]
  }],
  "omni": {
    "schema": "robit.ollama.omni-adapter.v1",
    "task": "chat"
  },
  "response_modalities": ["text", "audio"],
  "speech_mode": "always",
  "think": false,
  "stream": false
}
```

Speech returns under `message.audio` as a tagged base64 RIFF/WAVE envelope.
Transcripts, non-speech acoustic observations, and visual observations remain
separate so environmental sounds are never misrouted as the user's words.

## Runtime guarantees

- Every comprehension request sets `cache_prompt:false`; a prior audio/video
  embedding cannot be reused for a new clip.
- Media turns send only the current attachment as present-tense perceptual
  evidence while retaining bounded prior text dialogue for natural continuity.
- The portal defaults to one active GPU lane and four admitted active/queued
  requests, with request-local media, tools, voice settings, and streams.
- Same-origin IndexedDB restores messages, drafts, pending attachments, reply
  audio, and bounded image/video previews after reload. It is keyed by a
  one-way cookie-derived scope, begins a five-minute expiry on page leave, and
  is deleted immediately by trash. Restored media is display-only and is never
  submitted automatically. The server has no shared model conversation state.
  The document index follows the same session partition and expiry policy.
- Long speech is split before the per-generation codec-frame ceiling, streamed
  with continuous sequence numbers, and assembled into one complete final WAV.
- Qwen3-TTS keeps a matching voice profile resident on its assigned GPU and
  emits two codec frames (about 160 ms) per stream window by default. A voice
  profile change intentionally replaces the resident worker.
- Every turn receives a fresh privacy-bounded environment snapshot containing
  date/time, OS/architecture, CPU/load, RAM, interface counters, and NVIDIA
  utilization. It excludes hostnames, addresses, routes, sockets, processes,
  credentials, and session content.
- Reasoning is off until the client sends native `think:true`. Thinking is
  returned separately and is never synthesized.
- CUDA media inference has no CPU fallback. Broker allocation and exact UUID
  residency are deployment gates on the managed host; direct NVIDIA mode also
  verifies comprehension and every TTS PID with `nvidia-smi`.
- Session diagnostics are content-redacted, partitioned by an opaque cookie,
  deleted by the trash control, and expire five minutes after a client leaves.

## Repository map

| Path | Purpose |
|---|---|
| `src/qwen_omni_adapters/` | Wire contract, audio validation, GGUF views, Ollama sidecar resolver, CLI |
| `runtime/adapter_server.py` | Unified comprehension → language → optional TTS router |
| `runtime/tts_server.py` | CUDA-only Qwen3-TTS wrapper and PCM stream endpoint |
| `portal/` | Authenticated phone UI, proxy, supervisor, smoke tests, VAD harness |
| `clients/` | Minimal Python and JavaScript request examples |
| `docs/` | Protocol, architecture, runtime, deployment, ABI, testing, release evidence |
| `patches/` | Pinned llama.cpp Qwen3-TTS streaming and persistent-worker patches |
| `scripts/` | Bootstrap, build, validation, and scoped cleanup |
| `tests/` | Contract, routing, isolation, diagnostics, GGUF, and portal regression tests |

## Documentation

- [Architecture and ownership](docs/architecture.md)
- [Agent runbook](docs/agent-runbook.md)
- [Runtime guide](docs/runtime.md)
- [Phone deployment](docs/phone-portal.md)
- [Linux, macOS, and Windows services](docs/services.md)
- [Wire protocol](docs/protocol.md)
- [GGUF/Ollama sidecar ABI](docs/gguf-abi.md)
- [Testing](docs/testing.md)
- [Cleanup and storage safety](docs/cleanup.md)
- [Security model](SECURITY.md)

The original implementation remains in
[`robit-man/fine_tuning_suite`](https://github.com/robit-man/fine_tuning_suite)
for build and model-development workflows. This repository is the smaller,
stable runtime and integration baseline.
