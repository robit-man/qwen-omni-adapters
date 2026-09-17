# Qwen Omni Adapters

Standalone runtime, protocol, and deployment tooling for logical Ollama Omni
models. Verified profiles are:

```text
robit/qwen3.8-27b-e03-obliterated-omni:q4km
robit/ornith-1.5-omni:q4km
robit/ornith-1.5-obliterated-omni:q4km
```

The repository turns that one Ollama tag into one authenticated, Ollama-shaped
API for text, tools, optional thinking, images, audio/ASR, environmental sound
analysis, video understanding, and Qwen3-TTS speech. It also includes the
phone-first validation portal used to exercise microphone, camera, allowlisted
Female/Male voice presets, request-local voice clone,
streamed playback, call mode, and concurrent isolated sessions.

For a host that should simply listen, `harness/` runs that same call mode
locally: microphone in, speakers out, state in the desktop's top bar, no
browser involved. See [Always-listening call harness](#always-listening-call-harness).

## Start here

On the broker-managed GPU host:

```bash
git clone https://github.com/robit-man/qwen-omni-adapters.git
cd qwen-omni-adapters
./deploy.sh

# Stock Ornith 1.5 9B base
./deploy.sh ornith15

# OBLITERATUS Ornith 1.5 9B base
./deploy.sh ornith15-obliterated
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

On an arm64 NVIDIA Jetson, the same command works and picks the direct
supervisor, because a Tegra module has an integrated GPU and no GPU broker:

```bash
./deploy.sh ornith15
```

See [arm64 and NVIDIA Jetson](docs/arm-jetson.md) for the build architecture
pinning, residency evidence, and memory defaults that differ there.

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
│   └── selected Qwen-family base: text, image vision, tools, optional thinking
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
For Ornith, the generic profile is stock-backed and the explicitly named
`ornith15-obliterated` profile is OBLITERATUS-backed; their base tensors and
Ollama tags are never interchangeable.

## Capability map

| Capability | Runtime owner | Available through this adapter |
|---|---|---|
| Text and Markdown, including responsive GFM tables | Qwen3.8 through Ollama | Yes |
| Structured tools | Qwen3.8 renderer/parser | Yes |
| Portal web/document/session-memory tools | Explicit opt-in allowlisted portal loop with local-browser discovery | Yes |
| Thinking | Native Ollama `think` boolean, off by default in portal | Yes |
| Image understanding | Qwen3.8 or Omni comprehension path | Yes |
| Speech transcription | Qwen3-Omni comprehension | Yes |
| Environmental audio interpretation | Qwen3-Omni tagged observation | Yes |
| Video understanding | Qwen3-Omni bounded `input_video` | Yes |
| Silent video and animated GIF | FFmpeg probe/normalization → Qwen3-Omni | Yes |
| PDF/DOCX/text retrieval | Session-isolated portal extraction/index | Yes |
| Spoken response | Qwen3-TTS, 24 kHz mono PCM16 | Yes |
| Voice reference cloning | Qwen3-TTS Base speaker embedding path | Yes |
| Live-call turns | Adaptive VAD + bounded single-flight speech consolidation + streamed text/PCM | Yes |
| Headless always-listening call mode | `harness/`: local mic/speakers, GNOME top-bar state, systemd unit | Yes |
| Every camera at once | `harness/camera.py` snaps all V4L2 devices together, stitches and downscales to one image or clip | Yes |
| ReSpeaker ring and direction | Used when the array is attached, ignored when it is not | Yes |
| Tool execution by an external loop | `GET /api/tools`, `POST /api/tools/<name>/call` | Yes |
| Video generation | No component is shipped | No |

## Always-listening call harness

`harness/` turns a host that has the model into a host you can talk to. It
drives the same endpoint and the same request shape as the portal's browser
call mode, through the machine's own microphone and speakers.

```bash
# The adapter must be running; the harness waits for it either way.
PYTHONPATH=. .venv/bin/python -m harness
```

It listens continuously, answers out loud, and shows what it is doing in the
GNOME top bar (`Omni ●` listening, `◉` hearing, `◍` thinking, `▶` speaking).
The indicator's menu mutes the microphone, toggles tools, reasoning and
cameras, and copies the public link when the portal is published through a
tunnel. Without a desktop it runs headless and logs instead.

Defaults are chosen for a spoken conversation:

- **Reasoning off.** A hidden chain of thought is silence the other person has
  to sit through.
- **Tools on**, and chained: the first answer comes back with no tool loop at
  all, so it arrives at conversational speed, and a second pass runs with the
  full tool suite and only speaks again if it actually looked something up.
- **Every camera, together.** All V4L2 devices are snapped at the same moment,
  stitched into one grid and scaled down, so "what am I holding" needs no
  special mode and costs one vision pass rather than one per camera. Clips
  work the same way for questions about what just happened.
- **ReSpeaker when present.** Its ring follows the conversation and the
  direction a voice came from is attached to the turn as evidence. With no
  array attached the default microphone is used and nothing else changes.

The speech detector is a port of `portal/static/call_vad.js` with its constants
intact, so the same room behaves the same way in the browser and here.

### Running it as a service

```bash
services/linux/install.sh --with-harness
```

That installs the adapter as a system service and the harness as a **user**
service. The distinction matters: the harness needs the desktop session it
speaks into -- its audio devices, its top bar, and the USB permissions the
logged-in user already has -- so installing it system-wide would leave it
listening on behalf of nobody.

The unit template is `services/linux/omni-call-harness.service.in`. It `Wants`
the adapter rather than requiring it, so a restart of the adapter does not take
the listener down with it, and it caps its own memory: the adapter holds the
weights, this process only moves audio.

Two environment variables are worth knowing:

| Variable | Effect |
|---|---|
| `OMNI_PORTAL_URL` | Where the portal is (default `http://127.0.0.1:8920`) |
| `OMNI_CALL_CAMERA` | A single camera to use instead of every one found |

Keep `OMNI_TTS_PERSISTENT=1` on any host used for conversation. Spawning the
speech worker per utterance costs about 25 seconds of every turn; keeping it
loaded takes that to roughly 2.

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
- A wrench toggle, off by default, exposes 24 server-pinned structured tools
  for local-browser public-web discovery/fetch, attached-document retrieval,
  current time/capabilities, on-demand host snapshots, and temporary session
  web/memory recall and isolated text-only sub-agent delegation. Tool chains have
  no numeric call or round ceiling and continue until a final answer, with exact
  duplicate no-progress detection; live collapsible execution evidence
  appears in the response and phone UI. No hosted search API is used.
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
- Ordinary turns receive only a compact stable behavioral system policy. With
  tools enabled, `get_system_snapshot` can explicitly sample current date/time,
  OS/architecture, CPU/load, RAM, interface counters, and NVIDIA utilization.
  It excludes hostnames, addresses, routes, sockets, processes, credentials,
  and session content, and describes the portal host—not the user's device.
- Call cognition is single-flight per browser call. Rapid confirmed segments
  merge into one bounded latest-turn buffer; continuing speech cancels stale
  unanswered inference and preserves its input instead of filling the server
  queue.
- Sound-only call captures stop after comprehension, render as dim **Audio
  context**, and retain at most six bounded environmental observations for the
  next actual spoken turn. They never invoke language or TTS.
- `get_user_location` uses a browser-side HTTPS lookup and exposes only
  sanitized, approximate session geography. Raw IP and network metadata never
  reach the portal or model.
- Adapter responses state current media modalities and whether current visual
  input exists. Location/search/fetch results carry tool/source authority and
  claim limits, so tool evidence cannot legitimately be presented as vision.
- Reasoning is off until the client sends native `think:true`. Thinking is
  returned separately and is never synthesized.
- CUDA media inference has no CPU fallback. Broker allocation and exact UUID
  residency are deployment gates on the managed host; direct NVIDIA mode also
  verifies comprehension and every TTS PID with `nvidia-smi`. On an NVIDIA
  Tegra module, whose driver publishes no compute-app accounting at all, the
  same gate is proven from each worker's own integrated-GPU device handles.
- Session diagnostics are content-redacted, partitioned by an opaque cookie,
  deleted by the trash control, and expire five minutes after a client leaves.

## Repository map

| Path | Purpose |
|---|---|
| `src/qwen_omni_adapters/` | Wire contract, audio validation, GGUF views, Ollama sidecar resolver, accelerator/residency probes, CLI |
| `runtime/adapter_server.py` | Unified comprehension → language → optional TTS router |
| `runtime/tts_server.py` | CUDA-only Qwen3-TTS wrapper and PCM stream endpoint |
| `portal/` | Authenticated phone UI, proxy, supervisor, smoke tests, VAD harness |
| `harness/` | Always-listening local call mode: VAD, audio I/O, cameras, ReSpeaker, top-bar indicator |
| `clients/` | Minimal Python and JavaScript request examples |
| `docs/` | Protocol, architecture, runtime, deployment, ABI, testing, release evidence |
| `patches/` | Pinned llama.cpp Qwen3-TTS streaming and persistent-worker patches |
| `scripts/` | Bootstrap, build, validation, and scoped cleanup |
| `tests/` | Contract, routing, isolation, diagnostics, GGUF, and portal regression tests |

## Documentation

- [Architecture and ownership](docs/architecture.md)
- [Agent runbook](docs/agent-runbook.md)
- [Runtime guide](docs/runtime.md)
- [Verified model profiles](docs/model-profiles.md)
- [Phone deployment](docs/phone-portal.md)
- [Linux, macOS, and Windows services](docs/services.md)
- [arm64 and NVIDIA Jetson](docs/arm-jetson.md)
- [Wire protocol](docs/protocol.md)
- [Portal tools and tool chaining](docs/tools.md)
- [GGUF/Ollama sidecar ABI](docs/gguf-abi.md)
- [Testing](docs/testing.md)
- [Cleanup and storage safety](docs/cleanup.md)
- [Security model](SECURITY.md)

The original implementation remains in
[`robit-man/fine_tuning_suite`](https://github.com/robit-man/fine_tuning_suite)
for build and model-development workflows. This repository is the smaller,
stable runtime and integration baseline.
