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

This runtime is also the speech, perception, and local-agent integration used
by [EGG — Experimental Generalized Gateway](https://github.com/robit-man/EGG),
Robit's open-source edge-AI hardware and software platform. EGG is the larger
robot/peripheral system; this repository is the independently installable Omni
model runtime.

## Agent quick start

An automation agent starting from a clean checkout should read `AGENTS.md`,
select a model profile, bootstrap, run the deployment doctor, validate, and
only then start services. Do not copy a CUDA configuration from another host:
the launcher distinguishes broker-managed discrete GPUs from unified-memory
NVIDIA Tegra systems.

Required host tools are Python 3.10+, Git, CMake, a working NVIDIA CUDA or
Apple Metal toolchain, FFmpeg, and a running Ollama installation. Cloudflared
is optional; without it the portal remains available only on loopback.

### One-line clone and install

This command clones the repository, creates `.venv`, installs the Python
package and development checks, builds the pinned llama.cpp workers, pulls the
stock Ornith logical bundle and language backend, and validates the sidecar.
It installs but does not start the runtime:

```bash
git clone https://github.com/robit-man/qwen-omni-adapters.git && cd qwen-omni-adapters && OMNI_MODEL=robit/ornith-1.5-omni:q4km OMNI_LANGUAGE_MODEL=robit/ornith-1.5:9b ./scripts/bootstrap.sh
```

For the Qwen3.8 base, omit those two environment variables or use its explicit
profile during launch. Qwen3.8 needs substantially more memory than Ornith.
The profile variables apply to one command only; for a staged Ornith session,
export them before doctor/start as well:

### Validate and start

```bash
export OMNI_MODEL=robit/ornith-1.5-omni:q4km
export OMNI_LANGUAGE_MODEL=robit/ornith-1.5:9b
cat AGENTS.md
.venv/bin/qwen-omni doctor --deployment
./scripts/validate.sh
./portal/start.sh --daemon
./portal/start.sh --status
```

`doctor` must report the intended accelerator and no missing required
components. Validation must finish green. `start.sh --status` prints component
health and the portal URL; never publish the URL's `#access=...` fragment,
because the fragment is the portal credential.

To install managed Linux services, including the always-listening desktop
harness:

```bash
./services/linux/install.sh --with-harness
```

Use `./portal/start.sh --stop` for a foreground/staged deployment, or the
platform service manager after a service install. Do not delete
`runtime-data/components` while any worker is running.

### One-line install and launch

On a host already prepared with the prerequisites, `deploy.sh` performs any
missing bootstrap work and launches the selected verified profile:


```bash
git clone https://github.com/robit-man/qwen-omni-adapters.git && cd qwen-omni-adapters && ./deploy.sh ornith15
```

Available profiles are:

```bash
./deploy.sh qwen38                  # Qwen3.8 27B base; default
./deploy.sh ornith15                # stock Ornith 1.5 9B base
./deploy.sh ornith15-obliterated    # OBLITERATUS Ornith 1.5 9B base
```

On a broker-managed GPU host the launcher uses the scoped broker and daemonizes
the portal. On Tegra or another unmanaged NVIDIA host it uses the direct
supervisor in the foreground. A 32 GB Jetson must use the constrained residency
arrangement described below; do not attempt to keep the standard language,
comprehension, and TTS workers resident together.

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

On an arm64 NVIDIA Jetson, the launcher selects the direct supervisor because
a Tegra module has an integrated GPU and no GPU broker. That selection changes
the process lifecycle, not what fits in unified memory. A 64 GB module can use
the normal profile directly:

```bash
./deploy.sh ornith15
```

See [arm64 and NVIDIA Jetson](docs/arm-jetson.md) for the build architecture
pinning, residency evidence, and memory defaults that differ there. A 32 GB
module must use the constrained configuration below.

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

## Current implementation state

The portable/default architecture uses Qwen3-Omni for media comprehension,
the selected Qwen3.8 or Ornith base for language/vision/tools, and Qwen3-TTS
for speech. The currently validated 32 GB AGX Orin deployment uses a tighter
residency profile because all three graphs cannot safely coexist in its 29.98
GiB unified-memory pool:

| Component | Current constrained-host role | Residency |
|---|---|---|
| Qwen3-Omni + projector | Speech/audio/image/video comprehension **and** language/tool reasoning through its OpenAI-compatible endpoint | Resident; context chosen from live memory, currently validated at 16K |
| Ornith 1.5 base | Logical release/base option, but not loaded by the constrained profile | Not resident |
| Qwen3-TTS + codec projector | Final 24 kHz PCM16 speech | Loaded only after text/tools finish; exits after the utterance |
| Nomic text embedder | Passive semantic conversation memory | Admitted only while foreground, background-agent, TTS, and restoration work are idle and live headroom permits |

On that profile the harness finishes comprehension, reasoning, and tool calls
before TTS, stops the comprehension service to make room, streams decoder PCM
with a small startup lead, and starts restoring comprehension while audio is
still playing. If the user interrupts, playback ducks and then pauses/fades;
the microphone remains active throughout. This is a safe memory arrangement,
not the theoretical minimum-latency arrangement. Hosts with enough memory keep
matching TTS and language workers resident.

The essential constrained-host overrides are:

```bash
# Run Qwen3-Omni as an independently supervised worker on port 8901.
export OMNI_ENABLE_COMPREHENSION=0
export OMNI_COMPREHENSION_URL=http://127.0.0.1:8901/v1/chat/completions

# Reuse that resident worker for language instead of loading Ornith beside it.
export OMNI_LANGUAGE_API=openai
export OMNI_LANGUAGE_URL=http://127.0.0.1:8901/v1/chat/completions

# Do not retain TTS beside comprehension on a 29.98 GiB pool.
export OMNI_TTS_PERSISTENT=0
export OMNI_CALL_SPEECH_EVICT_UNIT=egg-omni-comprehension.service
export OMNI_CALL_COMPREHENSION_HEALTH=http://127.0.0.1:8901/health
```

The comprehension service uses `runtime/comprehension_launcher.py` rather
than a fixed `-c` value. Set a service `MemoryMax=` as an independent final
guard; live admission and automatic downshift remain the primary mechanism.

Current local-call behavior also includes:

- model-requested camera capture rather than transcript keyword heuristics;
- a compact discovery tool plus on-demand schemas, with raw shell available to
  the trusted local harness;
- screenshot-grounded control of a real, visible Chromium window and the full
  Ubuntu desktop, with every action followed by fresh visual evidence;
- crash-safe persistent background tasks that checkpoint after each
  inference/tool step, yield to foreground speech, accept later spoken
  guidance, and use native thinking without speaking or storing it;
- bounded conversational history with time-based relevance reduction and
  passive semantic recall that never blocks a turn;
- live memory-derived comprehension context selection and automatic
  downshifting rather than a board-specific fixed context size;
- continuous PCM playback, timing/starvation diagnostics, ReSpeaker echo
  handling, and automatic service/microphone-loop recovery.

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

## Architecture breakdown

### End-to-end data plane

```text
browser / API client                         local always-listening harness
        │                                                  │
        └──────────────────┬───────────────────────────────┘
                           ▼
              authenticated portal (optional)
              session isolation · admission queue
              tool execution · diagnostics · NDJSON relay
                           │
                           ▼
                    unified adapter API
                           │
          ┌────────────────┴─────────────────┐
          │                                  │
   text-only request                  current media request
          │                         audio · image · video
          │                                  │
          │                       validation / normalization
          │                                  │
          │                                  ▼
          │                      Qwen3-Omni comprehension
          │                                  │
          │                     tagged, untrusted evidence
          │                     ┌────────────┼────────────┐
          │                     │            │            │
          │               transcript   sound context   visual evidence
          │                     └────────────┼────────────┘
          └──────────────────────────────────┘
                                             ▼
                                  selected language backend
                              Qwen3.8 · Ornith · Qwen3-Omni
                                             │
                              ┌──────────────┴──────────────┐
                              │ unresolved structured calls?│
                              ▼                              │
                    portal tool/discovery loop ─────────────┘
                              │
                              ▼
                         final answer text
                              │
                    speech requested and safe?
                              ▼
                          Qwen3-TTS
                              │
                   ordered 24 kHz mono PCM16
```

The public model name remains the logical Ollama tag throughout. Component
URLs, process placement, model eviction, and derived GGUF views are deployment
details hidden behind the adapter.

### Stage ownership

| Stage | Owner | Important invariant |
|---|---|---|
| Request parsing | `qwen_omni_adapters.contract` | Bounds and validates media before inference; preserves native `think`, tools, options, and response modalities |
| Audio/image/video preparation | `runtime/adapter_server.py` + FFmpeg where needed | Only the newest attachment is current evidence; video duration/frame count and decoded audio are bounded |
| Media comprehension | Qwen3-Omni `llama-server` | Prompt caching is disabled so stale multimodal embeddings cannot cross turns |
| Semantic bridge | Adapter-generated tagged observation | Speech transcript, non-speech acoustics, and visual evidence stay separate and remain untrusted data |
| Language, reasoning, tool choice | Selected Ollama base or configured OpenAI-compatible worker | Reasoning remains in `message.thinking`; unresolved tool calls cannot enter TTS |
| Tool execution | Authenticated portal | Starts with compact discovery, exposes only relevant concrete schemas, records bounded evidence, and rejects repeated nonproductive calls |
| Computer use | `portal/browser.py` + `portal/gui.py` | Opens visible Chromium on the desktop, observes rendered screenshots, clicks/types through native DevTools input, and can see/control the wider workspace through `xdotool` plus fresh desktop screenshots |
| Speech | Patched Qwen3-TTS worker | Emits ordered decoder PCM; generation state is reset between prompts and never leaks one utterance into the next |
| Local conversation | `harness/` | VAD, interruption, ReSpeaker state/direction, camera capture, history, passive memory, and foreground scheduling |
| Persistent work | `harness/background_agent.py` + `portal/background_tasks.py` | Durable checkpoints and leases survive process restarts; long jobs may yield sparse spoken milestones, terminal speech is durable, and foreground speech wins every scheduling boundary |

### A normal spoken turn

1. The microphone loop continuously captures audio. Adaptive VAD accepts real
   near-end speech, joins brief continuation segments, and keeps the newest
   bounded turn rather than filling an inference queue.
2. The harness submits one audio-bearing request. Qwen3-Omni produces a tagged
   transcript and any non-speech observation; silence or a cough stops before
   language and TTS when no speech was found.
3. The configured language backend receives bounded text history, tagged
   evidence, and any already-prefetched relevant memories. The local voice
   harness requires a schema-constrained semantic dispatch result: ordinary
   reply, fresh foreground tools, or persistent background execution. On the
   constrained 32 GB profile this is the same resident Qwen3-Omni worker; a
   normal profile uses the selected Ollama base.
4. A background-execution route is written to the durable task store before an
   acknowledgment may be spoken. A fresh-evidence route gets compact discovery
   and must complete a real tool call before its answer is eligible for speech.
   Tool discovery and results use fresh request rounds, so the full catalog
   never consumes every ordinary turn's context.
5. Only final answer text is sent to TTS. PCM is played as decoder windows
   arrive, with one small initial lead to absorb packet jitter rather than
   waiting for the complete WAV.
6. Conversation persistence and embeddings are queued after answer, tools, and
   speech. They cannot delay the turn.

The local harness keeps foreground reasoning disabled by default for natural
response latency. Persistent background tasks enable native thinking because
deliberation is more valuable than sub-second response there; the thinking
channel is never synthesized or written into the task transcript.

### Camera and video flow

The browser may attach a current image or bounded clip directly. The local
harness deliberately does not attach a room image to every spoken question:
doing that biases the model into describing the scene even when the user asked
about something else. Instead, the model calls `request_camera_view`; the
harness then captures every configured V4L2 camera at the same moment, stitches
and downscales them, and performs a grounded multimodal follow-up. A motion
request captures a bounded clip. Internet uses of words such as “look up” are
therefore routed by the model to web tools rather than intercepted by a local
keyword list.

### Tools and long-horizon work

The browser portal keeps powerful tools opt-in. The trusted local voice harness
exposes the persistent `background_task` bridge beside tool discovery; its
worker, rather than the latency-critical spoken pass, owns unrestricted shell.
This structural split prevents a small model from entering a synchronous shell
retry loop and stranding the conversation. A request that inspects or mutates
files or the system, needs verification or retry, or spans multiple commands is
accepted into a crash-safe JSON store and acknowledged immediately, after which
the background worker:

```text
claim lease → reason once → execute/discover one or more tools
            → inspect results → checkpoint → yield to speech → continue
```

Shell commands return stdout, stderr, exit status, timeouts, and truncation
markers. Unsupported “done” claims are rejected until at least one concrete
action produced evidence. The worker has no fixed step horizon. On a genuinely
long task it may pause after a meaningful verified milestone, speak a short
progress explanation, then resume with the same task context. A later spoken
update is appended as authoritative task guidance before the next step,
including a final race check so stale completion cannot beat a new instruction.
Completed or blocked work ends with a natural spoken handoff only when the live
conversation is idle. Its pending-delivery flag survives a harness restart,
and the speech can be interrupted like any other reply.

Rendered computer work is not reduced to a fetched-text corpus. The
`browser_interact` tool launches a real Chromium window on the active desktop,
returns both a screenshot and a bounded accessibility/element map, performs
real pointer/keyboard input, then observes the changed page. `gui_interact`
extends the same screenshot → action → screenshot loop to the full Ubuntu
workspace. Screenshot bytes are shown to the multimodal model for one reasoning
pass and then removed from the durable transcript so long tasks retain visual
grounding without filling their context with base64. Local HTTP pages and all
desktop/file tools remain usable offline; public sites naturally require a
working network.

### Conversation state and passive memory

The adapter itself is stateless between requests. The browser owns its
cookie-isolated conversation record; the local harness owns a bounded recent
text history with time-based falloff. Current media is never replayed from that
history.

Optional durable voice memory stores completed text exchanges in SQLite and
uses a small semantic encoder rather than forcing chat-model hidden states into
an embedding role. Recall and storage run on a daemon worker. Admission checks
foreground activity, background-agent work, comprehension readiness, the
encoder's installed payload, current `MemAvailable`, and the active model's
measured KV slope. If any check fails, memory waits or the speculative recall
is skipped; conversation never waits for it.

### Context, residency, and recovery

On unified-memory hosts, `runtime/comprehension_launcher.py` reads the installed
GGUF, derives KV bytes per token, samples live available memory, and chooses the
largest standard context that fits with one adjacent context tier retained as
headroom. It records before/after residency and automatically downshifts a load
that leaves less than that model-derived reserve. The adapter reads the chosen
window per request and sheds old history/tool evidence before llama.cpp can
reject an oversized prompt.

The TTS and comprehension graphs may be simultaneously resident on a host with
enough memory. A constrained host uses explicit eviction callbacks and
non-persistent TTS. Service managers restart failed workers; the harness waits
through token rotation and adapter/comprehension restoration, reopens a failed
microphone capture loop, and resumes expired background-task leases. On Tegra,
GPU residency is proven from each process's `nvgpu`/`nvmap` handles rather than
the unsupported `nvidia-smi` process table.

### Trust and isolation boundaries

- The portal binds locally, requires a generated bearer capability, and may
  publish an authenticated Cloudflare tunnel. The URL fragment is secret.
- Browser documents, web cache, memory, location, diagnostics, and UI state are
  cookie-scoped and expire after disconnect or are removed immediately by
  Trash.
- Media text, OCR, transcripts, web pages, tool results, and model observations
  are evidence, never system instructions or authorization.
- Host snapshots exclude hostnames, addresses, routes, sockets, processes,
  credentials, and session content.
- CUDA execution has no silent CPU fallback; startup verifies the platform's
  actual GPU-residency evidence.

## Capability map

| Capability | Runtime owner | Available through this adapter |
|---|---|---|
| Text and Markdown, including responsive GFM tables | Selected Qwen3.8/Ornith Ollama base or configured Qwen3-Omni language worker | Yes |
| Structured tools | Selected language backend + portal executor | Yes |
| Portal web/document/session-memory tools | Explicit opt-in allowlisted portal loop with local-browser discovery | Yes |
| Visible Chromium interaction | Persistent rendered browser, screenshot + element evidence, click/type/scroll/back | Yes |
| Full desktop computer use | Fresh whole-desktop screenshots + coordinate/keyboard/scroll input | Yes |
| Trusted local shell | Unrestricted execution in the checkpointed voice-task worker; bounded output and timeout | Yes |
| Persistent background tasks | Checkpointed long-horizon worker with status/update/cancel, sparse spoken milestones, durable terminal speech, and restart recovery | Yes |
| Passive semantic voice memory | Idle-only encoder worker + SQLite; never gates a foreground turn | Yes |
| Thinking | Native backend `think` control, separate from answer text and never spoken | Yes |
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
cameras, copies the public link when the portal is published through a tunnel,
and cleanly reloads the voice service. It also appends live/recent durable tasks
as native submenus, so inspecting a task does not close the whole menu. Each
submenu shows the current-stage spinner, exact tools, retained checkpoints, and
terminal result. **Clear finished tasks** moves terminal records into the
human-readable archive, and **Open task archive** opens that log in the desktop
editor. A real themed state icon sits to the left of `Omni`. Without a desktop
the harness runs headless and logs instead.

Defaults are chosen for a spoken conversation:

- **Reasoning off.** A hidden chain of thought is silence the other person has
  to sit through.
- **Tools on, in the answering pass.** This is the same server-side chain used
  by the cloudflared portal: one tiny discovery schema rides with the turn,
  only the matching concrete contracts appear on the next tool round, and
  requested calls execute until the model has a grounded final answer. The
  trusted local harness also carries compact shell and persistent-task bridges.
  The full catalog no longer displaces conversation or memory context.
- **Every camera, together.** All V4L2 devices are snapped at the same moment,
  stitched into one grid and scaled down, so "what am I holding" needs no
  special mode and costs one vision pass rather than one per camera. Clips
  work the same way for questions about what just happened.
- **ReSpeaker when present.** Its ring follows the conversation and the
  direction a voice came from is attached to the turn as evidence. With no
  array attached the default microphone is used and nothing else changes.
- **Memory is passive.** Completed exchanges are embedded on a daemon worker
  only after the answer, tools and speech finish. Semantic recall is prefetched
  on that same worker as soon as a transcript exists; a completed result can
  enrich the next related turn, while an unfinished one is skipped immediately.
  It never gates hearing, answering, reasoning, tools, or speech. The
  Ornith/Omni chat weights have no embedding head and measured poorly when
  forced into that role, so the small dedicated encoder remains the deliberate
  exception and unloads after each job.
- **Long work is checkpointed.** The foreground turn can hand a sustained job
  to the persistent worker, acknowledge immediately, and keep listening. The
  worker reasons and uses tools between speech turns, records progress after
  each result, accepts spoken refinements, and reports when it completes.
- **Live host state is compact.** Every turn receives current local time,
  honest network attachment (a route is never presented as proof of internet),
  offline capability boundaries, and the fresh battery percentage/voltage from
  EGG's `/run/egg-battery/status.json` service when installed. Detailed
  hardware/load data still requires `get_system_snapshot`.

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
| `OMNI_CALL_MEMORY` | Persistent passive-memory SQLite path |
| `OMNI_CALL_SPEECH_EVICT_UNIT` | User service to stop before TTS and restore afterward |
| `OMNI_CALL_COMPREHENSION_HEALTH` | Readiness URL used after restoring that service |

Keep `OMNI_TTS_PERSISTENT=1` only when speech and comprehension genuinely fit
together. On constrained unified-memory hosts, use `OMNI_TTS_PERSISTENT=0` and
set `OMNI_CALL_SPEECH_EVICT_UNIT`: the harness completes hearing, reasoning and
tools as text, stops comprehension, synthesizes once, lets TTS exit, and restores
comprehension before listening again. This is slower than resident TTS, but it
prevents the kernel from overcommitting the machine.

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
- A wrench toggle, off by default, exposes server-pinned structured tools
  for local-browser public-web discovery/fetch, attached-document retrieval,
  current time/capabilities, on-demand host snapshots, and temporary session
  web/memory recall and isolated text-only sub-agent delegation. Productive
  tool chains continue until a final answer, while exact duplicates and
  repeated nonproductive rounds stop safely; live collapsible execution
  evidence appears in the response and phone UI. No hosted search API is used.
- Same-origin IndexedDB restores messages, drafts, pending attachments, reply
  audio, and bounded image/video previews after reload. It is keyed by a
  one-way cookie-derived scope, begins a five-minute expiry on page leave, and
  is deleted immediately by trash. Restored media is display-only and is never
  submitted automatically. The server has no shared model conversation state.
  The document index follows the same session partition and expiry policy.
- Long speech is split before the per-generation codec-frame ceiling, streamed
  with continuous sequence numbers, and assembled into one complete final WAV.
- When memory permits, Qwen3-TTS keeps a matching voice profile resident on its
  assigned GPU and emits two codec frames (about 160 ms) per stream window by
  default. A voice-profile change intentionally replaces the resident worker.
  Constrained hosts instead use a non-persistent worker and explicit residency
  handoff.
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
| `portal/` | Authenticated phone UI, proxy, supervisor, safe tools, persistent-task store, smoke tests, VAD harness |
| `harness/` | Always-listening local call mode: VAD, audio I/O, cameras, ReSpeaker, passive memory, background agent, top-bar indicator |
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

For the complete edge device, enclosure, peripherals, telepresence, and
orchestration project, see
[`robit-man/EGG`](https://github.com/robit-man/EGG).
