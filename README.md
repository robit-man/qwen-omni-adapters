# Qwen Omni Adapters

Standalone runtime, protocol, and deployment tooling for logical Ollama Omni
models. The guided Jetson deployer offers these verified reduced profiles:

```text
robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km
robit/ornith-1.5-omni-audio-bridge:q4km
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

Required host tools are Python 3.10+, Node.js, Git, CMake, a working NVIDIA
CUDA or Apple Metal toolchain, FFmpeg, and a running Ollama installation.
Cloudflared is optional; without it the portal remains available only on
loopback.

### Guided clone and install

Clone the repository and run the guided installer. On a Jetson it detects the
Tegra SoC, unified-memory size, existing runtime and managed-service state,
then presents arrow-key menus for install/upgrade, model, and local voice
harness. The Enter-through/default path always enables its desktop indicator;
core-only deployment requires the explicit `--no-harness` opt-out:

```bash
git clone https://github.com/robit-man/qwen-omni-adapters.git
cd qwen-omni-adapters
./deploy.sh
```

The installer pulls the selected logical Ollama tag, validates its trained
audio-bridge sidecar, builds or upgrades the runtime, runs doctor and regression
gates, persists the exact profile, and installs/restarts the systemd service.
The default desktop harness deployment also installs the Ubuntu
GTK/AppIndicator and PulseAudio client bindings, waits for a real top-bar
indicator, and does not declare the harness ready until it reads a microphone
frame. It does not require a separate adjacent language-model download.

### Validate and start

```bash
export OMNI_MODEL=robit/ornith-1.5-omni-audio-bridge:q4km
export OMNI_LANGUAGE_MODEL=$OMNI_MODEL
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

For non-interactive automation, the short profile names select the same two
bridge releases and deploy the managed service:


```bash
git clone https://github.com/robit-man/qwen-omni-adapters.git && cd qwen-omni-adapters && ./deploy.sh ornith15
```

Available profiles are:

```bash
./deploy.sh ornith15                # standard Ornith 1.5 9B bridge; about 8.15 GiB
./deploy.sh qwen38                  # Qwen3.8 27B E03 bridge; about 18.33 GiB
```

On a broker-managed GPU host the installed service uses the scoped broker. On
Tegra it uses the direct supervisor and proves residency from `nvgpu`/`nvmap`
handles. The Ornith bridge is the recommended 32 GB Jetson choice. Qwen's
18.33 GiB artifact set fits nominally, but an eviction-free production peak is
not claimed until measured on the target board with its real context and TTS
policy.

Upgrades perform a live-runtime handoff before replacing the unit: the
deployer identifies and stops recognized old Omni listeners, unloads their
relevant Ollama runners, verifies the ports are free, and samples Jetson GPU
load and unified-memory headroom. The old configuration, unit, and managed
services are restored if the replacement does not reach ready state.

The first run creates `.venv`, installs the Python package, clones a pinned
llama.cpp revision, applies the Qwen3-TTS PCM streaming and resident-worker
patches, builds the two
CUDA binaries, pulls the selected Ollama tag, validates the attached sidecar,
materializes its disposable TTS views, installs the managed service, runs local
smoke gates, and records the authenticated portal URL in the protected daemon
status.

For a staged installation:

```bash
./scripts/bootstrap.sh
.venv/bin/qwen-omni doctor --deployment
./portal/start.sh --daemon
./portal/start.sh --status
./portal/start.sh --stop
```

On an arm64 NVIDIA Jetson, the launcher selects the direct managed service
because a Tegra module has an integrated GPU and no GPU broker. Run without a
profile to choose interactively:

```bash
./deploy.sh
```

See [arm64 and NVIDIA Jetson](docs/arm-jetson.md) for build architecture
pinning, residency evidence, and unified-memory guidance.

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

The guided deployment uses a trained audio bridge: the selected Qwen3.8 or
standard Ornith trunk handles audio/ASR, native vision, language, and tools in
one llama.cpp server, while Qwen3-TTS provides speech. Legacy full-Omni bundles
remain supported and use Qwen3-Omni for media comprehension plus a selected
language base. The previously validated legacy 32 GB AGX Orin deployment uses
a tighter residency profile because all three graphs cannot safely coexist in
its 29.98 GiB unified-memory pool:

| Component | Current constrained-host role | Residency |
|---|---|---|
| Qwen3-Omni + projector | Speech/audio/image/video comprehension **and** language/tool reasoning through its OpenAI-compatible endpoint | Resident; context chosen from live memory and published to the adapter; post-speech recovery live-validated at 4K and 8K under desktop load |
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
  bounded inference/tool slice, finalize only through referenced tool evidence,
  yield cancellable inference to foreground speech, accept later spoken
  guidance, and use native thinking without speaking or storing it;
- bounded conversational history with time-based relevance reduction and
  explicit current-query memory tools rather than next-turn prefetch;
- live memory-derived comprehension context selection and automatic
  downshifting rather than a board-specific fixed context size;
- continuous PCM playback, timing/starvation diagnostics, ReSpeaker echo
  handling, and automatic service/microphone-loop recovery.
- a resident, typed Laya System-1 decision plane with batched routing,
  pre-action, post-action, and context-relevance waves; it is shadow-only until
  real calibration proves a per-family fast path, and failures always preserve
  the deliberative route. See [`docs/decision-plane.md`](docs/decision-plane.md).

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

The runtime also accepts the reduced `robit.ollama-audio-bridge.v1` profile.
There the standard projector combines target-native vision with the frozen
Omni audio encoder and its trained final projection. The sidecar carries only
TTS, and one local llama.cpp server is both the multimodal comprehension path
and the sole language/tool trunk. This removes the full secondary Omni Thinker
and avoids loading an adjacent Ollama language copy.

Legacy full-Omni tags remain semantic routers because Qwen3.8, Qwen3-Omni, and
Qwen3-TTS do not share compatible hidden-state interfaces. The trained bridge
profiles are different: they contain the frozen Omni audio encoder plus a
trained final projection into the selected trunk, while keeping evidence tags
and TTS as explicit runtime boundaries. Standard Ornith and Qwen3.8 E03 tensors
and Ollama tags are never interchangeable.

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
| Local conversation | `harness/` | VAD, interruption, ReSpeaker state/direction, camera capture, history, deferred memory writes, and foreground scheduling |
| Persistent work | `harness/background_agent.py` + `portal/background_tasks.py` | Durable checkpoints and leases survive process restarts; long jobs may yield sparse spoken milestones, terminal speech is durable, and foreground speech wins every scheduling boundary |

### A normal spoken turn

1. The microphone loop continuously captures audio. Adaptive VAD accepts real
   near-end speech, joins brief continuation segments, and keeps the newest
   bounded turn rather than filling an inference queue.
2. The harness submits one audio-bearing request. Qwen3-Omni produces a tagged
   transcript and any non-speech observation; silence or a cough stops before
   language and TTS when no speech was found.
3. The configured language backend receives the recognized speech as the latest
   user message, with bounded text history and non-speech observations as
   secondary tagged evidence. Empty negative acoustic boilerplate such as “no
   non-speech sounds” remains available in adapter diagnostics but is omitted
   from the language prompt so it cannot contradict a valid transcript. Visual evidence is added only after a structured
   camera request, or when the client explicitly attached media. The
   local voice harness enables the portal's real tool allowlist on every turn
   without a separate classification pass: the model either answers
   conversationally or calls the smallest tool that accomplishes the request,
   the portal auto-executes it, and the spoken answer is the grounded reply that
   follows the completed work. On the constrained 32 GB profile this is the same
   resident Qwen3-Omni worker; a normal profile uses the selected Ollama base.
4. A durable-task request goes through the same pass via the `background_task`
   tool, which writes the objective to the durable store before any
   acknowledgment. Camera-enabled embodied turns expose a capture bridge but do
   not attach an ambient still unless the request needs current physical-scene
   evidence; web and other current-information requests complete a real tool
   call in the pass. No answer is eligible for speech before the work it claims
   has actually completed. Once transcript-aware routing finds concrete matching
   schemas, the language trunk receives their names in a compact required-action
   context and still chooses the tool and arguments itself.
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
keyword list. The grounded follow-up carries the fresh visual evidence but no
second camera bridge, preventing a recapture loop. The pre-capture placeholder
is never spoken, logged as generated dialogue, or kept in conversation history;
the grounded pass is the sole answer. Broad casual questions get a short casual
overview, while questions about a particular item or feature stay focused on
that target instead of inventorying the rest of the scene.

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
Typed focus records advertise no paging operation while their detailed receipts
are still resident. After compaction they gain structured pointers to the
separate `task_expand` control function; that control name is never presented as
an action or argument of `workspace_file` or another external tool. The focus
ledger is a live working set, not another copy of history: its size follows the
currently resident comprehension window, repeated inspections and failures are
coalesced, a changed artifact supersedes its older resident version, and omitted
record counts remain visible. Exact receipts and every superseded version remain
append-only in task evidence storage. A compacted request carries this ledger
once in the pinned system contract rather than duplicating it in a recurrent
checkpoint. If an omitted record's evidence ID is no longer resident,
`task_expand` also accepts a distinctive exact path, URL, symbol, error, or
other query and returns the highest-scoring immutable receipts.
The ledger applies the same evidence authority as checkpoints: search results
marked discovery-only never become acquired-source successes, and an empty
`about:blank` browser snapshot remains an inspection. A blank visible browser
explicitly requires `navigate` to a grounded URL and cannot divert the next
round into pixel clicking or unrelated capability discovery.
At the 4K/8K tiers the worker also projects the already-selected tool schema to
its executable JSON constraints: names, types, enums, required fields, bounds,
and `additionalProperties` remain exact while repeated prose annotations are
removed. Only one selected concrete capability is exposed per action round;
discovery is suppressed until that leaf receives one concrete attempt, then
returns as the route to a different capability. At constrained tiers discovery
and evidence expansion alternate instead of crowding the same envelope. A
background request that still cannot fit the live pack fails explicitly and retries from
its checkpoint—it never silently falls back to native FIFO truncation.
Discovery queries describe only the missing interaction mechanism (for example,
public-web search, file editing, shell execution, or visible-browser control),
not the task topic. The deterministic router also recognizes product/SaaS,
comparison, interface, and design research as web discovery, preventing a word
such as “service” in the subject from accidentally selecting system shell.
Background web fetches are provenance-bound as well: a target must appear in
the accepted task/user input or prior search, browser, crawl, or fetch evidence.
An invented address is rejected before network access and returned beside the
exact admissible URLs so the next model step can choose a grounded source.
The pinned task envelope also carries an explicit execution frontier: work
advances the earliest unmet prerequisite in the user's stated order, and a
failed downstream probe returns the worker to that prerequisite instead of
encouraging variants of the same premature verification.
Completed or blocked work ends with a brief natural spoken status only when the
live conversation is idle. Detailed reports, evidence IDs, paths, and worker
self-assessment stay in the indicator and task archive and are never passed
verbatim to TTS. The pending-delivery flag survives a harness restart, and the
speech can be interrupted like any other reply. A direct `synthesize` request
is a literal text-to-speech transport pass: it retains the selected voice-clone
profile but bypasses conversation policy, tools, documents, and virtual-memory
indexing/replay so recalled text cannot be appended to the utterance.

Rendered computer work is not reduced to a fetched-text corpus. The
`browser_interact` tool launches a real Chromium window on the active desktop,
returns both a screenshot and a bounded accessibility/element map, performs
real pointer/keyboard input, then observes the changed page. `gui_interact`
extends the same screenshot → action → screenshot loop to Ubuntu. Its
default screenshot is the active window and its coordinates are relative to
that image. The runtime captures root-window pixels and crops them to the exact
X11 geometry used for pointer translation, so window-manager decorations cannot
offset clicks. Full-screen mode is explicit for panels and workspace navigation;
when a later pointer call omits its coordinate space, it remains bound to the
newest returned frame. Active-window clicks fail safely if focus changed after
observation. Every result also reports whether the coarse visual state materially
changed, which lets the worker reject a missed click as non-progress.
Chromium's rendered network-error documents remain visible evidence, but are
tagged as failed navigation and release the sticky browser action space; a
connection-refused page can never count as successful GUI verification.
Actionable DOM controls are always re-resolved through live CDP geometry. For
canvas, challenge, and image targets on Jetson, the conversational vision pass
supplies a concise referring expression and an isolated resident Moondream 2
point head resolves it against the exact CDP viewport. The browser re-captures
and compares that viewport after point inference before sending input, so a
coordinate is never carried onto changed pixels. Multiple returned matches are
disambiguated by the conversational model's coarse current-frame point; the
point head's structured coordinate remains the executed value. If the point
head is not configured, the existing bounded crop-refinement loop remains the
fail-safe fallback rather than widening hit tolerances.
With the point head active, every visual click requires a concise target phrase.
If the planner's coarse crop misses, the executor searches the same horizontal
band in bounded overlapping tiles before widening to a bounded tile grid; the
point head still selects every executable coordinate.
An explicit verification snapshot or completed visual click also asks that same
resident visual worker to transcribe visible status text and exact completion
identifiers from the exact returned CDP frame. That bounded reading is stored as
current-frame evidence, while a compact action receipt preserves the exact
grounded target, so terminal state and action reporting survive image compaction
without relying on an invented marker or renamed target.
Screenshot bytes are shown to the multimodal model for one reasoning
pass and then removed from the durable transcript so long tasks retain visual
grounding without filling their context with base64. Local HTTP pages and all
desktop/file tools remain usable offline; public sites naturally require a
working network.

### Conversation state and durable memory

The adapter itself is stateless between requests. The browser owns its
cookie-isolated conversation record; the local harness owns a bounded recent
text history with time-based falloff. Current media is never replayed from that
history.

Optional durable voice memory stores completed text exchanges in SQLite and
uses a small semantic encoder rather than forcing chat-model hidden states into
an embedding role. Storage runs on a daemon worker; retrieval is an explicit
tool call against the current request, never a result automatically carried
from one turn into the next. Admission checks
foreground activity, background-agent work, comprehension readiness, the
encoder's installed payload, current `MemAvailable`, and the active model's
measured KV slope. If any check fails, deferred storage waits; conversation
never waits for it.

### Context, residency, and recovery

On unified-memory hosts, `runtime/comprehension_launcher.py` reads the installed
GGUF, derives KV bytes per token, samples live available memory, and chooses the
largest context tier that fits with the runtime memory reserve. Compact models
may have a 262,144-token native positional range, but the managed compact
profiles intentionally expose a 16,384-token resident working-set ceiling.
The guided Ornith-on-Tegra profile uses its live-qualified q8 KV cache; other
model/platform pairs retain f16 until they pass the same answer, multimodal,
voice, tool, and memory-pressure gates. The configured 16K value is a ceiling:
a longer live action soak selected 8K and then 4K as the rest of the resident
stack consumed unified memory. The live tier is authoritative.
Longer history is paged through the lossless virtual-context layer instead of
preallocating a nominal 256K KV cache. First load uses the conservative complete
component-byte footprint; later loads also use measured residency. It records
before/after residency and automatically caps the next load below a tier that
exits or leaves too little memory. The adapter reads the chosen window per
request and sheds old history/tool evidence before llama.cpp can reject an
oversized prompt.

On discrete-memory hosts, sampling continues after readiness: a continuous
low-headroom interval downshifts one tier, and sustained surplus performs the
inverse only while every inference slot is idle and the cooldown has elapsed.
On Tegra, the startup-selected llama.cpp process stays pinned for the service
session because affected JetPack kernels can panic while closing GPU character
devices during an otherwise controlled worker restart. Prompt budgets remain
dynamic inside the allocated KV window; new work is admission-gated, and the
next supervised start reselects its tier from live memory. This avoids process
churn without reverting to a board-specific context limit.

The trained-bridge runtime keeps TTS and comprehension simultaneously resident,
but guided deployment does not block the desktop on generation probes. It marks
the core ready after local component health and starts the indicator service as
soon as the core unit starts. The full ASR/cloned-TTS/co-residency smoke remains
available as an explicit diagnostic. Legacy/manual constrained profiles may
still opt into explicit eviction callbacks and non-persistent TTS. Service
managers restart failed workers; the harness waits
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
| Portal web/document/session-memory tools | Explicit opt-in allowlisted portal loop with no-key DuckDuckGo HTML discovery | Yes |
| Visible Chromium interaction | Persistent rendered browser, screenshot + element evidence, click/type/scroll/back | Yes |
| Full desktop computer use | Fresh active-window screenshots with image-relative input; explicit full-screen workspace mode | Yes |
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
| Always-listening local call mode | `harness/`: local mic/speakers, mandatory GNOME top-bar state for its managed desktop unit | Yes |
| Every camera at once, on request | `harness/camera.py` snaps all V4L2 devices together, stitches and downscales to one image or clip only after a visual-evidence request | Yes |
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
opens a loopback-only live view of every camera stitched left-to-right, and
cleanly reloads the voice service. It polls `origin/main` in the background and
shows an **Update available — install and restart** action when a verified
fast-forward exists. Clicking it preserves untracked local files, refuses
tracked edits or diverged history, refreshes the runtime environment without
implicitly downloading or changing model weights, and asks the managed runtime
and indicator to restart on the new revision. Its
**Models** submenu lists both compact
audio bridges and both legacy full bundles. Missing tags expose **Download**
with live percentage/status text; downloaded tags expose **Activate**, **Load
into Ollama**, **Unload from Ollama**, and confirmed **Delete local copy**
actions as applicable. Activation atomically selects the one logical language/
Omni tag, requests a controlled core restart, and lets the indicator reconnect
with the new service environment. An active direct-daemon model is labelled
separately from an optional Ollama runner so the UI never hides a duplicate
allocation on a 32 GB Jetson. It also appends live/recent durable tasks
as native submenus, so inspecting a task does not close the whole menu. Each
submenu shows the current-stage spinner, exact bounded tool-call arguments and
outcomes, retained checkpoints, and terminal result. Long action rows wrap and
ellipsize within the menu while preserving the complete text in their tooltip,
so tool arguments cannot widen the indicator beyond the screen. Live tasks expose
**Cancel task** and every record exposes **Clear task record**. **Clear finished
tasks** moves all terminal records into the human-readable archive, and **Open
task archive** opens that log in the desktop editor. A real themed state icon
sits to the left of `Omni`. A manual invocation may run headless and log instead.
The managed desktop service fails closed if GTK, the StatusNotifier host, the
desktop audio server, or a usable microphone/sink is unavailable; it never
silently leaves an always-listening service running without its indicator.

Defaults are chosen for a spoken conversation:

- **Reasoning off.** A hidden chain of thought is silence the other person has
  to sit through.
- **Tools on, in the answering pass.** This is the same server-side chain used
  by the cloudflared portal: one tiny discovery schema rides with the turn,
  only the matching concrete contracts appear on the next tool round, and
  requested calls execute until the model has a grounded final answer. The
  trusted local harness also carries compact shell and persistent-task bridges.
  The full catalog no longer displaces conversation or memory context.
- **Every camera together, only when relevant.** The audio-only pass may call
  `request_camera_view` when the answer depends on the current physical scene.
Only then are V4L2 devices opened and probed, all working cameras are snapped
at the same moment, and their frames are stitched into one left-to-right row.
The indicator's explicit live-view action uses the same horizontal composition. Clips
work the same way for temporal questions. Merely starting the harness never
launches FFmpeg or activates a camera privacy indicator.
- **Observation does not force speech.** Sound-only events are retained as
  bounded context without language or TTS. The language model may also leave a
  transcribed room utterance unanswered when current evidence shows it was
  addressed elsewhere; if gaze would resolve genuine ambiguity, it can request
  a fresh still before deciding. Empty intentional responses do not invoke TTS.
- **ReSpeaker when present.** Its ring follows the conversation and the
  direction a voice came from is attached to the turn as evidence. With no
  array attached the default microphone is used and nothing else changes.
- **Memory storage is passive.** Completed exchanges are embedded on a daemon
  worker only after the answer, tools and speech finish. Recall uses an explicit
  current-query portal tool, so a result selected for one utterance cannot
  contaminate the next one. Storage never gates hearing, answering, reasoning,
  tools, or speech. The
  Ornith/Omni chat weights have no embedding head and measured poorly when
  forced into that role, so the small dedicated encoder remains the deliberate
  exception and unloads after each job.
- **Long work is checkpointed and self-checked.** The foreground turn can hand a sustained job
  to the persistent worker, acknowledge immediately, and keep listening. The
  worker reasons and uses tools between speech turns, reassesses objective
  alignment and remaining criteria after every concrete result, records
  evidence-backed progress, accepts spoken refinements, and reports only after
  the freshest result supports completion or a real blocker.
- **Live host state is explicit.** Ordinary turns carry no eager clock,
  location, network, battery, or process blob. Current time, client-browser
  approximate location, and bounded hardware/load facts come from their
  dedicated tools only when the request needs them. The local voice client
  warms the same privacy-filtered browser lookup while the model loads.

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

The unit template is `services/linux/omni-call-harness.service.in`. It is a
desktop-session user unit, while the adapter is a system unit, so it waits for
the loopback portal rather than declaring an invalid cross-manager dependency.
Its preflight runs in the exact user-service environment and requires the real
indicator and audio stack. The unit caps its own memory: the adapter holds the
weights, this process only moves audio.

The core daemon's optional smoke uses a tracked speech fixture and requires a
tagged transcript, a direct ASR-to-cloned-TTS route using the shipped default
speaker reference, valid 24 kHz mono PCM16 output, the normal streamed TTS gate,
and another tagged-ASR pass after speech. It then proves that the original
comprehension PID and persistent clone-profile TTS PID remain GPU-resident
together. A generic sound observation or unconditioned WAV cannot satisfy the
gate. Set `OMNI_STARTUP_SMOKE=1` only when this blocking diagnostic is wanted.
Guided deployment sets it to `0`, starts the indicator in parallel with core
readiness, and does not wait for inference. On Jetson diagnostic requests run
against the installed arm64/CUDA workers;
desktop-host unit tests do not substitute for that device gate.

After deployment, run the non-blocking post-training tool-routing suite
explicitly without delaying normal boot:

```bash
.venv/bin/python portal/smoke.py \
  --endpoint http://127.0.0.1:8920 \
  --token-file runtime-data/state/access-token.txt \
  --model "$(awk -F= '$1 == "OMNI_MODEL" {print $2}' .env)" \
  --tool-suite
```

It requires real structured calls for portal capabilities, arithmetic, current
runtime state, and time; a prose capability disclaimer does not pass.

These environment variables are worth knowing:

| Variable | Effect |
|---|---|
| `OMNI_PORTAL_URL` | Where the portal is (default `http://127.0.0.1:8920`) |
| `OMNI_CALL_CAMERA` | A single camera to use instead of every one found |
| `OMNI_CALL_MEMORY` | Persistent passive-memory SQLite path |
| `OMNI_CALL_SPEECH_EVICT_UNIT` | User service to stop before TTS and restore afterward |
| `OMNI_CALL_COMPREHENSION_HEALTH` | Readiness URL used after restoring that service |
| `OMNI_MEMORY_GOVERNOR` | Enable (`1`) or disable (`0`) generic runtime memory admission; enabled automatically on Tegra |
| `OMNI_MEMORY_SOFT_FLOOR_GIB` | Free-memory floor retained before work establishes new model/tool residency (default `3`) |
| `OMNI_MEMORY_HARD_FLOOR_GIB` | Emergency floor that cancels cancellable work before kernel OOM (default `2`) |
| `OMNI_MEMORY_OPERATION_RESERVE_GIB` | Additional per-operation reserve above the soft floor (default `1`) |
| `OMNI_COMPREHENSION_PRESSURE_GRACE_SECONDS` | Continuous low-headroom interval before a context downshift (default `8`) |
| `OMNI_COMPREHENSION_RUNTIME_RESIZE` | Restart llama.cpp to change its allocated KV tier at runtime; defaults off on Tegra to avoid unsafe JetPack GPU-device teardown and on elsewhere |
| `OMNI_COMPREHENSION_EXPANSION_GRACE_SECONDS` | Continuous idle-surplus interval before a one-tier expansion (default `60`) |
| `OMNI_COMPREHENSION_EXPANSION_COOLDOWN_SECONDS` | Minimum delay after a failed/pressured tier before retrying it (default `900`) |
| `OMNI_CONTEXT_FILE` | Optional complete `robit.omni.context.v1` catalog override; defaults to the packaged context catalog |
| `OMNI_CALL_LOG_CONTENT` | Opt in to exact structured heard/generated/TTS traces; disabled by default |
| `OMNI_UPDATE_INTERVAL_SECONDS` | Indicator Git update polling interval; minimum 60 seconds, default 900 |

Keep `OMNI_TTS_PERSISTENT=1` only when speech and comprehension genuinely fit
together. On constrained unified-memory hosts, use `OMNI_TTS_PERSISTENT=0` and
set `OMNI_CALL_SPEECH_EVICT_UNIT`: the harness completes hearing, reasoning and
tools as text, stops comprehension, synthesizes once, lets TTS exit, and restores
comprehension before listening again. This is slower than resident TTS, but it
prevents the kernel from overcommitting the machine.

Tool resource admission is declared beside each tool in `context.json`.
Standard work must clear the soft floor, bounded continuations may run within
the soft-to-hard safety band, executors dynamically distinguish new residency
from reuse, and control-plane work remains available so stalled work can be
inspected or cancelled. Every cancellable operation still stops at the hard
floor. An executor may declare a measured `memory_reserve_gib`; otherwise its
new residency uses the conservative standard admission boundary.

## Request example

Adapter v1 is `robit.ollama.omni-adapter.v1` and its portable route requires
`stream:false`:

```json
{
  "model": "robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km",
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
- Trained-bridge runtime keeps the matching shipped Qwen3-TTS voice profile
  resident alongside comprehension on its
  assigned GPU and emits two codec frames (about 160 ms) per stream window by
  default. A voice-profile change intentionally replaces the resident worker.
  Non-persistent workers and explicit residency handoff remain legacy/manual
  escape hatches. Guided startup does not run a blocking generation gate.
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
| `harness/` | Always-listening local call mode: VAD, audio I/O, cameras, ReSpeaker, deferred memory storage, background agent, top-bar indicator |
| `clients/` | Minimal Python and JavaScript request examples |
| `docs/` | Protocol, architecture, runtime, deployment, ABI, testing, release evidence |
| `patches/` | Pinned llama.cpp Qwen3-TTS streaming and persistent-worker patches |
| `scripts/` | Bootstrap, build, validation, and scoped cleanup |
| `tests/` | Contract, routing, isolation, diagnostics, GGUF, and portal regression tests |

## Documentation

- [Virtual context](docs/virtual-context.md) — lossless long-history storage,
  recursive retrieval, exact replay, and the bounded 16K working-set contract.

- [Architecture and ownership](docs/architecture.md)
- [Agent runbook](docs/agent-runbook.md)
- [Runtime guide](docs/runtime.md)
- [Context engineering and tool/phase map](docs/context-engineering.md)
- [Verified model profiles](docs/model-profiles.md)
- [Phone deployment](docs/phone-portal.md)
- [Linux, macOS, and Windows services](docs/services.md)
- [arm64 and NVIDIA Jetson](docs/arm-jetson.md)
- [Wire protocol](docs/protocol.md)
- [Portal tools and tool chaining](docs/tools.md)
- [GGUF/Ollama sidecar ABI](docs/gguf-abi.md)
- [Testing](docs/testing.md)
- [Deferred component candidates](docs/candidate-components.md)
- [Cleanup and storage safety](docs/cleanup.md)
- [Security model](SECURITY.md)

The original implementation remains in
[`robit-man/fine_tuning_suite`](https://github.com/robit-man/fine_tuning_suite)
for build and model-development workflows. This repository is the smaller,
stable runtime and integration baseline.

For the complete edge device, enclosure, peripherals, telepresence, and
orchestration project, see
[`robit-man/EGG`](https://github.com/robit-man/EGG).
