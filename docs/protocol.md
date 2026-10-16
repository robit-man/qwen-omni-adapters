# Omni Adapter v1 Wire Protocol

`robit.ollama.omni-adapter.v1` is an additive extension to Ollama's non-streaming
`POST /api/chat` request. It defines binary media envelopes, task routing, and
speech output while preserving normal language, thinking, and tool behavior.

## Normative language

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY describe requirements for
interoperable implementations. The executable authority for validation is
`qwen_omni_adapters.contract`; the JSON schemas provide a portable
structural description.

## Endpoint and transport

- Endpoint: `POST /api/chat`
- Content type: `application/json`
- Request body: UTF-8 JSON object
- Streaming: v1 MUST use `"stream": false`
- Inline binary: standard padded base64 without whitespace
- Maximum decoded payloads: 32 MiB per audio item, 20 MiB per image, and
  256 MiB per video
- Maximum items per request: 8 audio, 16 image, and 4 video items

Deployments MAY set lower limits. They MUST return a clear 4xx error instead of
silently truncating media.

## Request fields

### Preserved Ollama fields

The adapter passes these fields to the language stage when `omni.task=chat`:

| Field | Behavior |
|---|---|
| `model` | Selects the combined Ollama tag and is required |
| `messages` | Preserves roles, text, tool-call history, and tool results |
| `tools` | Passed unchanged to the language stage |
| `think` | Passed unchanged; `thinking` remains separate in the response |
| `format` | Passed unchanged |
| `options` | Passed unchanged |
| `keep_alive` | Passed unchanged |
| `logprobs`, `top_logprobs` | Passed unchanged when supported by the runtime |

An implementation MUST NOT include base64 media in the language prompt. It
passes media only to the comprehension graph and injects its semantic result as
clearly delimited, untrusted observation text.

For `chat`, the comprehension graph is a perception encoder, not a second
assistant. It MUST NOT answer the user. Audio-bearing observations SHOULD place
verbatim speech inside `<speech_transcript>` and non-speech acoustic evidence
inside `<audio_observation>`. Acoustic evidence includes environmental events,
ambience, music, speaker activity, temporal changes, and uncertainty without
duplicating transcript text. Visual evidence belongs in `<visual_observation>`,
with no conversational reply outside those tags. A client may attribute only
the extracted transcript—not either observation—to the user's chat role.

The runtime MUST withhold conversational message text from the comprehension
graph. That text is an instruction to the language model, while the media graph
receives only media plus a modality-specific extraction instruction. This
separation prevents prompts such as “reply naturally” from producing a second,
misrouted assistant answer during perception.

### Adapter fields

| Field | Required | Meaning |
|---|---:|---|
| `omni.schema` | Recommended | Must be `robit.ollama.omni-adapter.v1` when supplied |
| `omni.task` | No | `chat`, `transcribe`, `describe`, or `synthesize`; default `chat` |
| `omni.include_audio_from_video` | No | Whether video decoding exposes its audio track; default `true` |
| `omni.require_speech` | No | When `true`, a comprehension result without a tagged speech transcript finalizes before language/TTS; default `false` |
| `omni.live_engaged` | No | Compatibility hint describing whether a `require_speech` client is in an active exchange; ambiguity alone never discards intelligible speech; default `true` |
| `omni.tool_routing` | No | `client` preserves the supplied tool array; `relevant` retains configured gateways plus a bounded request-relevant subset after current-media comprehension; default `client` |
| `response_modalities` | No | Non-empty subset of `text`, `audio`; default `text` |
| `speech_mode` | No | `auto`, `always`, or `never`; default `auto` |
| `speech` | No | Backend-specific voice, base language, regional locale, cloning, sampling, and style hints |

`speech_mode` has the following precedence:

1. `always` requests TTS.
2. `never` disables TTS.
3. `auto` requests TTS only when `response_modalities` contains `audio`.

`omni.task=synthesize` always executes TTS. `transcribe` and `describe` bypass
the language stage; they return comprehension text directly and may synthesize
that text when `speech_mode` requests audio.
`require_speech` is intended for VAD-driven live-call turns: environmental
evidence is still returned in the observation/final adapter metadata, but a
sound-only capture cannot consume the language or TTS stages. Leave it false
for explicit environmental-audio questions.

A live turn may intentionally return empty assistant content when the observed
speech was not addressed to the client. A bounded closed-set semantic preflight
resolves direct-address evidence before answer generation or tools and records
`adapter.speech_addressee=self|other|ambiguous`. A different human addressee
reports `adapter.tts_skipped_reason=speech_addressed_elsewhere`. Ambiguous speech
always reaches the language stage instead of being discarded by the preflight.
The model may answer it or return `<observe_only/>`; that control is removed and
reported as `empty_assistant_response`. In either case clients should retain the
user transcript for continuity without rendering an assistant bubble or audio
failure. This differs from sound-only `require_speech`, which stops before the
language stage and reports `required_speech_not_found`.

The streaming `observation` event and authoritative final adapter metadata MAY
expose `input_transcript`, `audio_observation`, and `visual_observation` as
separate fields. `visual_observation` MUST be populated only when the current
request actually contains image/video input; clients may retain a bounded,
sanitized copy as explicitly historical evidence for later follow-ups, but
MUST NOT represent it as a current view or replay the raw media automatically.

`tool_routing=relevant` is a reversible context optimization for clients that
supply a larger allowlisted catalog with media input. The adapter waits for the
current transcript/observation, ranks only the client-supplied schemas, keeps
configured discovery/execution gateways, and sends the bounded result to the
language model. When that result contains a concrete schema, the adapter adds a
compact required-action instruction naming the retained candidates; the model
still selects the tool and constructs its arguments. The adapter never adds a
client-owned capability, selects arguments, or authorizes execution. Tool follow-up rounds return to `client` routing because
the model's concrete call and tool result are then more authoritative than the
original media transport sentence.

The reference Qwen3-TTS Base worker supports `language`, BCP 47 `locale`,
trusted server-local `speaker_file`, request-local base64 WAV `speaker_audio`, `temperature`,
`top_k`, `top_p`, `seed`, and `max_frames`. `speaker_audio` is bounded to 10
MiB and 0.5–30 seconds and is removed after its generation. Other fields are
backend capabilities, not portable guarantees. In particular, its current
llama.cpp API has no separate natural-language style-instruction channel.
The adapter preserves the full locale for response wording and regional
conventions, then maps its primary subtag to one of Qwen3-TTS's ten ISO 639-1
language tokens. Regional acoustic accent still comes from the speaker
reference; the locale does not claim a separate accent checkpoint.
The reference adapter treats `max_frames` as a per-block ceiling: long replies
are split at natural text boundaries, PCM sequence numbering remains continuous,
and the complete final WAV concatenates every block. The adapter trace reports
the number as `tts_blocks`.

## Message media fields

Media is attached to a message alongside Ollama's string `content`:

```json
{
  "role": "user",
  "content": "What changed after the speaker entered the room?",
  "audios": [],
  "images": [],
  "videos": []
}
```

The adapter accepts media on earlier messages so history can be reconstructed,
but `transcribe`, `describe`, and `synthesize` validate their required input on
the last user message.

### Audio envelope

Audio input MUST be a complete uncompressed RIFF/WAVE containing 16 kHz, mono,
signed 16-bit PCM samples:

```json
{
  "mime_type": "audio/wav",
  "encoding": "base64",
  "data": "UklGR..."
}
```

The validator checks the RIFF/WAVE header, PCM compression mode, sample rate,
channel count, sample width, decoded size, and base64 syntax. Clients SHOULD
resample and downmix before encoding:

```bash
ffmpeg -i input.m4a -ac 1 -ar 16000 -c:a pcm_s16le speech-16khz-mono.wav
```

### Image envelope

Native Ollama bare-base64 image strings remain valid. A structured envelope is
preferred when building generic clients:

```json
{
  "mime_type": "image/png",
  "encoding": "base64",
  "data": "iVBOR..."
}
```

Supported MIME types are `image/jpeg`, `image/png`, and `image/webp`. The
declared type MUST match the decoded container signature.

### Video envelope

```json
{
  "mime_type": "video/mp4",
  "encoding": "base64",
  "data": "AAAAIGZ0eXA...",
  "sampling": {
    "fps": 1.0,
    "max_frames": 96,
    "include_audio": true
  }
}
```

Supported inputs are MP4, WebM, and animated GIF. The reference runtime
normalizes GIF to a bounded, silent MP4 before comprehension. MP4/WebM clips
with no audio stream are valid visual-only inputs; optional audio demux MUST
return no audio part rather than failing the video turn. Container support does
not imply every codec is decodable; the runtime SHOULD use FFmpeg or another
sandboxed decoder and return a media error for unsupported streams.

Sampling fields:

- `fps`: requested sample rate greater than 0 and at most 30;
- `max_frames`: hard cap from 1 through 1024;
- `include_audio`: whether to demux the audio track for temporal audio-visual
  comprehension.

The runtime MUST preserve frame order and SHOULD record source timestamps. If
it clips or subsamples due to context limits, the response adapter trace SHOULD
report the actual frame count and covered time range.

Documents are intentionally not part of adapter v1. A portal or application
MAY extract and retrieve document text before forwarding a normal text message,
but it MUST remove raw document envelopes, mark retrieved excerpts as untrusted
data, bound the inserted context, and isolate any retrieval index per user
session. This keeps the portable adapter concerned with model-native media
while allowing PDF/DOCX/text workflows in the reference portal.

## Tasks and routes

### `chat`

`chat` is the combined-agent route:

```text
audio/image/video → comprehension → semantic observation
                                            ↓
normal messages/tools/think → language model → content/tool_calls/thinking
                                            ↓
                                   optional text-conditioned TTS
```

Text-only chat skips comprehension. Speech is not generated while the response
contains unresolved `tool_calls`; the client executes the tools, appends tool
results, and submits the next turn before TTS.

The authenticated reference portal can perform that client role when the user
enables its wrench toggle (`portal_auto_tools=true`). It pins its own tool
schema array, executes uncapped rounds/calls until final/timeout/disconnect, emits running/completed
tool events on the portal NDJSON extension, and returns bounded arguments and
result evidence under `response.portal`. Exact duplicate execution remains
allowed: matching arguments are executed again and produce a fresh tool result.
The toggle defaults off. This flag and trace are portal extensions, not fields in
portable adapter v1.

An authenticated portal caller may set `portal_tool_allowlist` to a JSON array
of unique, exact names from the portal's pinned tool registry, for example
`["native_ui", "memory_write", "memory_search"]`. The portal always retains
the `tool_search` gateway; an empty array permits discovery without executable
leaf tools. Invalid or unknown names return HTTP 400. The caller's immutable
scope limits initial schemas, family discovery, and every automatic tool-loop
dispatch, including forged model calls to tools absent from the current schema.
Existing execution-profile denials still apply. The field is consumed before
adapter transport; model arguments and results cannot enlarge it. Omitting it
preserves the portal's existing behavior. It does not change client-owned tools
on the portable adapter or authorize direct calls made in a separate request.
When supplied, the final response records the applied scope, including the
discovery gateway, in `portal.tool_allowlist` on both portal transports.

Within this automatic loop, the model-facing `native_ui` and `computer_use`
schemas expose bare `observe`, bounded semantic `workflow`, independent
`verify`, and `cancel`. Per-action enumeration/selection/execution belongs to
the existing UI controller. Model-authored `execute` and step-bearing `observe`
are rejected before dispatch with `ui_planner_phase_required`; the raw proposal
remains in the tool trace. This restriction does not change the pinned canonical
tool API for direct tool callers or client-owned adapter loops. A workflow tool
result includes `phase_handoff` (`robit.omni.ui-phase-handoff.v1`) describing
position-bound verified and remaining steps, newly applied effects, terminal
verification and whether to plan another phase, replan or stop. A verified phase
does not establish whole-task completion. These are tool-result details within
the existing portal extension, not new portable adapter request fields.

An explicit foreground text task can set
`portal_execution_profile: "native_task"` together with
`portal_auto_tools:true` and permission for `native_ui`. This portal-owned field
is consumed before adapter transport. The server runs a mandatory phase compiler
instead of the general discovery loop: fresh native state plus mapped inactive
window identities → one Omni JSON-schema response → whole-phase validation →
the existing rapid controller → observed result/replanning. Compiler requests
contain no tools or tool choice; they use `response_format` on the deployed
OpenAI-compatible language backend, preserving the caller's native `think` and
sampling options. The identical phase schema is also present in model-visible
context; a decoder grammar alone does not teach the planner its contract.
Only authorized `memory_write`/`memory_search` phases may use
session memory; a planner JSON value cannot enlarge the request scope.

The profile rejects media, detached turns and background-worker requests. It
holds the normal inference-queue ticket, has a 40-phase ceiling and a wall bound
of the smaller of the configured portal timeout and 1,800 seconds, and stops
after three consecutive invalid, unverified or repeated zero-effect phases.
For this explicit `native_task` compiler, a first verified no-op remains valid.
A repeated UI phase with the same host-observed state and effective action
targets, arguments and effects, but no new applied effect, reports
`ui_phase_repeated_without_new_effect` and returns to planning without resetting
that failure count. Planner step IDs and remaining-criteria claims cannot make
the phase new. Original input and verification receipts remain unchanged;
actual mutations reset this repetition history. Reader and memory phases are
separate knowledge operations and are unaffected. This check does not change
ordinary chat or voice tool-loop routing. Client disconnect and
deadline cancellation stop the owning native controller. Raw executed receipts
remain in standard tool events and `portal.safe_tools_executed`; the next
compiler sees bounded recent handoffs/readbacks, not repeated UI maps. A final
`portal.native_task.status:"planner_claim"` has `task_verified:false` and must
not be interpreted as independently verified task completion. Ordinary requests
and portable adapter/client-owned tool contracts are unchanged.

An authenticated `native_task` caller may additionally supply
`portal_native_task_scope` to restrict native authority to exact host-registered
windows:

```json
{
  "schema": "robit.omni.native-task-scope.v1",
  "windows": [
    {"window_id": "81788956", "process_id": 3036686, "process_start": "4365561"}
  ]
}
```

Both objects are closed. `windows` contains 1–16 unique XIDs. `window_id` is a
canonical positive decimal string in the unsigned 32-bit range; `process_id`
is an integer from 1 through 2³¹−1, excluding booleans; `process_start` is a
positive decimal string of at most 20 digits. Repeated PIDs must have the same
start identity. Canonicalization sorts windows by numeric XID; its SHA256 binds
ASCII, sorted-key, compact JSON. Unauthenticated requests return HTTP 401.
Invalid scopes or use outside a native-enabled foreground `native_task` request
return HTTP 400. Omission keeps
the ordinary native authority of existing callers.

The portal consumes this field before portable parsing, adapter transport,
continuation storage and model context. The model receives a digest/count,
not the authorization tuples. A direct portable adapter request containing the
reserved field is rejected rather than silently accepting an unenforced scope.
The scope is immutable within the native controller epoch, survives renewal,
and binds native revisions. Planning and finite candidates include only
positively mapped authorized windows. Input and visual dispatch repeat fresh
authorization checks. A different top-level window sharing a PID is excluded;
a same-PID/start direct `WM_TRANSIENT_FOR` dialog may be authorized only while
its registered base remains live. New top-level windows are not auto-enrolled.

Task finalization closes only its captured native epoch and scope. Late cleanup
cannot cancel a replacement task; browser state and session memory remain
available. This is a native-controller restriction, not an OS sandbox or a
promise that an application cannot itself launch another window. Cancellation,
identity, revision, hit-target and effect-verification guards still apply.

The portal's native-task phase schema also accepts `kind:"visual_inspection"`,
which invokes the existing `native_ui` tool with `operation:"inspect_visual"`.
This closed read-only operation accepts no query, image, coordinates or model
choice. A pinned local reader receives one guarded desktop content snapshot;
fresh native/document bindings and exact transmitted pixels are rechecked after
inference. Changed, cancelled, protected or unavailable reads provide no accepted
text. A successful result carries bounded `reading` plus an untrusted
`visual_inspection_receipt`, with no input or mutation/verification credit.

These are canonical tool arguments/results and portal compiler details within
existing tool events. They add no portable adapter media fields or event types,
and never set `current_visual_input`, images or user transcript evidence.
Client-owned adapter tools remain client-owned and are not auto-executed.

If an adapter language stream ends with a classified network/timeout failure
before audio starts, the portal retries that exact round once. It emits
`{"type":"reset","reason":"upstream_network_retry","attempt":1}` before
replacement deltas so clients discard only the partial content and reasoning
from the interrupted round. Tool receipts from already completed prior rounds
remain authoritative. A second failure is terminal and is emitted as an error;
the diagnostic journal records the recovered or terminal upstream failure.
The browser applies the same one-retry/reset rule when Chromium itself reports
a transient fetch/network change. It reconnects only while replay is safe:
audio must not have started, and every completed tool must be on the explicit
read-only retry allowlist. Stateful browser, GUI, shell, memory-write, and task
actions are never replayed automatically.

For automatic tool execution, a malformed structured call batch may receive one
model-generated format correction per request. The entire rejected batch remains
undispatched; prior completed tool receipts are retained. The correction asks for
one smaller valid action without repairing arguments or replaying prior effects.
It keeps the same admission, live tool budget, upstream timeout policy and client
cancellation semantics. Accepted speech is promoted to text and audio is removed;
missing speech transcripts, images/video, or any prior output audio forbid this
recovery. Textual-call, policy, execution and network failures do not trigger it.
Streaming emits `{"type":"reset","reason":"malformed_structured_tool_call","attempt":1}`
before correction deltas to clear speculative text and reasoning while retaining
earlier tool receipts. A second malformed structured batch is terminal.

### `transcribe`

Requires audio on the last user message. It executes only the comprehension
graph with a transcription instruction and returns its text as
`message.content`. This route is the ASR example and avoids a second language
model paraphrasing the transcript.

### `describe`

Requires at least one audio, image, or video item on the last user message. It
executes only comprehension and returns a temporal/semantic description.

### `synthesize`

Requires text and no input media on the last user message. It bypasses language
generation and sends that text directly to the TTS graph. The same text is
returned in `message.content` and as the optional `transcript` associated with
`message.audio`.

## Response

The adapter preserves the normal Ollama response object. It adds an `audio`
field to the assistant message when TTS runs:

```json
{
  "type": "audio",
  "mime_type": "audio/wav",
  "encoding": "base64",
  "container": "wav",
  "codec": "pcm_s16le",
  "sample_rate_hz": 24000,
  "channels": 1,
  "sample_width_bits": 16,
  "frames": 24000,
  "duration_ms": 1000,
  "decoded_bytes": 48044,
  "data": "UklGR...",
  "transcript": "Text synthesized into this waveform."
}
```

The adapter also adds a compact trace:

```json
{
  "adapter": {
    "schema": "robit.ollama.omni-adapter.v1",
    "task": "chat",
    "route": ["comprehension", "language", "tts"],
    "input_modalities": ["text", "audio"],
    "evidence_provenance": {
      "current_media_modalities": ["audio"],
      "current_visual_input": false,
      "tool_data_is_visual_input": false,
      "prior_dialogue_is_current_observation": false
    },
    "speech_synthesized": true
  }
}
```

The provenance booleans are authoritative origin metadata. In particular,
audio-only requests remain `current_visual_input=false` even if a perception
backend emits visual-looking prose. Tool results and prior dialogue can inform
an answer but cannot be represented as something currently seen.

Implementations SHOULD omit internal chain-of-thought from logs and traces.
Normal Ollama `message.thinking` behavior is controlled by the request and the
runtime; the adapter does not merge it into spoken content.

## Errors

Recommended status mapping:

| Status | Condition |
|---:|---|
| 400 | Invalid schema, malformed base64, wrong media format, unsupported task, or `stream:true` |
| 413 | Deployment-specific request or decoded-media limit exceeded |
| 422 | Container is valid but its codec or stream layout cannot be normalized |
| 500 | Adapter invariant or component loader failure |
| 502 | Component execution returned an invalid response |
| 503 | Required component is unavailable or not loaded |
| 507 | Insufficient CPU/GPU memory to load the requested graph |

Errors are JSON objects containing at least `error` and `schema`. They MUST NOT
echo base64 payloads, decoded media, model prompts, or secrets.

## Security requirements

- Validate decoded size before inference and enforce aggregate request limits.
- Treat media metadata, OCR, transcripts, and captions as untrusted input.
- Isolate media decoding; disable network resolution unless explicitly needed.
- Do not accept arbitrary local file paths from remote callers.
- Bound frame count, duration, dimensions, channels, and decompression ratios.
- Redact input media and generated speech from default logs.
- Authenticate and rate-limit the adapter when it is exposed beyond loopback.
- Preserve tool schemas but do not let semantic observations create or mutate
  tool definitions.

## Versioning

The schema identifier is the compatibility boundary. Additive response fields
may be introduced within v1. Any change to media encoding, task semantics,
streaming framing, required fields, or tensor-view selection requires a new
schema identifier. Runtimes MUST reject unknown explicit schema values.
