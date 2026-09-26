# Context engineering

All static model-facing instructions and tool descriptors have one source of
truth: `src/qwen_omni_adapters/context.json`. The file is packaged with the
runtime and validated as `robit.omni.context.v1` by
`qwen_omni_adapters.context`. Set `OMNI_CONTEXT_FILE` to an alternate complete
catalog before process startup when testing a revision; partial overlays are
not accepted.

## Context surfaces

| Surface | Catalog section | Runtime consumers |
|---|---|---|
| Default text behavior and media extraction | `prompts.default_language_system`, `prompts.media_encoder_system`, transcribe/describe entries | `runtime/adapter_server.py` |
| Spoken-turn behavior | `prompts.live_call_system` | local `harness/call.py` and browser context embedded by `portal/app.py` |
| Browser attachment behavior | `prompts.media_conversation_system` | `portal/static/portal.js` through the rendered `omni-context` JSON block |
| Portal provenance and tool policy | `prompts.portal_behavior_system`, `directives.tool_result_policy`, `directives.tool_use` | `portal/environment.py`, `portal/app.py`, `portal/tools.py` |
| Persistent execution policy | `prompts.background_agent_system`, background directives | `harness/background_agent.py` |
| Isolated helper policy | `prompts.subagent_system` | portal sub-agent runner |
| Public tool contracts and discovery vocabulary | `tools[].schema`, `tools[].discovery_hints` | portal discovery, foreground chaining, and background execution |
| Structured checkpoint and compaction contracts | `control_tools` | background `task_checkpoint`, `task_compact` |
| Human-visible durable-task phases | `task_stages` | task store, worker, and top-bar indicator |

Dynamic evidence stays in code because it is request state rather than policy:
the current objective and later spoken guidance, live time/battery/network
facts, current media provenance, recalled memories, and concrete tool results.
Those values are wrapped by catalog policy but are never written back into the
catalog.

The browser does not carry its own prompt copy. The server serializes the two
browser prompt values from the catalog into the page, and JavaScript fails
closed if either is absent. This keeps browser calls and the local microphone
harness aligned.

Live speech is also protected at the completed-answer boundary. The catalog
instructs the language trunk to speak as a participant rather than announce an
assistant role, availability, readiness, or a generic offer. Before a live
`require_speech` answer is exposed or synthesized, the adapter removes complete
sentences that consist of known support-agent closings or unsolicited
microphone/message-delivery commentary. Explicit questions about those phrases
or the audio path remain answerable. A reply containing only disallowed filler
fails closed rather than sending the filler to Qwen3-TTS. Text-only API calls
are not rewritten by this live-only guard.

The same live boundary applies an ordinary two-sentence limit, a generation
ceiling that still leaves room for tool-call arguments, plus maximum TTS block
and decoded-audio-duration limits. Explicit detail, list, comparison, drafting,
and multi-example requests retain their requested expansion. These are
runaway-response circuit breakers; they do not change the trained audio
encoder, codec graph, shipped voice reference, persistent worker protocol, or
PCM decode window.

Language-template behavior is model-profile-specific. Standard Ornith 1.5 uses
its native explicit no-thinking template when the request has `think:false`;
otherwise it can consume the entire live output budget in hidden reasoning and
return no speakable answer. Qwen3.8 retains the omission path required by its
template. A real `think:true` request overrides either profile and keeps the
reasoning channel separate.

Language prompts receive a bounded `<self_state>` generated at runtime. An
explicit `OMNI_AGENT_NAME` overrides the default; otherwise the name comes from
the actual OS service account, never a hard-coded device name. Live speech
history remains in native user/assistant roles without adding model-copyable
“Earlier in this conversation” prose. Live and background language turns also
set llama.cpp `cache_prompt:false`: the bounded messages in the request are the
authoritative state, so a prior generated reply cannot survive as stale slot KV
and discarded context can be reclaimed on unified-memory devices.

Tool discovery is also context-bounded. Once `tool_search` selects concrete
capabilities, its follow-up exposes those contracts instead of retaining the
unrelated initial bridge schemas. The adapter estimates the fully rendered
message-and-schema prompt before submission. If the language backend still
reports an exact context overflow, the adapter sheds one generic stale-context
layer and retries; it preserves the current user turn, system policy, current
tool chain, and the concrete capability selected by discovery.

The spoken foreground also receives `background_task` as an executable gateway
to the worker's full discoverable allowlist. A leaf schema omitted from the
bounded foreground prompt is therefore not presented as a missing capability:
the foreground hands off the requested outcome, and the worker discovers and
invokes only the concrete schemas needed to complete it.

Camera availability is likewise a gateway, not eager evidence. The local
harness sends the spoken turn without a room image and exposes only
`request_camera_view`. The model may request a still or bounded clip only when
the current intent depends on physical-scene facts. A successful request starts
a new multimodal pass; unrelated turns never receive, describe, or carry an
ambient frame merely because cameras are enabled.

The durable worker applies the same boundary independently of the model. Its
tool discovery query must identify a concrete interaction mechanism; generic
catalog fishing such as asking for whatever tools are available is rejected and
replanned. A physical-camera result is filtered from background discovery, and
an attempted call is rejected before capture, unless the immutable task scope or
a later human direction explicitly depends on a physical scene. Browser and
desktop visual inspection authorize only their respective scoped tools.

Trained audio bridges have an additional reproducibility boundary. Audio-only
chat and direct ASR use the same tagged system/directive pair as projector
training and release evaluation, with native `enable_thinking=false` prefill.
The pair's release digest is
`9f73862652e0226ec3f9690f0a783d1c21dc1113285b4dc18d0edd51f2766758`.
Caller text is withheld from this perception pass, and the result must contain
separate `speech_transcript` and `audio_observation` elements. This prevents a
plain-text transcription prompt from defeating the attributed-speech parser or
turning acoustic evidence into visual evidence.

## Task phases and evidence

The durable store distinguishes human-visible scheduling phase from model
evidence. Memory pressure can leave a task pending, but is never injected into
the model transcript and can never justify a blocked checkpoint. Claiming and
deterministic transcript compaction are bounded control work admitted in the
soft-to-hard safety band; new residency still waits for the soft floor and the
emergency watcher cancels continuing work at the hard floor. Administrative
`background_task` list, status, update, cancel, and start operations remain
available at the memory floor. Shell declares a bounded peak reserve and owns a
runtime pressure watcher, while browser work and other substantive tools retain
their declared admission policies.

Background transcript limits are derived from the live resident comprehension
window on every turn. At the 4K floor the raw recurrent chain is bounded to a
32 KiB high-water mark, keeps four newest native messages after compaction, and
allocates roughly 1.2 KiB of text to the pinned typed frontier; the limits expand
at larger KV tiers up to 96 KiB, twelve messages, and an 8 KiB frontier. The
constrained focus contract has the same authority and page-in rules in less
prose, leaving room for the freshest typed receipt. This is working-set eviction
only. Exact bounded tool receipts are archived append-only before their turns
leave L0 and remain recoverable with `task_expand`.

The background action surface is tier-aware too. At 4K/8K it exposes one
already-selected concrete tool plus an eligible checkpoint, removes only
documentation annotations from their schemas, and preserves names, types,
enums, required fields, bounds, and object closure. Discovery is suppressed on
the immediate post-routing round, then returns after one concrete leaf attempt
so a later phase can choose another capability. At constrained tiers it and
`task_expand` alternate as secondary controls. Tests account for the complete
system/query/tool/output envelope at 4K; an active background overflow is a
visible retryable failure, never an unreported native-context fallback.
`tool_search.query` is capability-only: it describes the missing mechanism,
while the subject remains in the pinned task. Catalog hints explicitly route
current product/SaaS, comparison, interface, and design research to public-web
discovery so topical words cannot win an unrelated shell/service match.

The deterministic compacted chain contains one current system contract, one
small page marker, and the newest native tool cycles. It never duplicates the
focus ledger in both the system and checkpoint messages. The ledger coalesces
repeated inspections/failures, keeps the newest resident version of a changed
artifact, exposes counts of nonresident records, and retains exact evidence IDs
for page-in. Omitted receipts can be located without a pre-known ID through the
exact/lexical `task_expand` query path, then replayed verbatim. `task_compact`
receipts report the live resident token tier, working-set limits, before/after
message and byte counts, and retained evidence IDs so budget movement is
observable rather than inferred.
One page-in is offered at a time. After `task_expand` succeeds or returns
`evidence_not_found`, it is removed until a new concrete external result makes
another page-in relevant. It cannot retrieve capabilities or schemas;
`tool_search` is the only route for those. This keeps evidence paging from
becoming a self-reinforcing action loop after compaction.

Focus categories preserve evidence authority. `web_search` discovery is a
route to source-bearing fetch/browser evidence, not a completed research
source. A rendered HTTP page can become an acquired source; `about:blank` is an
inspection with `task_progress=false`. The browser then exposes only its
`navigate` contract for the next action, preventing a blank viewport from being
clicked or from opening an unrelated discovery branch.

Every concrete background call is retained as a bounded audit record containing
call ID, exact tool name, bounded/redacted arguments, outcome, and success
state. The local voice foreground and its background worker use one stable,
opaque portal-session scope derived from the daemon capability, so a handed-off
task can continue in the same visible browser without crossing into another
user session. The top-bar task submenu shows those calls directly. A live task has a
**Cancel task** action, and every task has **Clear task record**; the global
**Clear finished tasks** action archives terminal records before removing them.

After each concrete result, the worker injects a bounded `<task_self_check>`
that requires the next reasoning pass to compare that result with the durable
objective, completion criteria, and latest spoken guidance. The model must
identify remaining work before choosing another structured action. Checkpoints
carry a separate criteria assessment and must cite the freshest concrete result;
an older successful call cannot hide a newer failed verification. These checks
remain private task-control context and are never synthesized as reasoning.

Eight concrete actions without an accepted progress checkpoint form a mandatory
phase boundary. The next inference receives only the checkpoint control and must
name the earliest unmet requirement before external work can continue. An
accepted progress boundary clears the prior active-tool scope so the next phase
is planned from the durable milestone instead of inheriting a research, shell,
or browser loop. For progress only, the runtime may replace malformed or invented
provenance with the actual freshest successful tool-call ID; it records that
normalization explicitly. Completion and blocked checkpoints never receive this
repair and still require exact, valid evidence.

## Conversation tracing

Normal diagnostics remain content-redacted. Setting `OMNI_CALL_LOG_CONTENT=1`
on the trusted local harness adds single-line `conversation_trace` JSON records
for `heard`, `audio_observation`, `generated`, `tts_input`, and `playback`.
Tests inject a trace callback instead of writing private content, which makes
the ASR-to-generation-to-TTS flow observable without globally weakening the
privacy default.
