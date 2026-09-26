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
| Public tool contracts and typed family membership | `tools[].schema`, top-level `tool_families` | portal discovery, foreground chaining, and background execution |
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

For a live-call audio request, the portal does not run query-specific virtual
memory retrieval before transcription. At that point its only textual query is
a transport sentence describing an attachment; retrieving against it can page
an unrelated old topic beside ambiguous speech. The bounded role-preserving
dialogue and audio still reach comprehension unchanged. Text and durable-task
queries continue to use the full provenance-preserving virtual hierarchy.

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

The durable worker applies the same boundary independently of the model. It
must select one closed capability family or explicitly select `uncertain`; the
runtime never interprets natural-language wording to accept, reject, or rewrite
that selection. A physical-camera result is filtered from background discovery,
and an attempted call is rejected before capture, unless the immutable task
scope or a later human direction explicitly depends on a physical scene.
Browser and desktop visual inspection authorize only their respective scoped
tools.

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
window on every turn. Compaction starts from measured working-set pressure at
72 percent of the current resident token tier, not from a small count of chat
messages; 128 messages remain only as a defensive ceiling. The compacted chain
retains at least the two newest complete action/observation cycles plus the
current typed frontier. At the 4K floor, that frontier receives roughly 1.2 KiB
of text; larger KV tiers expand it up to 8 KiB. The constrained focus contract
has the same authority and page-in rules in less prose, leaving room for the
freshest typed receipt. This is working-set eviction only. Exact bounded tool
receipts are archived append-only before their turns leave L0 and remain
recoverable with `task_expand`.

The renewable model transcript is not the task database. Every durable task has
an external `robit.omni.background-task-state.v1` record with three separately
versioned ledgers: knowledge for newly closed evidence slots, environment for
executor-observed mutations and their changed paths, and controller state for
the current requirement, next transition, and stagnation fingerprint. Tool
success and task progress are deliberately different: a successful inspection
can close one knowledge slot without changing the environment, while repeating
that same slot against the same environment version does not advance either
ledger. Model prose cannot increment these versions.

The state loop is `PRETHINK -> RETRIEVE or ACT -> AUDIT -> PRETHINK`. A
deterministic auditor classifies every external receipt as discovery,
inspection, mutation, verification, concrete evidence, or failure. After a
mutation, a terminal checkpoint requires a verification receipt from the
current environment version. A verification is itself new milestone evidence
only when it checks an environment version newer than the last accepted phase
frontier; the initial verify-only phase uses a `-1` frontier so pre-existing
state can still be verified. Relabeling an unchanged inspection as verification
therefore cannot reopen a retired transition or advance another checkpoint.
Accepted phase checkpoints create a fresh
executor working set from the external state rather than carrying forward the
previous executor's private reasoning. That fresh working set is persisted
without inventing an extra task round, so a process restart cannot revive the
discarded chain.

Fresh does not mean evidence-free. At every accepted phase checkpoint and
audited stagnation reset, the controller selects a bounded, class-diverse set
of typed milestone IDs and pages their exact stored tool-result text beside the
next executor request. Source acquisitions, mutations, verifications, and
external interactions have independent retention quotas, so a later directory
creation cannot displace the research source needed to write its contents.
Checkpoint evidence IDs accumulate as a bounded page table instead of being
silently overwritten. The replay budget is derived from the live resident KV
tier (about 4K characters at the 4K floor and 16K at the normal 16K tier), and
working-set truncation is explicit; the append-only receipt and digest remain
available through `task_expand`. Replayed web/document payloads remain
untrusted evidence, never instructions. This implements evidence replay rather
than summary-of-summary continuity.

Milestone progress is narrower than tool success, new knowledge, or environment
change. A checkpoint milestone must cite a typed source acquisition, verified
mutation, external interaction, or verification receipt. A clock lookup,
system snapshot, repeated inspection, or other successful information probe
may be useful evidence but cannot advance or complete an execution phase by
itself. This separation prevents the model from converting an easy ancillary
tool success into apparent application progress.
When a typed milestone comes from a non-sticky capability such as source
fetching, the next action contract contains only the checkpoint transition;
broad discovery reopens after the controller has written that evidence into
durable state. A completed discovery transition likewise consumes the single
post-inspection replan pass, so the selected leaf tools replace the router
instead of competing with it on subsequent rounds.

The background action surface is tier-aware too. It exposes one already-selected
concrete family plus an eligible checkpoint, removes only
documentation annotations from their schemas, and preserves names, types,
enums, required fields, bounds, and object closure. Discovery is suppressed on
every round while a concrete family remains active; it returns only after an
accepted phase checkpoint, a typed capability failure, or loss of the active
leaf. This makes the scoped action space stable across a multi-action phase
instead of repeatedly reclassifying the same task. `task_expand` remains a
bounded secondary control. Tests account for the complete
system/query/tool/output envelope at 4K; an active background overflow is a
visible retryable failure, never an unreported native-context fallback.
The generation budget is action-aware as well. Routing and motor actions retain
the small default ceiling, while a selected `workspace_file` or `shell`
capability may use up to 3,072 output tokens at the 16K tier (and
proportionally less when KV is downshifted) so a research document, heredoc, or
code-file JSON object can close cleanly.
An explicit operator/test ceiling is never overridden. This changes output
headroom, not the physical context length or evidence-retention budget.
`tool_search.family` is a closed capability-family enum selected by the model.
The runtime resolves that exact family and never interprets request words with
prefixes, suffixes, stemming, prompt-specific vocabulary, or lexical scoring.
When no single family is justified, `uncertain` returns the typed family
descriptions without exposing a guessed leaf. The task subject remains pinned
separately and is never transformed into routing keywords.

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
Shell calls carry an explicit inspect, filesystem-mutation, runtime-mutation,
or verification intent. Filesystem mutations declare bounded effect paths; the
executor fingerprints those paths before and after execution, and only an
observed delta becomes mutation evidence. An exit code or command vocabulary is
never interpreted as progress. Runtime mutations stay unverified until a
separate verification succeeds. Read-only shell and workspace probes remain
inspections, with their bounded observation pinned directly rather than hidden
behind command text or promoted to a completed success. An inspection does not
make `task_expand` compete with the next action; paging returns after new
durable evidence or at a phase boundary.
The round immediately following typed non-progress evidence receives a bounded
private planning pass and can either continue with the current concrete family
or invoke the small typed-family selector to recover through a different
capability. Concrete successful action chains continue without discovery and
with thinking disabled for latency. This gives the transformer room to change
strategy after an observation without reopening the full tool catalog or
exposing private reasoning. If that one deliberate pass emits prose instead of
a structured action, the adapter's required-tool retry retains the same
evidence and closed tool contract but switches the backend's native reasoning
control off, turning the retry into an execution pass instead of repeating the
same thought cycle. The runtime never fabricates a tool call.

Every concrete background call is retained as a bounded audit record containing
call ID, exact tool name, bounded/redacted arguments, outcome, success state,
evidence slot, environment version, and deterministic progress classification.
Repeated no-progress actions with the same current subtask, environment
version, unresolved evidence, and action family increment a stagnation counter.
The second repeat forces replanning and retires that exact typed action family
until either the knowledge or environment ledger advances. The next executable
tool grammar removes only the retired enum value—for example, a stagnant
`shell:inspect` leaves `shell:mutate_filesystem`, `shell:mutate_runtime`, and
`shell:verify` available. A call that disregards the narrowed grammar is
rejected before external execution. This policy reads audit fields, not command
text or task-specific keywords. The third repeat also discards the renewable
executor context and restarts from the audited state, while preserving a typed
alternative capability already selected by the retry controller. Retirement
is stored per action family and is not cleared by arbitrary new knowledge such
as a clock or system lookup; only a typed milestone or verified environment
change reopens it.

The local voice foreground and its background worker use one stable, opaque
portal-session scope derived from the daemon capability, so a handed-off task
can continue in the same visible browser without crossing into another user
session. The top-bar task submenu shows those calls directly. A live task has a
**Cancel task** action, and every task has **Clear task record**; the global
**Clear finished tasks** action archives terminal records before removing them.

After each concrete result, the worker injects a bounded `<task_self_check>`
that requires the next reasoning pass to compare that result with the durable
objective, completion criteria, latest spoken guidance, and runtime-owned audit
state. The model must identify remaining work before choosing another
structured action, but it does not grade or version its own result. Checkpoints
carry a separate criteria assessment and must cite the freshest concrete result;
an older successful call cannot hide a newer failed verification. Exact user
text, rather than a model-authored paraphrase, becomes durable guidance when a
foreground turn updates a task. These checks remain private task-control
context and are never synthesized as reasoning.

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
