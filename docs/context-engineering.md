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
| Structured checkpoint contracts | `control_tools` | background `task_checkpoint` |
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

Tool discovery is also context-bounded. Once `tool_search` selects concrete
capabilities, its follow-up exposes those contracts instead of retaining the
unrelated initial bridge schemas. The adapter estimates the fully rendered
message-and-schema prompt before submission. If the language backend still
reports an exact context overflow, the adapter sheds one generic stale-context
layer and retries; it preserves the current user turn, system policy, current
tool chain, and the concrete capability selected by discovery.

## Task phases and evidence

The durable store distinguishes human-visible scheduling phase from model
evidence. Memory pressure can leave a task pending, but is never injected into
the model transcript and can never justify a blocked checkpoint. Admission is
checked before a task is claimed, preventing repeated running/pending churn in
the indicator. Administrative `background_task` list, status, update, cancel,
and start operations remain available at the memory floor; inference, shell,
browser work, and every substantive tool remain governed.

Every concrete background call is retained as a bounded audit record containing
call ID, exact tool name, bounded/redacted arguments, outcome, and success
state. The local voice foreground and its background worker use one stable,
opaque portal-session scope derived from the daemon capability, so a handed-off
task can continue in the same visible browser without crossing into another
user session. The top-bar task submenu shows those calls directly. A live task has a
**Cancel task** action, and every task has **Clear task record**; the global
**Clear finished tasks** action archives terminal records before removing them.

## Conversation tracing

Normal diagnostics remain content-redacted. Setting `OMNI_CALL_LOG_CONTENT=1`
on the trusted local harness adds single-line `conversation_trace` JSON records
for `heard`, `audio_observation`, `generated`, `tts_input`, and `playback`.
Tests inject a trace callback instead of writing private content, which makes
the ASR-to-generation-to-TTS flow observable without globally weakening the
privacy default.
