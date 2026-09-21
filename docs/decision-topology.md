# Decision topology and Laya migration inventory

This document is the phase-1/phase-2 inventory for the System-1 decision-plane
refactor. It records the decisions the runtime actually makes before any fast
path is enabled. The classifications are deliberately conservative:

- **A — deterministic:** exact parsing, validation, policy, bounds, or state
  transition. A model must not be inserted.
- **B — Laya-fast:** a bounded judgment may take a direct, reversible fast path
  after workload-specific calibration.
- **C — Laya-gated:** Laya may recommend or precompute; low confidence,
  disagreement, or consequential scope escalates to the deliberative model.
- **D — LLM-required:** open-ended understanding, planning, synthesis, or novel
  tool argument construction remains generative.
- **E — human/authority-gated:** deterministic policy or explicit authority is
  required regardless of a prediction.

Laya participates in prediction only. Policy authorizes, and typed executors
perform actions.

## Existing execution topology

```text
microphone/browser request
  -> VAD + bounded utterance queue
  -> adapter request validation
  -> media comprehension (when media exists)
  -> deliberative language model
  -> portal tool-call parser
  -> deterministic authorization/validation
  -> tool executor
  -> deliberative language model again
  -> repeat until prose or durable checkpoint
  -> TTS (spoken mode)

background_task
  -> durable claim/lease
  -> deliberative model chooses/discovers a tool
  -> tool executor
  -> deliberative model assesses result and chooses retry/continue/complete
  -> repeat, with deterministic slice and memory-pressure scheduling
```

The serial generative edges after discovery and after every observation are the
primary optimization target. The initial model call remains necessary for
substantive answers and open-ended task decomposition.

## Lifecycle map

| Lifecycle | Current implementation | Decision-bearing boundaries |
|---|---|---|
| Spoken input | `harness/vad.py`, `harness/call_queue.py`, `harness/call.py` | signal admission, echo rejection, turn consolidation, camera/tool bridge exposure |
| Request ABI | `contract.py`, `runtime/adapter_server.py` | schema, media route, comprehension admission, language-context shedding, TTS admission |
| Foreground agent | `portal/app.py` | input policy, initial schemas, iterative tool calls, progress/stall, completion |
| Durable agent | `background_agent.py`, `background_tasks.py` | claim, plan, tool choice, call admission, result assessment, retry, checkpoint, completion |
| Tool plane | `portal/tools.py`, `browser.py`, `gui.py` | discovery/ranking, input validation, authority, web safety, execution |
| Retrieval | `tools.py`, `documents.py`, `harness/memory.py` | whether to retrieve, ranking/relevance, bounded context inclusion |
| Runtime | `daemon.py`, `memory.py`, `comprehension_launcher.py`, `tts_server.py` | capacity admission, residency, health, retry, readiness |
| UI/state | `portal.js`, `indicator.py`, `session_cache.js`, `background_tasks.py` | local opt-in authority, cancellation, expiry, display-only restoration |

## Decision inventory

Latency values describe the current cost class. Deterministic values are local
CPU time. `1 LLM round` is workload- and residency-dependent; recent live traces
show a simple spoken comprehension/language request at about 6.0 s, while tool
rounds have ranged from roughly 3 s to more than 20 s. Frequencies are per turn,
tool step, or service event as noted.

| Decision | Current implementation | Current latency / frequency | Category | Laya candidate | Consequence | Fallback and expected benefit |
|---|---|---|---|---|---|---|
| Accept audio frame as speech | energy/native-VAD thresholds and hysteresis | <1 ms/frame, high | A | No | medium | Keep exact; no model failure surface |
| Merge speech segments / cap newest audio | elapsed-time/sample bounds | <1 ms/segment | A | No | medium | Keep exact and OOM-bounded |
| Reject recent playback echo | normalized token-sequence comparison | <1 ms/turn | A | No | low | Keep exact; conservative false negatives are safe |
| Adapter schema/media validity | typed validation, signatures, byte/count limits | <5 ms/request | A | No | high | Reject invalid ABI deterministically |
| Select task route (`comprehension/language/tts`) | task, modality, and speech-mode table | <1 ms/request | A | No | high | Keep exact route contract |
| Stop sound-only live call before language/TTS | `require_speech` plus empty transcript | one comprehension pass | A | No | high | Preserve source/provenance invariant |
| Retry video with fewer frames after explicit overflow | response code plus bounded frame reduction | rare, no extra judge | A | No | medium | Keep exact |
| Shed context after explicit overflow | ordered deterministic removal | rare, per retry | A | No | high | Keep exact and generic across tasks |
| Decide whether a new request is actionable/social/question/task | currently implicit in the language model | 1 LLM round/turn | C | Yes, wave 0 | low | Shadow, then use only to select route hints; LLM on uncertainty |
| Decide whether tools/retrieval/planning are likely needed | currently implicit in first language pass | 1 LLM round/turn | C | Yes, wave 0/1 | medium | Pre-expose small schemas when calibrated; otherwise discovery/LLM |
| Produce an ordinary substantive answer | language generation | 1 LLM round/turn | D | No | low-medium | Deliberative model generates content |
| Interpret ambiguous spoken instructions | language generation | 1 LLM round/turn | D | No | medium | Deliberative model or user clarification |
| Decompose a long-horizon task | background language generation | 1+ LLM rounds/task | D | No | medium-high | Deliberative planner; Laya only gates admission |
| Select a broad tool family | lexical discovery only after model calls `tool_search` | extra LLM discovery round | C | Yes, wave 1 | low | Hierarchical Laya + lexical agreement can pre-expose schemas |
| Rank a small concrete tool subset | lexical hint score and top-3 cutoff | <1 ms/discovery | C | Yes | low | Shadow against lexical baseline; never flatten large tool sets |
| Construct novel tool arguments | language model structured tool call | 1 LLM round/step | D | No | medium-high | Keep LLM unless values are closed candidates or copied verbatim |
| Validate tool name and JSON arguments | allowlist/schema/typed bounds | <1 ms/call | A | No | high | Deterministic rejection |
| Authorize tools for a browser session | explicit local opt-in and server allowlist | <1 ms/call | E | No | high | Authority always overrides predictions |
| Authorize shell/background side effects | execution profile plus server policy | <1 ms/call | E | No | high | Never infer authorization from confidence |
| Detect exact duplicate calls/results | stable hash/digest sets | <1 ms/call | A | No | medium | Keep exact |
| Admit a proposed action as goal-consistent/non-duplicate | currently only prompt plus exact duplicate block | part of LLM round | C | Yes, wave 2 | medium | Shadow; high confidence can admit only policy-permitted typed actions |
| Validate public URL/redirect/DNS target | URL parser, resolver, public-IP checks | network-dependent | A/E | No | high | Fail closed deterministically |
| Detect web challenge / unsupported interactive flow | markers and rendered state | <1 ms after fetch | A for markers, C for semantics | Shadow semantics only | medium | Deterministic marker wins; ambiguous state goes to LLM |
| Choose browser operation from current candidates | language model sees bounded DOM/screenshots | 1 LLM round/action | C/D | Yes only for closed operations/targets | medium | Jev-style bounded candidate wave; novel URLs/text remain LLM |
| Generate arbitrary GUI coordinates | visual language model | 1 LLM round/action | D | No | high | Laya has no visual grounding input |
| Determine whether tool result made progress | coarse deterministic heuristic foreground; LLM background | <1 ms foreground, 1 LLM round/background | A for exact failures; C otherwise | Yes, wave 3 | medium | Deterministic failure markers first; calibrated Laya; LLM on ambiguity |
| Select retry disposition | background prompt implicitly chooses next call | 1 LLM round/failure | C | Yes, wave 3 | medium | Choice: same/different tool/replan/ask/terminate; modifications may need LLM |
| Detect task completion | evidence-ID validation plus LLM checkpoint assertion | 1 LLM round/step | C | Yes, wave 4 | medium-high | Laya recommends only; exact criteria/policy validate; ambiguous uses LLM |
| Mark task blocked | failed evidence ID required | <1 ms after model assertion | E/A | No unilateral Laya | high | Concrete failed evidence and policy required |
| Final task report wording | language model checkpoint report | included in LLM round | D | No | low | Deliberative synthesis, bounded length |
| Stop runaway foreground tool loop | repeated nonproductive-round bound | <1 ms/round | A | No | high | Keep deterministic circuit breaker |
| Yield a background work slice | round/call/stall counters | <1 ms/round | A | No | medium | Keep deterministic fairness bound |
| Retry backend/resource failures | typed error plus exponential schedule | <1 ms/failure | A | No | medium | Keep scheduler state out of task semantics |
| Preempt background work for live speech | event flag and stream cancellation | <50 ms polling | A | No | high | Keep exact single-flight behavior |
| Claim/lease/cancel/archive durable tasks | locked state transitions | local I/O | A/E | No | high | User cancellation remains authoritative |
| Choose whether retrieval is needed | implicit in language tool selection | 1 LLM round/turn | C | Yes, wave 1 | low-medium | Precompute gate; uncertain requests retain retrieval tool |
| Rank session memory entries | token overlap or embedding cosine plus age/strength | local/embed call | A for numeric rank, C for semantic relevance | Shadow relevance | medium | Never delete from probabilistic output |
| Admit passive memory write | duplicate/length/strength checks and async embedding | local/embed call | A | No | medium | Preserve reversible, generic memory policy |
| Select document chunks | hashed sparse cosine and hard caps | local | A for top-k, C for final relevance | Shadow relevance | medium | Laya may filter reversibly; deterministic fallback includes original set |
| Select web/session search results | search engine plus lexical session index | network/local | C | Yes, relevance shadow | low | Use source authority and deterministic URL safety independently |
| Include prior dialogue in context | recency/age/count bounds | <1 ms/turn | A, later C shadow | Limited | medium | Never discard durably; Laya can shadow reversible inclusion |
| Current visual/audio provenance | explicit attachment origin and newest-media rule | <1 ms/request | A | No | high | Preserve contract exactly |
| Subagent admission and role | main model calls a bounded helper tool | 1 LLM round | C for admission, D for delegated work | Yes | medium | Laya may recommend stable role class; LLM creates objective/context |
| Model capability routing | fixed current model/route | <1 ms/request | C where multiple classes exist | Yes | medium | Route to stable capability class; unavailable plane keeps current model |
| Human confirmation / sensitive action | explicit policy and UI authority | varies | E | Risk signal only | high | Laya cannot grant permission |
| Runtime memory admission | `MemoryGovernor` against `MemAvailable` and reserve | <1 ms/operation | A | No | critical | Generic host-wide policy; cancel at hard floor |
| Select comprehension context size | measured residency/KV calibration and caps | startup seconds | A | No | critical | Keep resource math deterministic |
| Start/stop model services | supervisor health/residency state | seconds/start | A/E | No | critical | Explicit config and proven device residency |
| Choose Laya checkpoint by script/language | Laya Router deterministic script/language analysis | <1 ms/wave | A | No recursive model | medium | Explicit override > task > language route |
| Decide whether Laya itself may load | config + memory admission + platform device facts | startup | A/E | No | critical | Degrade to existing behavior when unavailable |
| TTS profile reuse/reset | exact profile key and worker protocol | <1 ms/request plus synthesis | A | No | high | Preserve fresh decoded state and resident weights where possible |
| Split TTS synthesis blocks | punctuation/contract bounds | <1 ms/reply | A | No | medium | It is transport framing, not response-length policy |
| Browser/UI session expiry and Trash | cookie-scoped TTL and explicit clear | timer/local I/O | A/E | No | high | Never probabilistically delete |

## Proposed decision waves

```text
WAVE 0 — admission (one batch after transcript/current request exists)
  response disposition, actionable class, ambiguity, risk signal

WAVE 1 — routing (same batch when state is available)
  tool need, retrieval need, planning need, tool family, response mode

WAVE 2 — pre-action (one batch for proposed typed calls)
  goal consistency, duplicate likelihood, scope surprise, result-dependent gate
  -> deterministic authorization -> executor

WAVE 3 — observation (one batch after new tool state)
  outcome, relevance, completeness, retry usefulness, replan need

WAVE 4 — completion (may share wave 3 when inputs are identical)
  goal satisfied, pending actions, missing information, safe-to-finalize signal
  -> deterministic evidence/policy validation -> finalize or deliberate
```

Questions in a wave share one compact canonical state and one Laya forward
pass. A later wave exists only because a tool observation changed state.

## Target architecture

```text
external input
  -> deterministic normalization / provenance / authority snapshot
  -> DecisionPlane.evaluate(wave, typed definitions)
       -> resident Laya backend (batched, timeout, cache, trace)
       -> confidence/calibration policy
       -> shadow comparison or reversible fast recommendation
       -> on failure: deterministic answer where available, else LLM
  -> deterministic authorization
  -> direct typed path OR deliberative LLM
  -> executor
  -> normalized observation
  -> next DecisionPlane wave
```

No application module imports `laya`. Application code consumes only typed
`DecisionResult` objects. The Laya SDK is isolated behind a loopback resident
worker so dependency failures, CUDA OOM, and model reloads cannot crash the
portal or voice harness.

## Initial activation policy

Existing production behavior remains authoritative in shadow mode. A decision
family can leave shadow only after replay data provides an error bound and a
per-family threshold. High-consequence action admission and completion remain
escalate-by-default until their false-fast-path rate is measured. The first
eligible optimization is reversible schema pre-exposure when hierarchical
Laya routing agrees with deterministic lexical discovery; this can remove the
otherwise mandatory `tool_search` generative round without executing an
action or bypassing policy.
