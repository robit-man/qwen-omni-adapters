# Portal tools and tool chaining

The reference portal includes a deliberately small tool harness for validating
the logical Omni tag's standard Ollama-compatible tool calling. The language
stage still produces ordinary `message.tool_calls`; the portal executes only
its server-owned allowlist, appends normal `role: "tool"` results, and asks the
same model for the final answer. Audio, image, and video comprehension therefore
can lead into the same tool loop without changing the adapter ABI.

This is runtime plumbing, not a claim that web access or memory is embedded in
the GGUF weights. Direct adapter users may continue to pass and execute their
own Ollama tool schemas. Portal execution is off by default. A user opts in for
the current browser session with the small wrench button beside the brain
button, which makes the client send `portal_auto_tools: true`. The server still
owns the schemas and implementations.

## Built-in tools

| Tool | Purpose | State or network scope |
|---|---|---|
| `get_current_time` | Current portal-host date, time, timezone, and UTC offset | Read-only host metadata |
| `get_system_snapshot` | Fresh bounded platform, CPU/load, RAM, NVIDIA GPU, network-counter, date, and time snapshot | Read-only portal-host metadata; no hostnames, addresses, processes, credentials, or session content |
| `get_user_location` | Return coarse IP-derived city/region/country, rounded coordinates, and timezone for location-dependent requests | Browser performs the HTTPS lookup; sanitized result is isolated to the current session; raw IP is never sent to or retained by the portal |
| `get_portal_capabilities` | Report model, media, document, and safe-tool capabilities | Read-only runtime metadata |
| `web_search` | Discover public result links and snippets through the no-key DuckDuckGo HTML page, or search this session's local page index | Public search page for `discover`; no network for `session` |
| `web_fetch` | Fetch one source URL with a retrieval receipt and bounded text or raw HTML | Public HTTP(S) only |
| `document_search` | Search already attached PDF, DOCX, text, or code chunks | Current browser session only |
| `memory_write` | Store a compact temporary fact or research note | Current browser session only |
| `memory_read` | Read an exact temporary topic/key | Current browser session only |
| `memory_search` | Lexically retrieve temporary memories by relevance | Current browser session only |
| `tool_search` | Search the allowlisted catalog when capability mapping is unclear; discovery never completes an action request | Read-only runtime metadata |
| `background_task` | Hand an executable outcome to the persistent worker, which discovers and invokes the required allowlisted tools | Current voice session and durable local task store |
| `safe_math_eval` | Evaluate bounded arithmetic and common math functions with an AST interpreter | Pure computation; no code execution |
| `structured_read` | Read/query attached JSON, JSONL, CSV, TSV, or YAML | Current browser-session attachments only |
| `web_crawl` | Fetch a bounded same-origin page graph | Public HTTP(S), 8 pages and depth 2 maximum |
| `ocr_pdf` | OCR an attached scanned PDF and index the recognized text | Current browser-session attachments only |
| `session_search` | Federated recall over dialogue, memory, notes, tasks, documents, and fetched pages | Current browser session only |
| `audio_analyze` | Inspect observed audio streams, duration, format, and volume | Current browser-session media only |
| `video_scan` | Inspect observed video/audio streams and timeline metadata | Current browser-session media only |
| `working_notes` | Add, list, search, or remove bounded research notes | Current browser session only |
| `task_list` | Maintain bounded pending/in-progress/completed/blocked tasks | Current browser session only |
| `shell` | Run a raw `bash -lc` command and return stdout, stderr, exit status, cwd, and timeout state; optional bounded stdin supports safe generated-file writes | Unrestricted portal-host shell; 900-second runtime, 64-KiB stdin, and 64-KiB-per-stream capture bounds |
| `subagent_delegate` | Run one fresh helper completion for isolated analysis, planning, synthesis, or critique | Synchronous text-only model call; no tools, media, host access, or parent history; result stored in the current browser session |
| `subagent_list` | List completed helper delegations | Current browser session only |
| `subagent_result` | Retrieve one completed helper result by task ID | Current browser session only |
| `subagent_forget` | Delete one stored helper result | Current browser session only |

The portal publishes the exact JSON schemas as `safe_tools` from `/api/status`.
It also advertises `tool_execution.client_opt_in=true` and
`default_enabled=false`. The phone UI sends the schemas only while the wrench
is enabled, while the server replaces them with its authoritative copy whenever
automatic execution is requested. A client cannot redefine a safe tool's
implementation by changing its schema. The trusted tool-use contract is also
injected only for opted-in turns; tools-off turns receive neither that contract
nor schemas. The model-facing first pass contains only the compact `tool_search`
contract (about 126 pessimistically estimated tokens, instead of about 3,900
for the full catalog). Its result makes at most three matching concrete schemas
visible for exactly the next inference; after a concrete call, the contract
collapses to discovery again.

Host awareness is deliberately tool-only. Ordinary turns receive a short,
stable behavioral system policy and no hardware/utilization blob. When a user
asks about the portal host, or a task materially depends on current resources,
the model can call `get_system_snapshot`; each call samples fresh bounded data.
The result describes the server running the portal, never the user's phone.

User location is also explicit and tool-only. When tools are enabled, the
phone client calls `https://ipwho.is/` directly in the browser; the local voice
client performs the same lookup in an isolated headless Chromium process while
the model stack starts. Both clients allowlist only coarse geographic fields,
round coordinates to two decimals, and send only that sanitized object with
the model request. The portal never receives the lookup's raw `ip`, ISP,
connection, security, or currency fields. `get_user_location` returns only the
current opaque client session's sanitized value; Trash clears the phone value
and the normal five-minute portal TTL expires either session value. A failed
phone lookup is retried after a short cooldown instead of being cached as
unavailable for the page's lifetime. IP location is approximate and may
identify a carrier gateway or VPN rather than the physical device. A typical
dependent chain is `get_user_location -> web_search -> web_fetch`, while an
unavailable location requires an explicit city from the user.

Every location, search, fetch, and crawl result includes a machine-readable
`provenance` object in the same JSON shown by the portal's **Tools** disclosure.
It identifies the producing tool, evidence type, authority, source URL when
applicable, and whether the material is ready to cite. Location additionally
includes binding `claim_limits`: it may seed an approximate-area search but
cannot establish device GPS, a street/address, visible surroundings, or a
camera observation. Search metadata is discovery-only; factual web claims must
come from a fetched `source_url` and be attributed to it.

## Tool round lifecycle

```text
user/media turn
  -> optional Qwen3-Omni comprehension
  -> deterministic routing exposes a small matching schema set when confident
  -> an actionable match requires one structured call (not a capability disclaimer)
  -> otherwise Qwen3.8 calls tool_search when a capability is needed
  -> portal exposes only matching concrete schema(s) for the next round
  -> Qwen3.8 emits the concrete structured tool_call
  -> portal validates and executes allowlisted calls
  -> portal appends assistant tool_calls + role=tool results
  -> Qwen3.8 may call another tool or produce the final answer
  -> optional Qwen3-TTS after no unresolved calls remain
```

There is no numeric call, round, or per-turn ceiling in either the synchronous
or streaming portal loop. The model may emit one call at a time or batch any
number of independent calls in a round, and chaining continues until it emits a
final answer. Exact call fingerprints are de-duplicated; if an entire round
contains only calls already executed during the turn, the portal stops with an
explicit no-progress error instead of spinning forever. A request timeout,
client disconnect, or upstream failure also terminates execution. These are
progress and transport boundaries, not hidden call quotas. Media bytes and raw document
envelopes are removed from older follow-up rounds; a browser/desktop tool's newest
fresh screenshot is passed through the adapter's native image field for exactly one
multimodal follow-up. Its `role=tool` JSON retains bounded frame metadata and the
visual digest, never the base64 bytes as language text. Tagged observations,
retrieved text, and prior dialogue remain as bounded text context. TTS is deferred
while a tool call is unresolved and runs only for the final answer.

A successful `tool_search` resolves an action address but is not evidence that
the requested action occurred. When discovery exposes one or more bounded leaf
schemas and no concrete tool ran in that round, the following model-selected
leaf call is required. Normal final-answer behavior resumes immediately after
that concrete result; this prevents stochastic prose completion between
discovery and execution without hard-coding a task-specific tool choice.

The portal's NDJSON stream adds `type: "tool"` start/completion events between
normal adapter events. Start events identify running calls; completion events
carry their success state and a bounded result preview. The authoritative final
response repeats this evidence in `portal.safe_tools_executed`. The UI merges
the two phases by call ID and exposes arguments/results in a collapsible
**Tools** row, parallel to the reasoning disclosure. The browser retains every
call receipt for the turn; JSON arguments and results render as nested key/value
rows, while each plain-text result preview is capped at 12,000 characters to
protect the DOM. Traces are retained only in that browser session's existing
five-minute cache.

## Sub-agent delegation

`subagent_delegate` is a deliberately narrow orchestration primitive. It sends
one independently answerable objective, an optional specialization
(`general`, `researcher`, `planner`, or `critic`), and explicitly selected
evidence to a fresh helper context. The model-facing schema is reference-only:
`context_source` must select `none`, `current_user_message`, or
`latest_non_discovery_tool_result`, so a large source is copied server-side
rather than regenerated inside tool-call JSON. The helper request always has a system message first,
`think=false`, text-only output, no speech, no media, no portal schemas, and no
tool access. It therefore cannot recursively delegate or perform external
actions. The parent remains responsible for web/document/tool work and for
verifying source-dependent claims.

The call is synchronous: its completed result is immediately returned as a
normal `role="tool"` observation and stored under an opaque task ID. This avoids
orphan background workers and polling loops. `subagent_list`, `subagent_result`,
and `subagent_forget` manage those completed records. Storage is keyed by the
hashed Secure browser-session cookie, expires under the same idle TTL as other
portal state, and is deleted by Trash. A task ID from another browser session
cannot retrieve a result.

The same 24,000-character delegation bound applies. Current-user handoff copies
no earlier conversation, tool result, media, or hidden state. Tool-result handoff
skips `tool_search` metadata and copies only the latest concrete result. The
direct execution endpoint still accepts the legacy bounded `context` argument
for compatibility, but it is not exposed to model inference.

Native `message.tool_calls` remain authoritative. For compatible renderers that
emit Omnius-style `<tool_call>{...}</tool_call>` text, the portal parses the
bounded JSON into the same structure and removes the control block from visible
answer text. Mixed or malformed textual calls fail closed; they are never shown
or executed as plausible answer text.

## Request example

The server injects its schemas, so callers need only opt into execution:

```json
{
  "model": "robit/qwen3.8-27b-e03-obliterated-omni:q4km",
  "messages": [{
    "role": "user",
    "content": "Find the current project release, read the primary source, remember the version, and summarize it."
  }],
  "omni": {
    "schema": "robit.ollama.omni-adapter.v1",
    "task": "chat"
  },
  "response_modalities": ["text"],
  "speech_mode": "never",
  "think": false,
  "stream": false,
  "portal_auto_tools": true
}
```

A typical dependent chain is `tool_search(web discovery) -> web_search ->
tool_search(page retrieval) -> web_fetch -> tool_search(memory) -> memory_write
-> final answer`. Discovery exposes a bounded contract rather than dumping the
full catalog into every context. After a concrete call, only that active schema
stays beside discovery, so iterative shell work can correct or verify a command
without rediscovering the same capability.
A later turn in the same browser session can use `memory_read` or
`memory_search`; `web_search(mode=session)` searches the already indexed result
and fetched-page text without another discovery request. Independent read-only
calls may share a tool round, but dependent calls wait for `role="tool"`
evidence. The runtime contract directs the model to select the narrowest tool,
avoid duplicate calls, fetch primary sources before citing them, and write
memory only when the user requests it or a compact fact is needed later in the
same session.

## Omnius-derived design selection

The implementation was selected after reviewing Omnius's search, fetch, crawl,
browser-action, network-egress, tool-executor, exposure-policy, batching,
textual-call parser, trace-collapse, and memory search/read/write paths. This
portal retains the pieces appropriate to a small public demonstration:

- trusted server-owned schemas and explicit client opt-in;
- native structured calls with a strict textual compatibility parser;
- uncapped progress-checked multi-round execution and `role="tool"` observations;
- dependent chaining, duplicate suppression, and read-only batching guidance;
- no-key DuckDuckGo HTML discovery separated from verified page retrieval and bounded crawl;
- per-session fetched-page indexing and lexical term/bigram recall;
- attachment-scoped structured reads and OCR, pure AST math, session working
  state, and technical media probes;
- compact, collapsible running/completed tool receipts; and
- URL, DNS, redirect, media-type, size, session, and TTL boundaries.

The portal does not copy Omnius's full Playwright validation surface. It does
expose a smaller persistent Chromium interaction tool with fresh screenshots
and bounded visible elements. Its crawl remains read-only, same-origin, and
bounded to eight pages at depth two. This deployment also exposes the
separately discoverable unrestricted `shell` tool at the operator's request;
it is raw host authority, not a sandbox, and retrieved text remains untrusted
data rather than command authority.

## Web safety and limits

There is no search API, API SDK, credential, provider fan-out, or browser scrape
in ordinary discovery. `web_search(mode=discover)` ports Omnius's production
`WebSearchTool`: it sends one bounded GET to DuckDuckGo's public no-key HTML
results page and extracts only result links, titles, and snippets. Search
metadata remains unverified until `web_fetch` retrieves the exact selected URL.
Interactive and JavaScript-heavy work is a separate `browser_interact` path
that always opens on the attached desktop. If the service cannot join that
desktop it returns an explicit capability handoff instead of silently launching
headless Chromium. Browser and desktop interaction support drag gestures, and
rendered `<summary>` controls are clickable for reasoning/tool inspection.

Browser actions use a hybrid grounding order. A returned DOM `element_id` is the
preferred authority: immediately before acting, the executor re-resolves that node,
refreshes its current viewport box, verifies visibility and enabled state, and
confirms that the center hit-test is not occluded. Canvas, challenge, image-map, and
other non-DOM targets stay in the exact CDP viewport screenshot instead of switching
to a whole-window or desktop frame. Their `visual_click` points use Qwen's normalized
0–1000 coordinate convention. When perception emits exactly one strict current-frame
`target`, `point`, and `bbox`, the executor admits that point directly and records both
the language proposal and executed coordinates; ambiguous or multi-target frames remain
with language reasoning. The first point on a full viewport is treated as a
region proposal rather than a click: the executor returns a bounded 400×300 target crop,
passes the target identity—but no parent-frame coordinates—into the crop perception
pass, and the second point is
deterministically mapped through the crop into current viewport CSS coordinates. The
executor rejects a crop whose pixels changed while the model was deciding. Every
browser action returns a new screenshot and visual change receipt; a changed frame is
causal evidence, not proof that the intended state was reached.

Native form controls remain element-grounded as well. `type` enters ordinary text,
`set_value` sets date/time/month/week/number/range/color controls through their native
value contract, and `select` chooses one or more exact option values. `upload` uses
Chromium's file-input protocol only for regular files of at most 16 MiB beneath
`OMNI_BROWSER_UPLOAD_ROOTS` (default `runtime-data/browser-uploads`); arbitrary host
paths are rejected before Chromium sees them.

The visible browser is globally single-instance for this runtime. A cancelled or timed-out
GUI fixture closes its stable voice-agent browser session in `finally`; before another
launch, the store reaps only orphan processes carrying an exact runtime-owned
`omni-visible-chromium-*` temporary profile. This prevents failed gates and portal
restarts from accumulating Chrome windows without using broad executable-name kills.

`gui_interact` is reserved for controls outside the browser viewport. It returns an
active-window crop by default and interprets its coordinates relative to that returned
image. The crop comes from the root-window capture using the same X11 bounds used to
translate input, including window-manager decorations; its declared width and height
therefore match its pixels exactly. It accepts deterministic pixel points and
normalized 0–1000 visual-grounding points. The model selects the full-screen coordinate
space only when operating a panel, workspace, or another window. If a later action
omits the space, the runtime reuses the newest returned frame rather than silently
changing its meaning. An active-window action is rejected when focus changed after its
observation.

The focused deterministic gate is `tests/test_gui_action_loop.py`. It renders
synthetic desktops with offset and resized browser windows, asserts that captured
target pixels and translated clicks share one coordinate frame, exercises full-screen
handoff, and distinguishes a hit from an unchanged miss. On a running desktop
deployment, `python runtime/verify_gui_action_loop.py` adds an end-to-end multimodal
gate: the background worker must navigate visible Chromium and solve three canvas-only
image targets, including a modal and a shifted lower strip. The fixture keeps target
coordinates out of the DOM, verifies normalized viewport clicks independently, rejects
escape to desktop-wide GUI control and nonvisual bypass tools, and requires fresh visual
evidence before completion.

Discovery indexes at most 48 result/fetched pages and 128,000 characters for
the opaque browser session. `web_search(mode=session)` ranks that local index
with a deterministic lexical term/bigram scorer and performs no network call.
`web_fetch` separately retrieves one chosen page, issues a content hash and
source receipt, caches the full bounded response for 60 seconds, and can return
plain text or bounded raw HTML. `web_crawl` applies the same checks and receipts
to at most eight same-origin pages, depth two, and 20,000 returned characters.
This split applies the portal's
public-tunnel constraints:

- only absolute HTTP(S) URLs are accepted;
- URL credentials, localhost, `.local`, metadata endpoints, and every
  non-global resolved IPv4/IPv6 address are blocked;
- every redirect is revalidated, with at most four redirects;
- response bodies are capped at 5 MiB and fetched output at 12,000
  characters;
- only textual MIME types are accepted, and known binary signatures are
  rejected even when a server labels them as text;
- scripts and styles are stripped from normal fetched evidence; `raw_html`
  exposes bounded untrusted source only when explicitly requested;
- page fetching does not support authentication, cookies, forms, downloads, or
  arbitrary browser automation;
- fetched pages and search snippets are labelled untrusted data and cannot
  change system or tool policy.

Production deployments that require a stronger network boundary should place
the portal behind an egress proxy or firewall allowlist. DNS validation in an
application process is defense in depth, not a substitute for network policy.

## Memory and document boundaries

Temporary memory is keyed by a SHA-256 hash of the opaque Secure browser
session cookie. It allows 64 entries, 4,096 characters per entry, and 32,768
characters per session. Exact reads suggest related keys on a miss; searches
use the same deterministic lexical term/bigram relevance scorer as the local
web index. Memory is in process only, expires after five idle minutes, and is
deleted immediately with the Trash control. It is never shared between users
and is not persistent knowledge-base storage.

`document_search` queries the same session-isolated index used by automatic
PDF/DOCX/text retrieval. Raw attachment bytes are retained in process for five
idle minutes, bounded to 48 MiB per session, solely so `structured_read` and
`ocr_pdf` can operate without host paths. `ocr_pdf` uses `pdftoppm` and
`tesseract`, processes at most 50 pages, then adds recognized text to the same
session index. `structured_read` uses safe JSON/YAML parsers and bounded CSV/TSV
rows; it never evaluates document content.

`working_notes`, `task_list`, observed-media metadata, and conversation recall
share the same opaque session boundary and Trash/TTL cleanup. `audio_analyze`
and `video_scan` use `ffprobe` (and `ffmpeg` volume detection for audio) during
ingestion, retain only bounded technical results, and do not retain a second
copy of media bytes. `safe_math_eval` walks a limited arithmetic AST; imports,
attributes, variables, comprehensions, and arbitrary Python are impossible.
The `shell` tool deliberately permits arbitrary Bash, host-filesystem, and
process control. Only transport safety is imposed: a 900-second maximum and
64 KiB captured from each output stream; timeout kills the command's process
group. Credentialed browser automation remains excluded.

## Verification

Unit gates cover:

- structured multi-round `memory_write -> memory_search -> final` streaming;
- DuckDuckGo HTML result/snippet parsing and redirect decoding followed by page fetch;
- network-free session-index recall and absence of browser/provider fallback;
- fetch receipts, raw-HTML/text cache reuse, binary refusal, script/style
  removal, response bounding, and private-address rejection;
- browser-session memory and document isolation;
- allowlisted tool discovery and forbidden math-expression rejection;
- raw shell stdout/stderr/exit-context return and bounded capture;
- structured JSON/YAML paths and attachment-scoped OCR indexing;
- bounded same-origin crawling and federated session recall;
- audio/video observation isolation, working notes, and task state;
- Trash-triggered memory, web-cache, document-index, and diagnostic cleanup;
- preservation of the original media-removal and no-context-bleed invariants.

Run the focused suite with:

```bash
PYTHONPATH=src:. .venv/bin/pytest -q tests/test_omni_portal.py
```
