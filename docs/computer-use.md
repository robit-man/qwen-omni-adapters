# Semantic computer use

The `computer_use` portal tool ports the finite-action control layer demonstrated
in Omnius's `packages/execution/src/rapid-computer-use/`. The reference is
Omnius main `0ae974bd6c08ed9a129075fd184e63462f6b0021`, including the October
12–14 desktop verification and native-controller integration. This runtime
implements the contracts in Python around the signed-in Linux desktop and its
existing application/browser windows. It does not require an adjacent checkout
at runtime. An isolated browser context is used only when explicitly requested.
The [later Omnius audit](computer-use-omnius-layer-audit-2026-10-15.md) pins the
recent reference layer and distinguishes fixed-planner native demonstrations
from model-driven evidence.

The selected Qwen or Ornith Omni model produces a typed plan. The controller
resolves each step against current semantic controls, offers opaque finite
action IDs, and executes a unique match without another reasoner call. Every
mutation declares observable effects. Bounded polling verifies those effects,
and a separate fresh observation verifies the plan's terminal predicates.
Ambiguity, unavailable observation, stale state, cancellation, and missing
effects remain explicit outcomes.

Workflow feedback retains a bounded selection receipt for each actual selector
call, including the step, semantic revision, current scope, typed selection
status, executable candidate count and selected kind. If no executable action
is offered, bounded exact target matches report their observed capabilities and
whether their identity is the current scope. Widget focus is not evidence that
a window is inactive. Truncation and omitted matches remain explicit; an empty
list cannot prove absence. Protected names are redacted, and neither these
receipts nor policy confidence authorize input or count as mutation evidence.
The next Omni phase receives this feedback through the compact result, while
the controller retains its original candidates and exact execution checks.

Native capability failures retain a bounded diagnostic code in unresolved step
reasons and the native-task terminal message. Only ASCII identifier codes of at
most 128 characters are displayed; arbitrary provider text and tracebacks are
excluded at the task boundary. Cancellation and deadlines retain precedence.
This improves failure evidence without retrying input, accepting a failed
observation as state or changing executor guards.

## Host-owned execution

An action ID binds the session, scope, document/control generation, semantic
revision, operation, arguments, and expected effects. Execution accepts that ID
alone. Both the controller and native adapter revalidate before dispatch.
Dispatch uncertainty is retained before verification, preventing a repeated
request from applying the same input again. The host's existing authenticated
tool opt-in grants execution authority; model confidence cannot grant it.

A foreground native task can further restrict that authority with the
portal-only `portal_native_task_scope` described in the
[protocol](protocol.md). The caller registers exact XID/PID/process-start tuples;
the planner and finite-action generator see only positively mapped authorized
windows, and the executor freshly checks authority before input or visual
dispatch. A shared application PID does not authorize its other top-level
windows. The private immutable scope binds the controller epoch and revision;
only its digest and count enter model context. Task cleanup cannot remove a
replacement epoch's scope. Ordinary callers that omit this restriction retain
their existing native authority.

Chromium semantic observations skip screenshots and inference. The revision
hash remains stable when the observed state is unchanged. Navigation generates
a new document identity even if a new page reuses an element ID. Native DOM
input retains the existing visibility, enabled-state, and hit-target checks.
Password/one-time-code controls are excluded from finite candidates. Bounded
or truncated observations cannot prove that an unobserved control is absent.

Native identity history follows complete parent/child coverage. A budget cap or
inactive-window view retains unknown membership without renewing generations;
a complete child list can retire a genuinely removed subtree. Retained history
contains bounded identity metadata only. Input handles still come exclusively
from the current observation, and failed reads roll back the identity history.

For a prepared native `switch_window` step, the execution observation contains
the active window's controls and every observed inactive window matching the
typed target query. The revision binds that query and each matched window's
fresh XID, PID and process start. Unrelated inactive roots do not change that
revision, while changed active evidence, matching windows or ownership still
invalidate it. Both controller and bridge keep their exact revision checks.
Planning observations remain broad. Traversal limits remain conservative: a
changed bounded active view can still reject input, and truncation never proves
that no other window matches.

An activation whose exact target is already the active mapped root can return
`receipt.status="already_satisfied"` with `input_attempted:false`. This is
limited to `switch_window`, empty args and one exact `active_window` predicate.
The controller independently re-observes the stable binding and predicate
before advancing. The receipt is never dispatched input or mutation credit;
ordinary inactive-window activation and all other actions retain their guards.

When an `invoke` step has no native match, a named visual target can be grounded
by the configured local point worker. Its synthetic candidate is bound to the
exact source image and semantic revision. The executor checks both again and
maps browser image pixels into CDP CSS coordinates, or native image pixels into
the bound X11 screen rectangle, before input. It never executes
coordinates supplied by a decision model. This A100 deployment uses
**Moondream 3.1 9B-A2B**, pinned to
`1da27e6c2d2f9118de14560df456881dc62d91f3`, through the native Photon backend.
The worker verifies checkpoint, tokenizer, and runtime hashes before loading.
The existing Moondream2/Transformers backend remains available on Tegra.
Broker hosts prepare the exact scoped lease before point-worker CUDA growth,
prove residency, and shed weights after each request. Tegra retains its own
integrated-GPU residency proof.

Native vision uses `import -screen -window XID` to capture current root-screen
pixels cropped to the foreground window's visible rectangle, without resizing.
The frame binds the verified default-root ID and dimensions, full window surface,
clipped image rectangle, pixel offset, PID/process start and image hash. Exact
before/after checks reject changing bindings; they do not make capture atomic.
Point normalization uses the image rectangle while physical dispatch retains
the full surface geometry guard. This follows ImageMagick 6's
[root capture](https://raw.githubusercontent.com/ImageMagick/ImageMagick6/6.9.12-98/magick/xwindow.c)
and [crop implementation](https://raw.githubusercontent.com/ImageMagick/ImageMagick6/6.9.12-98/magick/transform.c).

When the native observer publishes a positive WINDOW-space coordinate binding,
the adapter validates its exact owner, full surface, declared CSD margins and
content inset before cropping the accepted root image to the visible content
intersection. This uses host-derived geometry without resizing or pixel masking.
The transmitted PNG is deterministic RGB, matching the point worker's decoded
color input; source palette and PNG metadata cannot change that content hash.
Its image origin and pixel offset describe the cropped region in root-screen
coordinates, while dispatch still guards the full window surface. Point receipts
retain the original `source_frame_sha256` and `source_image_geometry` as
provenance, alongside `content_region`. Outside-source pixels are excluded from
the content action key. Missing native mapping preserves full-window capture;
a malformed present mapping withholds capture rather than guessing an inset.

For a known native control, physical `click`, `double_click` and `right_click`
can delegate to the pinned point worker before candidate selection when the
native center hit is unavailable. This requires one exact semantic match in an
untruncated observation, current eligibility and bounds, and a complete private
read-only probe bound to the original ID/generation, revision, scope, owner,
process start and coordinate mapping. A native target hit keeps the ordinary
native path. A protected or different-control hit blocks the handoff; generic
panel-to-NULL evidence is provider-unknown, not proof of physical occlusion.
Unbound ancestry, foreign-looking roles without an ownership proof, cycles,
depth limits, stale evidence and read failures cannot authorize vision.

The point must fall inside the same semantic control and captured image, away
from protected controls. Its candidate keeps the semantic ID, generation,
values and other capabilities; only the requested pointer verb uses the exact
frame/probe-bound route. There is no duplicate synthetic control or private
step rewrite. Dispatch repeats the bindings without another model call and
never retries an uncertain input through another backend. Effects compare the
base semantic revisions, so frame/probe metadata alone cannot count as a
mutation. Transmitted-region hashes remain exact: changing clocks, animations or
other pixels inside that region can reject the action before input.

Linux native controls use a fixed AT-SPI bridge and native action/editable-text/
selection interfaces. Pointer and keyboard input use guarded native XTEST;
the executor verifies foreground PID/window identity, absolute client geometry,
native hit proof or the bound visual route, and actual pointer position. It never retries an uncertain
action using a different input backend. Missing accessibility or ambiguous foreground ownership
is a capability failure. Windows UIA and the architecture's full
alternative-model comparison matrix are outside this tested port.

A null accessibility child marks its parent's child list incomplete while
independently read siblings remain available in a truncated snapshot. Unknown
subtrees retain identity history without input handles; complete sibling lists
can still prove removals. Desktop/provider failures and the separate DocumentWeb
root-completeness checks retain their existing failure behavior.

For a foreground Flatpak window with an absent `_NET_WM_PID`, a read-only
XRes 1.2 query can identify the server's host owner. A distinct AT-SPI relay is
admitted only through one complete, fresh Flatpak instance proof: authenticated
D-Bus credentials, process starts/UIDs/namespaces, host-to-sandbox lineage,
registry and cgroup, installed proxy identity, matching namespace metadata, and
one active native frame with exact SCREEN/X11 geometry. The metadata trust
boundary follows [Flatpak 1.12.7's proxy copy](https://github.com/flatpak/flatpak/blob/1.12.7/common/flatpak-run.c#L1101)
and [xdg-desktop-portal 1.14.4's peer identification](https://github.com/flatpak/xdg-desktop-portal/blob/1.14.4/src/xdp-utils.c#L534).
The proxy copy may be writable; this proves neither proxy ancestry nor a socket
peer edge. The full tuple is refreshed before input and its private digest binds
the observation revision. Raw proof metadata never enters model-facing state.
Inactive namespace windows are mapped only through a complete managed-client
inventory, XRes host ownership, the full provider proof and unique exact native
SCREEN geometry. Identity ambiguity is resolved before applying any optional
session window scope; that scope narrows existing native authority. Missing proof
never falls back to a title or relay PID. Private bindings are refreshed before one
ordinary activation, then the active observer independently verifies the result.
This grants no owned Flatpak launch/cleanup authority.

Native browser URL predicates use `scope.url` from the active window's sole
visible top-level `DocumentWeb` and its fresh AT-SPI `DocURL` attribute.
`document_evidence` binds that URI to the document node, generation, window and
provider identity. Browser chrome, inactive documents, nested frames and typed
address-bar text cannot supply it. An incomplete structural scan, ambiguity,
unavailable capability or changing document withholds URL evidence; read-budget
failures remain fatal. This uses the ordinary browser's accessibility API and
does not introduce a separate browser session or input backend.

Native `key_chord` receipts identify the attempted keyboard route as
`backend: x11_guarded_xtest`. Nonzero exit, launch error or timeout remains an
uncertain dispatched attempt and is never retried automatically. A rejection
before input carries no backend claim. This metadata establishes the chosen
route; fresh application effects still determine whether the task succeeded.

Native text nodes may carry optional `text_observation` knowledge from the
provider reads already made: full `character_count`, `selection_count`, and
`captured_range: {start: 0, end: N}` with `truncated`. Counts are nonnegative
provider integers; omitted components are unknown, never zero. A positive
`selection_count` can coexist with an unavailable `text_selection` range.
Only an actually read zero count establishes zero reported selections.

Coverage describes the actual returned Unicode-character prefix in `value`,
including a short provider return or the existing text budget. Empty captured
text is `0..0`; `truncated` is false only when that prefix covers the valid
full count. Coerced text, malformed counts and unavailable reads supply no
coverage proof. Native offsets are not browser UTF-16 units, graphemes or key
counts. Protected text exposes none of this metadata. Bounded planner views
preserve it with the node; an omitted node remains unknown. Historical views
retain only permitted control evidence with their existing historical labels.

This object is observation knowledge, not a predicate or input capability.
It is excluded from the existing native revision and idempotency payload, so
clarification alone cannot create mutation credit, retry authority or failure
counter progress. Raw state, tool and planner-packet hashes still cover it.
Existing value and positive selection-range changes retain their normal
verification semantics. Knowledge alone grants no input or retry authority.

Native `select_text` may choose one guarded Ctrl+A input for an explicitly
requested complete, nonempty `0..N` range. The revision-bound value must have
exactly N Unicode characters, the captured prefix must cover the full provider
count, and the focused editable control must already advertise `select_text`,
`type_text` and `key_chord`. Immediately before input, the backend independently
rereads the complete value/count and rechecks provider, owner, scope, generation,
focus, editability and window geometry. A failed check rejects before input.
The original action, exact range, idempotency and selection predicate remain
unchanged; a fresh observed range still determines success. The physical route
reports `backend: x11_guarded_xtest`, including uncertain attempts. It is chosen
before any native setter, never after a failed setter, and cannot fall back to
or retry native input. Partial, incomplete or ineligible text retains the native
selection path and receives no inferred caret or Unicode offset mapping.

On GNOME, enable the signed-in session's accessibility before using native
browser controls: `gsettings set org.gnome.desktop.interface toolkit-accessibility true`.
Firefox must register its actual AT-SPI tree; a missing tree remains an explicit
capability failure. This does not create a separate browser profile. GTK4
window-relative coordinates are translated only when the foreground X11
geometry, declared frame margins, and accessibility dimensions agree exactly.
Clipped descendants retain semantic observations but receive no pointer bounds.

Native `type_text` takes `args: {"text": "..."}` and sends one bounded keyboard
input to an observed editable textbox or combobox. Text must be nonempty, at
most 2048 characters, and contain no NUL. The control must expose readable text,
remain visible, enabled and unprotected, and be focused and editable at the
final input boundary. Foreground identity and geometry are checked again before
the fixed XTEST command. Typing inserts at the application's current selection;
it does not implicitly focus, select all, replace a field, use the clipboard, or
retry an uncertain action. `set_value` retains its separate accessibility-setter
semantics. A failed setter never automatically triggers typing. Both operations
need fresh observed effects and independent verification. The explicit Chromium
adapter does not advertise this native keyboard verb.

## Portal tool

The schema is server-pinned in `context.json`. Browser discovery offers
`computer_use`; `surface` chooses `browser` or `desktop`. Desktop discovery
prefers `native_ui`, a fixed-desktop alias using the very same session and
controller. The retired `gui_interact` entry returns a canonical observation
and typed migration instructions. Its old coordinate, drag, text and keyboard
arguments never dispatch input.

In the portal's automatic model loop, these schemas are projected to a
**phase-planning interface**: bare `observe`, `workflow`, `verify` and `cancel`.
Omni supplies semantic steps and expected effects for a bounded phase. The
existing controller resolves and executes the steps internally, using direct
unique matches, the fast policy for ambiguity and the configured visual layer
for missing semantic targets. It freshly observes and verifies before advancing,
and returns to Omni at the phase boundary or first failure. Automatic model
calls to `execute` or step-bearing `observe` are rejected before host dispatch;
the schema restriction alone is not relied on as enforcement.

Workflow results include `phase_handoff`, with position-bound verified and
remaining steps, new applied-effect count and independent terminal status.
The failed step remains in the remaining work; a terminal failure still requires
replanning after all step receipts verified. Cached receipts add no new effects.
A completed phase does not verify the whole user goal. See the
[long-horizon evaluation](computer-use-long-horizon-evaluation.md) for the live
acceptance criteria and original task suite.

The `native_task` compiler can also replan after a failed workflow whose
verified prefix contains a new, dispatched and producer-bound applied effect.
This is an execution-recovery allowance: the failed workflow, unresolved tail
and whole-task verification status remain unchanged. Repeated semantic
action/effect transitions do not earn another allowance through new action IDs
or changed expectation wording. Three consecutive unsuccessful phases without
such a new transition still stop the task; cancellation, the deadline and the
40-phase ceiling remain in force. Existing successful knowledge phases keep
their separate behavior.

Foreground text API callers can explicitly select
`portal_execution_profile:"native_task"` for mandatory phase compilation.
The compiler receives the same canonical phase schema in its model context and
decoder constraints, with full host validation before dispatch. Ordinary portal
clients currently use the generic automatic loop; they do not select this
profile. Live Ornith evidence establishes valid phase generation and controller
handoff, with zero dispatched inputs in the retained long-task attempts. It
does not yet establish successful action chaining.

Each goal starts a fresh desktop controller and captures that controller's
cancellation event. Native cancellation and Trash latch the outer goal as well
as its UI executor. Planning and execution callbacks stay bound to that session,
surface and controller epoch, so renewal cannot turn a cancelled goal into new
work. A late timer or generator cleanup only cancels its captured epoch; it
cannot cancel a later goal. Browser state and session memory survive a new
native goal. Already admitted input or model transport may settle before the
next cancellation boundary; cancellation does not claim atomic rollback.

Direct tool clients and client-owned adapter loops retain the canonical API:

1. `operation: "observe"` returns semantic state. Supply `step` to enumerate
   finite candidates, including `observe` and `abstain`, plus
   `selected_action_id`. Unique controls skip decision inference; ambiguous
   controls use the configured policy and return an actual `policy_receipt`.
   `selected_kind`, `executable_candidate_count`, and `dispatch_status` distinguish
   an offered action from abstention and from input that actually occurred.
2. `operation: "execute"` accepts only a returned `action_id`.
   Rejected and abstained requests report an explicit error and
   `dispatch_status: "not_dispatched"`; they cannot establish task progress.
3. `operation: "workflow"` accepts 1–25 typed steps and independently resolved
   `final_predicates`. It stops at the first unverified step.
4. `operation: "cancel"` cancels the current operation. A fresh observation
   starts a new controller for ordinary unscoped operations; old candidate IDs
   remain invalid. A scoped native task stays cancelled until its captured
   epoch ends or the host explicitly starts a new task. Observe cannot silently
   remove its window restriction.
5. `operation: "verify"` takes `predicates` and performs a fresh independent
   observation. It returns a bound verification receipt only when those
   explicit predicates hold; unresolved checks remain failed evidence.

Example step:

```json
{
  "id": "title",
  "verb": "set_value",
  "target": {"role": "textbox", "name": "Title"},
  "args": {"value": "Verified note"},
  "expect": [{"kind": "value", "node_id": "$target", "value": "Verified note"}]
}
```

Supported predicate kinds are `value`, `checked`, `selected`, `focused`,
`text_selection`, `active_window`, `visible`, `text`, and `url`.
`active_window` verifies that the exact observed window node is the current
scope root. `focused` verifies the exact node's reported keyboard-focus flag;
an active window can have that flag false while an editable child has it true.
A provider-reported focused window remains valid. Use `switch_window` with
`active_window` for activation, and request `focus` only from a control that
advertises that operation. Neither a click nor activation proves keyboard
focus, and typing still needs a separate value readback. These distinctions
are included in both canonical predicate schemas and the native-task planner's
model-visible response schema; predicate evaluation and input authority are
unchanged. An already-active verification gives no mutation or task-progress
credit.
A terminal predicate cannot use `$target`.
File creation still requires the existing filesystem receipt/readback path;
clicking Save alone does not prove a file exists. Session Trash/TTL removes
candidates and outcomes and cancels outstanding work.

Prefer a workflow after initial observation when the causal sequence is known.
Use observed roles and names for unique controls; copy a full current opaque ID
only when disambiguation needs it. A click does not promise keyboard focus.
Expected effects must refer to the control whose state actually changes, which
can differ from the clicked button. Native `click` moves the physical pointer;
an accessibility `invoke` can activate a control without pointer movement.

Policy requests preserve the exact admitted step, candidate arguments/effects
and relevant control values/geometry so repeated labels remain distinguishable.
The portal's existing exact-user context supplies the task goal outside model
arguments. Calls without that context explicitly identify their host-admitted
step context. Selection and point receipts remain session-owned; a receipt is
evidence of the model request, never permission to execute input.

Policy projection v2 factors identical fields and uses request-local node
references. Exact offered action IDs, observed semantic differences, arguments,
and effects remain intact. Execution retains the original scope, revision,
generation and identity bindings. This representation does not increase Kev's
token limits or lower its confidence threshold.

The background controller recognizes the `operation` selector and exposes only
observation/verification in its audit grammar. Executor-bound UI effects advance
the current UI environment version without inventing changed filesystem paths.
Ordinary observations and verified no-ops cannot become mutation milestones.
An independent fresh verification must match the same resource before completion.

An optional System One compatible local decision endpoint is configured with
`OMNI_COMPUTER_USE_POLICY_URL`. It can use the existing Laya service or another
compatible local backend. `OMNI_COMPUTER_USE_POLICY_MODE=shadow` is the default:
ambiguous predictions are observed while unique-match execution remains deterministic.
Explicit `active` mode accepts only a complete valid distribution over the
offered IDs above the confidence threshold; all other results abstain. This
port makes no claim that the UI policy is calibrated across applications.

The supervised local trained-audio-bridge route enables
`OMNI_LANGUAGE_LLAMA_CONTEXT_PROBE=1`. Before language generation it renders and
tokenizes the exact request with that worker, reserving the requested output
budget. Older completed groups can be retired; the current user request,
policy, tool schemas and newest evidence remain intact. Incomplete function
arguments fail before tool dispatch. External OpenAI-compatible services and
Ollama retain their existing behavior unless explicitly configured for this
local llama.cpp contract.

## Explicit visual inspection

`native_ui {"operation":"inspect_visual"}` explicitly requests one bounded reading
from the configured pinned local Moondream `/observe` endpoint. The mandatory
native-task compiler can request `{"kind":"visual_inspection"}`. Neither accepts
a question, image, coordinates or model selection. Ordinary observations and
pointer grounding are unchanged. Browser-surface `computer_use` calls cannot
request this desktop capability.

For a positively identified active DocumentWeb, the reader receives the exact
visible intersection of that node's native bounds with the verified root image.
Node ID, generation, scope, committed DocURL and provider identity must agree;
an unproven document binding never supplies source authority. Other desktop
reads retain the existing guarded capture and exact pixel check. Pointer
captures and input freshness checks are unchanged. A visible
protected control intersecting the capture, or with unknown bounds, also
withholds the read. Native truncation alone permits the explicitly authorized
local inspection, with unknown coverage and no absence claim. No masking,
automatic retry or input fallback is used.

Accepted `reading` and `visual_inspection_receipt` are untrusted producing-tool
snapshot evidence. Native, document, owner, process start, root, surface, region,
protected-control and cancellation checks still apply after inference. If only
the document pixels changed, `status=captured_snapshot` preserves knowledge
about the original image, with both frame hashes and capture/check times.
Unchanged pixels retain `status=observed` compatibility. Both statuses explicitly
carry `snapshot_only=true`, `temporal_scope=captured_snapshot`,
`current_ui_verified=false` and `input_authority=false`; neither attests that the
reading describes the current screen. Nonpixel changes withhold the reading.
Only positive native DocURL proof supplies a source URL. These results
are never a current user image, speech transcript, input authorization, mutation
receipt or independent verification. No image bytes enter the Omni prompt.
Only an accepted nonempty reading with its typed snapshot receipt makes that
knowledge phase successful; it grants no input, mutation or task-progress credit.
Whole-task completion remains independently scored.

## Validation

Run `./scripts/validate.sh` for controller, adapter, session, policy, deployment,
and existing media/tool regressions. Live fixture runners record actual
completion evidence under ignored `runtime-data` directories. Live results
must identify the exact selected model and leased GPU; historical Omnius
success is provenance, not evidence of success on this host.

The real desktop suite runs the existing application runners sequentially for
one exact deployed model. It covers Gedit selection, word double-click,
replacement/save, context menus and the complete spelled-word task; Calculator
arithmetic, visual grounding and required ambiguous policy selection; Files;
and installed default-profile Firefox documentation plus a second spoken
memory-recall turn. Its task source is actual generated TTS WAV passed through
ASR and the ordinary tool loop.

```bash
.venv/bin/python -m runtime.verify_real_desktop_suite \
  --model '<exact deployed model tag>' --endpoint http://127.0.0.1:8920 \
  --token-file runtime-data/state/access-token.txt
```

The suite stops at the first failure. Use repeatable `--case` selections for a
fresh subset run, such as `--case gedit-double-click-word`; `--help` lists all
cases. Every run creates a new ignored `runtime-data` directory containing a
compact `manifest.json`, application artifacts, and captured child stdout and
stderr. Earlier failures remain intact. Unselected or interrupted cases remain
unrun, and `full_scope_passed` requires every case to pass in that manifest.
A subset can report `selected_scope_passed` while full scope remains incomplete.
`--keep-going` continues only when the prior runner confirmed its owned window
was closed. The wrapper does not start services or load models.

The [100-case live acceptance plan](computer-use-100-case-plan.md) extends this
ten-case baseline to pointer/keyboard interaction, native controls, files,
real browser/documentation navigation, sourced memory, live Kev/Moondream,
physical capture and cross-application workflows. It specifies 200 results
across Qwen and Ornith on the operator-selected GPU 1. Readiness labels
distinguish the ten existing runners from cases still needing orchestration;
the plan itself supplies no passing live results.

All real runners default to native `think:false`. For an explicit Ornith
deliberation comparison, select its already-running endpoint, exact model tag
and token file, then add `--think`; for example, select
`--case gedit-double-click-word --think` in the suite. The same flag reaches
every selected child and both Firefox speech turns unchanged. Manifests,
runner summaries and speech evidence record the exact boolean. Use a new run
for each mode so failures and successes remain distinct.

Speech synthesis has its own 180-second transport budget. Each action turn
uses its requested positive finite deadline, including a watchdog that closes
the current request socket if reads stall. A deadline remains a failed case;
the harness does not retry or substitute a typed transcript. Check aggregate
request idleness before starting another live case after a transport failure.
