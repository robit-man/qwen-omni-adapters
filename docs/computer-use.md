# Semantic computer use

The `computer_use` portal tool ports the finite-action control layer demonstrated
in Omnius's `packages/execution/src/rapid-computer-use/`. The reference is
Omnius main `0ae974bd6c08ed9a129075fd184e63462f6b0021`, including the October
12–14 desktop verification and native-controller integration. This runtime
implements the contracts in Python around the signed-in Linux desktop and its
existing application/browser windows. It does not require an adjacent checkout
at runtime. An isolated browser context is used only when explicitly requested.

The selected Qwen or Ornith Omni model produces a typed plan. The controller
resolves each step against current semantic controls, offers opaque finite
action IDs, and executes a unique match without another reasoner call. Every
mutation declares observable effects. Bounded polling verifies those effects,
and a separate fresh observation verifies the plan's terminal predicates.
Ambiguity, unavailable observation, stale state, cancellation, and missing
effects remain explicit outcomes.

## Host-owned execution

An action ID binds the session, scope, document/control generation, semantic
revision, operation, arguments, and expected effects. Execution accepts that ID
alone. Both the controller and native adapter revalidate before dispatch.
Dispatch uncertainty is retained before verification, preventing a repeated
request from applying the same input again. The host's existing authenticated
tool opt-in grants execution authority; model confidence cannot grant it.

Chromium semantic observations skip screenshots and inference. The revision
hash remains stable when the observed state is unchanged. Navigation generates
a new document identity even if a new page reuses an element ID. Native DOM
input retains the existing visibility, enabled-state, and hit-target checks.
Password/one-time-code controls are excluded from finite candidates. Bounded
or truncated observations cannot prove that an unobserved control is absent.

When an `invoke` step has no native match, a named visual target can be grounded
by the configured local point worker. Its synthetic candidate is bound to the
exact source image and semantic revision. The executor checks both again and
maps image pixels into CDP CSS coordinates before input. It never executes
coordinates supplied by a decision model. This A100 deployment uses
**Moondream 3.1 9B-A2B**, pinned to
`1da27e6c2d2f9118de14560df456881dc62d91f3`, through the native Photon backend.
The worker verifies checkpoint, tokenizer, and runtime hashes before loading.
The existing Moondream2/Transformers backend remains available on Tegra.
Broker hosts prepare the exact scoped lease before point-worker CUDA growth,
prove residency, and shed weights after each request. Tegra retains its own
integrated-GPU residency proof.

Linux native controls use a fixed AT-SPI bridge and native action/editable-text/
selection interfaces. Pointer and keyboard input use guarded native XTEST;
the executor verifies foreground PID/window identity, absolute client geometry,
native hit testing, and actual pointer position. It never retries an uncertain
action using a different input backend. Missing accessibility or ambiguous foreground ownership
is a capability failure. Windows UIA and the architecture's full
alternative-model comparison matrix are outside this tested port.

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
   starts a new controller; old candidate IDs remain invalid.
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
