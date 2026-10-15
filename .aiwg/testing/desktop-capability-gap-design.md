# Bounded native capability gaps for CU012, CU014, CU015, CU022, CU023 and CU029

Status: **proposed; production implementation and live acceptance are pending**.
Date: 2026-10-15. Scope: the six explicit capability blocks in
`runtime/verify_extended_native.py`, traced to
`computer-use-100-cases.json`. This document changes no executable contract.
The 1,786-test source checkpoint does not establish any new live GUI result.

## Acceptance criterion, evidence and bounded hypothesis

The failing criterion is that each named case must be executable through the
ordinary generated-TTS → ASR → finite native action loop using its required
physical method, with an independent effect oracle. The runner currently blocks
all six before application launch because the necessary contracts or oracles
are absent. These are source-proven capability gaps, not measurements of model
ability on an attempted task.

`select_text` currently calls AT-SPI selection methods. `drag_drop` binds two
observed controls, not character positions. `click` has empty arguments, and
`key_chord` releases its modifiers before a subsequent click. `_pointer_dispatch`
requires the client geometry to remain identical throughout ordinary input.
Inactive-window mapping uses `xdotool search --onlyvisible`; retained
`window_visibility` records intentionally grant no input authority. The native
observer does not read AT-SPI `Value`, and the runner has no owned-player oracle.

The hypothesis is that explicit typed operations and narrowly scoped read-only
metadata can close these gaps while preserving current guards. Implement one
contract family at a time, run its focused negative gates and full validation,
then let the designated operator observe one fresh bounded live case. Do not
unblock a runner merely because a verb was added.

| Case | Smallest proposed addition | Independent terminal evidence |
| --- | --- | --- |
| CU012 | `drag_select_text {start,end}` using native character geometry | Actual physical drag; fresh selection 6..18, unchanged buffer/file |
| CU014 | `modified_click {modifier:"Control"}` | Physical held-Control click; exactly alpha/gamma selected |
| CU015 | `modified_click {modifier:"Shift"}` | Physical held-Shift click; first three observed rows selected |
| CU022 | Separate `move_window {direction}` and `resize_window {size}` gesture contracts | Actual frame/header/corner drags; fresh X11 geometry and visible unchanged document |
| CU023 | Physical minimize plus a session-owned retained-window restoration capability | Positive minimized/restored sequence for one exact XID/PID; result 42 retained |
| CU029 | Read-only native numeric range plus owned MPRIS oracle; existing visual pointer verbs initially | Player volume near 0.25, then paused, then original volume restored; sink volume unchanged |

## Shared dispatch boundary

All new input remains on `native_ui` / `computer_use surface=desktop`, using the
same controller, host authorizer, finite candidate pool and idempotency journal.
No model-supplied pixels, raw XIDs, executable code, shell text, arbitrary key
names or application-specific keyword routing are introduced.

The adapter must re-observe and validate the exact session, scope, revision,
node generation, eligibility, actual foreground XID/PID and process start time,
coordinate binding and native hit ancestry at dispatch. Protected, defunct,
disabled, obscured or ambiguously mapped targets remain ineligible. Existing
GTK3 SCREEN and exact GTK4 WINDOW/CSD mappings remain unchanged. Provider or
read-budget errors remain explicit failures rather than target absence.

Use one input transaction across the complete gesture. Retain `_NATIVE_LOCK`
and add a desktop-session lock if inputs can originate in multiple processes:
an owner-only, non-symlink lock under the recovered `XDG_RUNTIME_DIR`, keyed by
display, with a bounded acquisition timeout. Do not serialize unrelated model
inference behind this lock.

Before injecting a press, read actual modifier/button state through fixed
X11 queries. Reject an already-held button or modifier; do not release the
user's preexisting input. Arm release cleanup before the first press, retain
it through cancellation/timeout, and verify final release. A small independent
release guardian is needed if the transaction worker may be terminated after
press: it owns only the specific injected key/button and releases on EOF or a
short deadline. It cannot perform clicks or target actions. Normal cleanup is
`finally`-based; guardian cleanup is protection against worker loss.

The action is reserved as unresolved before input. Once any press may have
reached X11, failure remains dispatched/unresolved, with phase and release
evidence. No automatic replay or backend fallback occurs. Failure to prove
release blocks further session input until clean physical state is observed.
No implementation can promise release while the X server itself is unavailable;
that condition must remain explicit and failed, never reported as success.

Receipts record the chosen `x11_guarded_xtest` backend, host-derived gesture
binding/hash, phases attempted, release confirmation and observation revisions.
They do not alone prove selection, movement, restoration or volume change.

## CU012: physical selection of a text range

Add `drag_select_text` with exactly integer `start` and `end`, using the existing
`0 <= start < end <= 1_000_000` schema ceiling. Native admission additionally
requires the entire range to fit the currently observed bounded Text value.
Initially advertise only for visible, enabled, non-sensitive text controls with
Text/Component interfaces and positively read character geometry.

GNOME documents native glyph extents and inverse point-to-character queries;
glyph geometry is meaningful for visible/showing text.
[Character extents](https://docs.gtk.org/atspi2/method.Text.get_character_extents.html),
[offset at point](https://docs.gtk.org/atspi2/method.Text.get_offset_at_point.html).

At step preparation, read the character count, exact bounded text/range digest,
start and last-character glyph rectangles, and range extents. Normalize through
the already validated coordinate binding. For the first bounded implementation,
require a single visible horizontal left-to-right line, positive non-overlapping
glyph rectangles, no clipped endpoints, and native point/offset roundtrips.
Combining, bidirectional, wrapped, vertical or off-screen ranges stay unsupported
until separately measured; do not infer a path through hidden content.

Compute the drag endpoints inside the native start/end glyph geometry, near the
respective insertion edges, and bind their native offset checks and text hash
into the candidate. Roundtripping a glyph point does **not** prove a toolkit's
caret insertion boundary: the actual first Gedit drag must establish that the
chosen bounded rule produces offsets 6..18. If it does not, preserve that failed
receipt and investigate native caret geometry; do not silently shift offsets,
use `set_selection`, or replace the physical drag with a keyboard action.

Immediately before pressing, re-read text, character geometry, binding and
endpoint hits against the same real text node. Execute left-button down,
bounded motion and left-button up. Recheck foreground/geometry during the
gesture; on change, release and stop. Fresh AT-SPI selection and independent
file readback must prove the exact range and unchanged text/file. Keep semantic
`select_text` separate, so it cannot satisfy CU012's physical-method check.

Required tests: stale text/count/revision/generation, wrong foreground or CSD
binding, unsupported layout, clipped/zero glyphs, out-of-range offsets,
roundtrip mismatch and endpoint occlusion cause zero press; errors after press
still release and never replay; actual mouse phases precede exact independent
selection verification. Existing semantic selection tests must stay unchanged.

## CU014 and CU015: one modifier-held physical click

Add `modified_click` with exactly `modifier`, enum `Control` or `Shift`. Map
these two values internally to fixed native tokens `Control_L` / `Shift_L`;
do not expose arbitrary chord syntax. Offer it only where ordinary physical
click is already positively supported. Existing click/double/right-click
argument shapes remain unchanged.

Resolve and hit-test the control center using the existing native path. Move
there with no modifier held, verify actual pointer window/position and repeat
foreground/geometry/hit checks. Then arm cleanup, press the one modifier, check
the same binding again, perform exactly one left-button press/release, and
release the modifier in `finally`. Verify button and modifier release even
when the click, transport, cancellation or post-check fails. Releasing injected
input is cleanup, never an automatic replay of the action.

For Control selection, the independent final selected-node set must be exactly
alpha/gamma. For Shift selection, observe all five rows and their real order
before the first click, then verify exactly the first three selected rows.
Do not substitute two semantic Selection calls, a key chord followed by a
plain click, clipboard operations or selecting an invented row order.

Required tests: both modifier values and fixed argv ordering; unrelated key
tokens rejected; no keydown on stale/obscured/protected targets or preexisting
held input; guaranteed cleanup paths after every injected phase; guardian
worker-loss behavior; no second click on uncertainty; correct exact selection
and unchanged files. Source tests do not establish that Nautilus's currently
occluded virtual rows can receive pointer input; its hit-test gap is separate.

## CU022: physical window movement and resizing

Use separate operations rather than relaxing `_pointer_dispatch`:
`move_window {direction:"left"|"right"|"up"|"down"}` and
`resize_window {size:"smaller"|"larger"}`. The host computes a bounded relative
destination from the observed window and monitor work area. Suggested initial
limits are one move of at most 160 pixels and one size change of at most 20%,
with a 320×240 minimum and the complete client kept inside the work area.
These are proposed operator policy limits, not measured support claims.

Read exact WM identity, client/frame geometry, `_NET_WM_ALLOWED_ACTIONS`, work
area, frame ancestry and CSD/server decorations. Require positive move/resize
support and normal, resizable state. A native `title_bar` or independently
grounded current-frame header target supplies the move grip; the resize grip
needs a positively mapped frame corner. An empty rectangle near an edge is
insufficient proof. Missing native frame/grip identity requires a read-only
operator diagnosis before capability advertisement.

Before button down, preserve exact foreground, geometry and hit checks. During
this distinct transaction, geometry may change only for the same XID/PID/start
and admitted transform, within the host-derived envelope. For a server-owned
frame or WM pointer grab, verify the fresh frame ancestry/grab owner rather
than requiring the application's client XID to remain the pointer window.
Other foreground, frame-parent, monitor/work-area or process changes abort with
release. Ordinary control clicks/drags retain their strict geometry equality.

Complete the gesture using real titlebar/corner down-motion-up input. Do not
use `windowmove`, `windowsize`, `_NET_MOVERESIZE_WINDOW`, an accessibility action
or a WM command as a substitute for the required physical drag. EWMH distinguishes
direct geometry requests from interactive move/resize and declares frame
extents; these specifications inform observation, not a substitute executor.
[WM interaction specification](https://specifications.freedesktop.org/wm/latest/ar01s04.html),
[window properties](https://specifications.freedesktop.org/wm/latest/ar01s05.html).

Require fresh independent position/size changes in the requested direction,
visible unchanged document and exact file bytes. Add typed observed window
geometry/state fields and effect predicates if controller-level completion is
needed; a focus or unchanged-text predicate cannot verify a window transform.
Geometry values in verification are observations, never input coordinates.

Required tests: false frame ancestry, unsupported WM action, absent grip,
maximized/unresizable state, unexpected geometry delta, occlusion or ownership
change reject/abort; legitimate bounded same-window changes succeed; cleanup
survives cancellation; all old ordinary-pointer geometry tests still reject
movement. A live exact-frame/grip diagnosis must precede a real CU022 attempt.

## CU023: minimized-window authority without general desktop input

Minimization may use a physical click on a freshly bound native/visual minimize
control. Restoration needs a distinct retained-window capability because the
application is no longer foreground or visible. Do not turn existing
`window_visibility` history into input authority, remove visible eligibility
checks globally, activate an arbitrary XID, or operate whichever app becomes
foreground after minimization.

Mint a short-lived session-owned capability only for an admitted exact owned
window, with XID, PID/start, native generation and pre-minimize state. Limit it
to a small set (for example four) and expire on cancellation, Trash/TTL, owner
change or completion. A private lifecycle observation can read that exact
hidden window's WM/native state without traversing unrelated foreground apps.

Proposed public operations are `minimize_window {}` and `restore_window {}`.
A restore candidate targets a **visible bound desktop proxy control** associated
with the retained window, not the hidden node. This keeps ordinary candidate
visibility requirements intact. Bind both the target application's identity
and the proxy's current compositor/dock XID/PID, native generation, geometry,
scope and hit ancestry. Capture only the authorized proxy region when visual
grounding is necessary; no general full-desktop point capability is granted.

The positive proxy-to-window mapping is the gating prerequisite. An actual
taskbar/window-list item with authoritative window identity is preferred.
An optional WM icon geometry or a same-name application icon alone is not
enough. An app icon can be considered only when the desktop application mapping
and exactly one managed client uniquely prove the retained XID; multiple
windows, duplicate titles or missing mapping remain blocked. No GNOME Shell
Eval, arbitrary compositor code, title-only guess or repeated activation probe.

The restore executor performs one real click on that bound desktop control.
Verify the original exact XID/PID/start becomes foreground, minimized state
clears and the result remains 42. A scope transition into this private lifecycle
mode must be explicit and session-owned; it cannot expose unrelated desktop
controls. Retained hidden nodes are observation-only and offer no ordinary
click/typing operations. Add a `window_state` predicate with fresh positive WM
and native state; absence in a truncated tree remains insufficient.

The result-42 precondition also needs a real model-owned prerequisite TTS/ASR
turn; host setup may open the Calculator but may not enter 42 itself. Verify
42 before the primary source instruction and keep prerequisite receipts
separate. Every required primary speech artifact remains attributable to that
primary turn.

Required tests: expiry/session reuse, XID/PID reuse, hidden-node input, wrong or
ambiguous proxy mapping, preexisting user-window collision, stale dock geometry,
occlusion and unrelated foreground transitions; one physical restore click
and positive exact-window restoration; no automatic `windowactivate` fallback.
The sole live operator must first measure available GNOME proxy metadata.
If it exposes no authoritative mapping, CU023 stays blocked while an explicit
compositor integration is designed; keyboard restoration cannot pass it.

## CU029: owned player and read-only numeric/player evidence

VLC 3.0.20 is installed on this host. `gdbus`, `busctl`, GI AT-SPI, dbus-python,
libX11 and libXtst are present; dbus-next and playerctl are not required.
This was established through dependency metadata/help only, without opening
a player. Live VLC accessibility and MPRIS registration remain unmeasured.

Add optional `numeric_range {minimum,maximum,current,increment,orientation}`
metadata for non-sensitive native Value controls, rejecting non-finite or
inconsistent values. Keep string `value` semantics unchanged; remove numeric
metadata from protected controls and project it faithfully into policy context.
[AT-SPI Value](https://docs.gtk.org/atspi2/method.Value.get_current_value.html).

The initial smallest input path can use existing physical click/drag and
current-frame named Moondream grounding. Do not add `pointer_set_value` until
native track/handle endpoints and their mapping are positively available.
Interpolating the whole widget width from its numeric range is unsafe: VLC's
SoundSlider uses internal padding and can have a maximum above 100%.
[Official VLC slider implementation](https://raw.githubusercontent.com/videolan/vlc/3.0.x/modules/gui/qt/util/input_slider.cpp).

Setup may generate a short local tone and open a normal separate VLC process
with the installed `--no-one-instance`, `--dbus`, `--no-volume-save` options.
Preserve user player windows and settings; register exact new XID/PID/start.
Do not change default audio devices or global sink volume. The model changes
volume/pause only through real GUI input, not D-Bus setters or playerctl.

The independent observer resolves a fresh unique MPRIS bus owner, validates its
Unix PID and process start against the owned player, and reads only Volume,
PlaybackStatus, CanPause and metadata needed to prove the owned tone. Revalidate
the unique owner on every observation. Do not attribute the first VLC-like bus
name to this process or log metadata from unrelated players. MPRIS Volume is
normalized around 1.0 and can exceed it; PlaybackStatus distinguishes Playing,
Paused and Stopped. [MPRIS Player contract](https://specifications.freedesktop.org/mpris/latest/Player_Interface.html).

Record ordered independent values: initial player volume, volume within 0.05
of 0.25, positive Paused state, then original volume restored within the
declared UI/native increment tolerance. Compare read-only system sink volume
before/after. Preserve exact tone hash and unchanged unrelated files. If named
visual grounding supplies a target, require the pinned Moondream model/revision,
frame hash, candidate binding and actual guarded XTEST receipt. No accessibility
Value setter or direct MPRIS mutation can satisfy the physical method.

Required tests: bus-owner/PID reuse, unrelated player, wrong tone metadata,
missing/non-finite range, >100% normalization, wrong playback status,
missing intermediate quarter-volume value, wrong restored preimage or changed
sink volume fail. Initial live discovery must confirm native control identity
and MPRIS owner binding before unblocking the case.

## Atomic synchronization and staged delivery

For each approved public verb/predicate or observation-field addition, update:

- `src/qwen_omni_adapters/computer_use.py`: validation, finite enumeration,
  snapshot constraints, typed effects and cancellation/uncertainty tests.
- `portal/native_ui.py`: advertisement only for proven capabilities, fixed
  bridge validation, fresh dispatch guards, cleanup and bounded observation.
- `src/qwen_omni_adapters/context.json`: both canonical `Operation` / `ui_step`
  schema copies, verb-specific required args, predicates and shared guidance.
  The retired gui alias stays observation-only; it gains no raw input path.
- `portal/computer_use.py`: relevant-state projection, host authority and typed
  effect evidence; include new metadata without weakening existing tool scope.
- `runtime/computer_use_speech.py`: any typed grammar additions; retain WAV-only
  goals, exact native think boolean and no injected source transcript.
- `docs/computer-use.md`, catalog/schema tests, controller/native behavioral
  tests, and `tests/test_extended_native_acceptance.py`: method/effect negative
  controls and exact runner oracle changes. Browser adapters do not advertise
  new native-only operations.

The portable adapter request/response envelope is unchanged by a server-pinned
tool-catalog extension. If an implementation changes that envelope, also update
contract.py, both adapter JSON schemas, protocol docs, adapter_server.py and both
clients as required by AGENTS.md; do not claim an exemption for a real envelope
change.

Suggested order: character drag; modified click (covering two cases); read-only
Value/MPRIS support; actual window transform; retained minimize/restore last.
Each phase remains additive, disabled where capability proof is absent and
independently reversible by withholding its advertisement. No weights, service
topology, CUDA scopes or model confidence thresholds need change.

Dependencies still to verify in read-only live probes: text insertion edges;
Nautilus row hit identity; WM frame/grip and grab ownership; GNOME taskbar proxy
mapping; VLC's accessible slider and its unique MPRIS owner. These are explicit
gates, not assumptions. The sole desktop operator owns every GUI/model probe
and fresh acceptance run. Preserve failed artifacts and the baseline
`think:false` campaign; additional deliberation comparisons stay separately
labeled. A full 200-result pass requires all original independent criteria.
