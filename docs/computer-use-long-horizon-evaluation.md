# Long-horizon native computer-use evaluation

Status on 2026-10-16: live evaluation in progress; no full long-suite pass or
SOTA parity established.
The user's current direction is **text goals into live Qwen/Ornith weights**,
followed by real desktop work. Speech admission is outside this track. Earlier
[100-case speech/gesture results](computer-use-live-validation-2026-10-15.md)
remain historical evidence; repeating those primitives does not establish
long-horizon capability.

The target is reliable task completion with the complete local agent system:
Omni planning, native observation/execution, Kev selection, Moondream grounding,
memory and independent verification. Parameter counts alone do not define this
target. Report auxiliary models separately and do not infer undisclosed closed
model sizes.

## What the current research changes

Use the official [OSWorld 2.0 paper](https://arxiv.org/abs/2606.29537v2) for its
original protocol and the [OSWorld 2.1 release
manifest](https://github.com/xlang-ai/OSWorld-V2/blob/osworld-v2.1/benchmark_releases/osworld-v2.1.json)
for the later release. These are different evaluation cohorts. The paper's
primary budget is 500 steps; its workflows have a median human duration of
about 1.6 hours. A 30-minute local pilot is a reduced adaptation. Its challenge
families motivate cross-source reconciliation, hidden-state recovery, multiple
items, changing requirements and substantive artifact verification.

[OSWorkerBench/UI-Mate](https://arxiv.org/abs/2608.15930) adds sustained office
work across applications. Instruction-only, same-task demonstrations and
related-task demonstrations are separate conditions. [OSWorld-Science](https://arxiv.org/abs/2609.39903)
motivates tutorial transfer and grading actual artifact properties; its original
tasks may permit CLI, so our native-only track cannot inherit those scores.
[WebArena-Verified](https://github.com/ServiceNow/webarena-verified) motivates
structured, independently evaluated final state. [ScreenSpot-Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)
is a grounding diagnostic, not an end-to-end completion benchmark.

Reported small/medium comparators include
[UI-Mate-9B](https://huggingface.co/tencent/UI-Mate-9B),
[UI-Mate-27B](https://huggingface.co/tencent/UI-Mate-27B) and
[EvoCUA-1.5-32B](https://arxiv.org/abs/2607.09773v2). Their published results do
not transfer to this runtime. Use their exact prompts, parsers, budgets and
observation conditions for a matched comparison. Full success, partial credit,
pass@1, pass@k and selecting the best of multiple rollouts are separate metrics.

## Execution and authority

The initial pilot uses authenticated, foreground `/api/chat/stream`, native
`think:false`, `portal_auto_tools:true`, an immutable `portal_tool_allowlist`
containing `native_ui`, `memory_write` and `memory_search`, both shell and
background bridges disabled, `speech_mode:never`, and text response modality.
Server discovery remains available. No audio or `require_speech` is supplied.
The automatic model's native tool contract is a phase-planning profile: bare
observation, bounded semantic workflows, verification and cancellation. The
server blocks per-action `execute` and step-bearing `observe`; the existing
controller owns fresh resolution, deterministic/fast-policy selection, visual
grounding, guarded input and observed effects. The first delegation acceptance
criterion requires a model-authored phase to execute multiple unique inputs
without another Omni call between them. Whole-task artifact checks remain
separate. Direct tool clients retain their canonical per-action interface.
The runner imposes a 30-minute wall deadline and an 80-round pilot ceiling;
these are our limits, not an official benchmark protocol.

All solver interactions occur in real desktop applications and the ordinary
browser profile. No shell, direct file tool, CDP, developer console, application
macro or direct fetch/search may substitute for the GUI work. Tool scoping alone
does not prevent entering code into an application; audit actual method evidence
as well. Record physical pointer/key dispatch, accessibility activation, Kev
selection and current-frame Moondream use separately. A configured model that
was not invoked is not exercised coverage.

Setup may create an empty owned workbench, seed declared input artifacts, and
open blank owned application windows. It must not navigate task sources, write
answers/output files or feed the model a hidden executor plan. Independent
read-only evaluation may inspect saved files, application state and receipts.
Only exact owned windows/transients may be closed. Preserve unrelated windows,
browser tabs/profile state and files.

Use the foreground route for this first native-only cohort. Background-task
scope inheritance is a separate unresolved integration requirement. The current
memory TTL is five minutes without a memory operation: write near completion
and immediately test recall in a fresh text turn with the same cookie. Do not
silently change TTL or count conversation echo as a successful memory search.

Local CUDA runs reserve exactly the authorized GPU UUID through the broker and
retain prepare/ready/residency checks. Observe external inference-hold state;
the runner must not toggle it. Remote Tegra/MD2 configurations are separate
cells from local A100/Moondream 3.1/Kev. Source changes start a new cohort.

## Pilot LH001: documentation reference pack

The complete user goal asks the model to find official Python `Path.mkdir` and
Git short-status documentation through the browser; write sourced explanations
and practical examples in Gedit; create a five-check checklist; use Files to
create `Research` and `Archive`, rename and organize the notes, and preserve an
original while editing an archive copy; reopen both; then store and recall the
paths, sources and verification in session memory.

Mandatory checkpoints are:

1. Both relevant official source sections were actually acquired through GUI
   navigation, with observed source addresses.
2. The saved note correctly explains `parents`, `exist_ok` and the existing
   non-directory case, plus both Git status columns and the untracked marker.
3. Both source URLs and two valid practical examples are present.
4. The separate checklist contains five relevant concrete checks.
5. `Research/desktop-reference.txt` and `Research/checklist.txt` exist with
   correct saved contents; the original temporary note name is gone.
6. `Archive/desktop-reference.txt` is a genuine preserved copy with the required
   archive-only verification line; the Research original lacks that line.
7. Both saved notes were reopened and inspected; the archive note is left visible.
8. An actual memory write records the verified paths and source-bound summary.
9. A fresh same-cookie turn uses memory search to recall those facts without
   another browser/file action.
10. Method/scope and unrelated-state preservation checks pass, and the model's
    final completion claim agrees with current independent evidence.

No checkpoint passes merely because the model asserts it. Score terminal
outcomes and acquisition/method evidence separately. The pilot is an integration
diagnostic, not the long-suite denominator or a published benchmark instance.

### Retained pilot evidence

| Configuration | Result | Evidence and limitation |
| --- | --- | --- |
| Local Ornith, source `92dd678`, text, think false, temperature 0, output cap 8192 | Failed after 52.76 seconds at the 80-round ceiling | Two discovery calls, four native observations and 74 execute proposals. Two proposals copied an offered abstention; 72 copied a never-offered ID. No executable candidate was offered, no input dispatched, no task file created and no authoritative final received. All three owned windows closed, source hashes unchanged, requests idle and GPU scope healthy. |
| Egg31 Ornith setup, source `92dd678` | Setup failed before any model call | The installed Firefox is Flatpak. Its window lacks `_NET_WM_PID`; authenticated XRes and accessibility identities refer to different host processes in the same sandbox instance. Current native identity handling cannot bind them. Zero model/audio/tool/input requests. A launched but unregistered window has uncertain ownership/cleanup and was left untouched. The accessibility prerequisite was enabled once; no source or GPU lifecycle changed. |

Local evidence is retained under ignored
`runtime-data/gpu1-acceptance/long-horizon/ornith-LH001-20261015T232600Z/`.
Remote evidence is under
`runtime-data/gpu1-acceptance/remote-egg31/LH001-ornith-text-20261015T232730Z-c95d81/`.
These failures selected separate hypotheses: enforce planner-to-phase delegation
locally; independently repair authenticated sandbox identity before remote GUI
acceptance. Neither failure is silently replaced by a simpler primitive test.

A subsequent local phase-profile replay also failed: 77 discovery calls and
three native observations, with no workflow or GUI input in 41.67 seconds. The
working view omitted inactive windows, and the generic tool loop never entered
phase compilation. The current explicit `portal_execution_profile:"native_task"`
therefore makes phase compilation mandatory and supplies bounded mapped window
identities. Its inner requests use native JSON-schema output without tool
discovery. The same user goal and native/sampling options remain frozen for the
next live attempt; only the execution profile changes.

The explicit compiler was then exercised against the loaded Ornith stack on
GPU 1, with three retained results:

| Cohort | Observed result |
| --- | --- |
| Mandatory compiler, first deployment | The worker rejected the generated grammar before a plan or input. CPU reproduction through the pinned llama.cpp converter/parser isolated the private `blocked.reason` limit of exactly 2,000 characters. Reducing that limit to 1,999 made the grammar parse; canonical native action contracts were preserved. |
| Grammar-compatible compiler | Twelve planner requests in 64.09 seconds: eleven memory writes, then unterminated JSON rejected by the host. No UI workflow, GUI input or file. Some stored model claims falsely asserted that files had been created; they receive no outcome credit. |
| Model-visible phase contract | The exact response schema was supplied both in model context and decoder constraints. Three valid UI phases of 4, 9 and 12 steps reached the controller in 28.49 seconds; each controller attempt contained zero intervening Omni requests. The first action failed fresh revision validation; the next two attempted to switch to already-active Firefox and abstained. Zero dispatched inputs, new effects or task files. |

The last two runs closed all three owned windows after downstream idleness,
preserved the scoped GPU workers and skipped recall because no artifact existed.
All remain failed LH001 attempts. Contract delivery, valid phase generation and
the controller handoff are verified Ornith milestones; at that checkpoint,
multi-input execution and whole-task completion remained unverified. The earlier
full state for the first stale action was not retained, so its exact changing
field is unknown. Do not weaken
revision validation or infer that cause from the retained target alone.

Compact evidence is retained under ignored
`runtime-data/gpu1-acceptance/long-horizon/grammar-bound-cpu-20261016/`,
`ornith-LH001-native-task-20261016T001159Z/`,
`ornith-LH001-native-task-grammar-bound-20261016T002034Z/` and
`ornith-LH001-native-task-visible-schema-20261016T002601Z/` within the same
long-horizon evidence directory. The latest source gate passed 2,270 tests in
94.19 seconds with 238 unchanged source fingerprints. These CPU gates do not
count as desktop task successes.

The division follows the supplied *Open-Weight Rapid Computer Use* design:
Omni compiles phases and replans, the rapid controller owns each input loop,
unique semantic matches avoid policy inference, Kev arbitrates ambiguity, and
Moondream grounds eligible missing semantic targets. Native guards and fresh
effects remain authoritative. The failed runs exercised neither Kev selection
nor Moondream grounding. The mandatory compiler currently requires explicit
foreground API opt-in; ordinary portal clients do not yet select it. Other
remaining boundaries include causal references to newly created controls,
selection-failure detail in replanning receipts, and verified workflow caching.

Qwen was then loaded on GPU 1 with source `d698cc5` and the same LH001 goal,
schema and sampling options. Its 101.50-second run issued three four-step UI
phases and twelve actual guarded input dispatches. In phases two and three,
Ctrl+A created a new address-bar selection and typing replaced its value and
cleared the selection, with **zero intervening Omni requests**. This verifies
the multi-input delegation milestone. It does not verify complete phases or
LH001: every Return stopped at a URL predicate because native observations
published no `scope.url`, the field read by the canonical verifier. No task
file was created; recall was skipped; all owned windows closed after idleness.

The loaded Python documentation title was observed, but the required `mkdir`
facts were absent from the captured bounded native state. Source acquisition
and comprehension therefore remain failed criteria. The exact traversal cutoff
was not captured and must not be guessed from the `truncated` flag alone.

This run retained full enumeration and pre-dispatch observations in
`runtime-data/gpu1-acceptance/long-horizon/qwen-LH001-native-task-20261016T003811Z/`.
A separate read-only probe of one new owned blank Firefox window confirmed that
the visible `document_web` exposes committed `DocURL:"about:blank"` through
AT-SPI's Document interface. Firefox chrome also exposes a different Document
URL, so navigation evidence must bind the top-level web document to the active
window. The probe made no model request or task-source navigation and closed
its owned window. Its evidence is in
`runtime-data/gpu1-acceptance/gecko-document-diagnostic-20261016T004400Z/`.

The DocURL patch passed 2,301 regression tests in 96.30 seconds, then a live
Qwen replay completed both four-step navigation phases. Fresh committed URLs
changed from `about:blank` to the Python page and then the Git page, with all
step and independent terminal predicates verified and no intervening Omni
requests inside either workflow. Typing alone left the committed URL unchanged;
Return produced the document evidence that satisfied verification.

The full task still failed. It then repeated focus-only phases with no new
effects or files. The operator interrupted the test client after sufficient
stall evidence. The client received thirteen workflows; the server completed
fourteen before noticing disconnect, including one late owned focus. The
server's twenty dispatched outcomes comprise eight navigation-phase inputs
and twelve focus-only inputs. Owned windows and their ownership watch remained
until that late work settled; both queues and trunk slots were then idle and
the windows closed. No GPU worker was stopped. An aborted HTTP stream is not
therefore a synchronous guarantee against another native input.

Evidence is under
`runtime-data/gpu1-acceptance/long-horizon/qwen-LH001-native-docurl-20261016T005809Z/`.
The exact captured focus-state differences affect only IDs and generations of
sixteen hidden tail nodes omitted by intervening bounded planning views;
visible content, focus, URL and capabilities are unchanged. The native identity
history treats unseen identities as newly appearing even after truncation.
Provider raw paths were not captured, so object continuity still needs a
controlled reproduction. This is a separate follow-on defect; revision guards
were preserved. Source comprehension, editor/Files work, memory integration and
whole-task completion remain unverified.

A controlled CPU reproduction subsequently confirmed two identity failures with
unchanged provider objects and paths: generation renewal after planning-view
truncation, and renewal after a temporary inactive-window view without
truncation. The positive control for actual removal and reappearance passed.
The coverage-aware patch preserves unknown membership, retires only from
complete native child boundaries, keeps current-only input handles and bounded
transactional history, and marks cyclic/duplicate coverage incomplete. Its 239
focused CPU checks passed; the full regression gate passed 2,313 tests. The
subsequent live Qwen replay preserved the same IDs and generation 1 across
truncated planning views and inactive-window exclusion. The identity criterion
passed without changing the loaded GPU1 workers or lease.

That replay ended naturally after nine phases and 237.88 seconds. Python
navigation and a native scroll phase completed, and later observations confirmed
the Git document URL. Three failed window-switch phases then stopped the task:
two fresh prechecks rejected a global revision changed solely by one unrelated
inactive window; the middle failure replayed the first cached rejection. The
workbench target remained unchanged, and no native window activation was
attempted. No output files were created. Source comprehension, editor/Files work,
memory integration and whole-task completion remain unverified. Owned windows
closed, both portal queues and the inference slot were idle, and GPU1 worker
identities and the scoped lease remained unchanged. Evidence is under
`runtime-data/gpu1-acceptance/long-horizon/qwen-LH001-native-identity-20261016T012520Z/`.

The next bounded hypothesis scopes prepared window-switch observations to the
typed target query while preserving the active controls and exact revision
checks at both execution boundaries. A controlled pre-patch reproduction failed
in both the real controller and native bridge when only an unrelated inactive
window changed. Target ownership, active evidence and matching-window ambiguity
remain required rejection controls. Whole-task results require a new live
cohort after the scoped patch passes its source gates.

The scoped patch passed 2,341 regression checks after one older CPU fixture was
updated to provide its mocked process-start identity. Its single live replay
ended naturally after four phases and 110.86 seconds. Python navigation passed,
but the planner repeatedly requested `select_text` from 0 to 12,000 on a
`document_web` node exposing five characters. The native range guard rejected
the first selection before input; the next two phases replayed that same
rejection. The model never requested a window switch, so the scoped-switch live
criterion was **not exercised**. There were no output files or memory recall.
Owned windows closed, queues and the inference slot were idle, and the original
GPU1 workers and scoped lease were unchanged. Evidence is under
`runtime-data/gpu1-acceptance/long-horizon/qwen-LH001-native-scoped-switch-20261016T015830Z/`.

## Expanded native suite

The [packaged runner](computer-use-long-horizon-runner.md) contains 100 original
specifications and 200 planned Qwen/Ornith first-attempt cells. Its isolated
regression gate passed 2,374 checks; the combined runtime gate passed 2,445.
These are source checks, separate from model task outcomes. Fourteen cases
still need media or controlled-event support.

The first packaged attempt, Qwen LHN-047, failed independent checks after three
phases and 56.27 seconds. Fresh source proofs, five declared inputs and three
owned ordinary-app observations passed setup. The model's attempt to open the
Files `Input` cell was rejected before pointer dispatch; semantic `invoke` had
no matching capability, and the third phase replayed the first rejection. No
native input was dispatched or deliverable created; all inputs and unrelated managed windows were
preserved, and owned windows closed with queues idle. The scoped-switch live
criterion was again not exercised. The immutable bundle is under
`runtime-data/gpu1-acceptance/long-horizon/qwen-LHN-047-20261016T021140Z/`, with
SHA256 `690be8a40578da864b46b4ce8ed0092da668da64bc1a9d604ed993bc9f546c6e`.
One distinct model/case cell has been attempted in two source cohorts; neither
run passed. Failed attempts remain part of the record when later revisions are
tested.

The subsequent capture diagnostic exposed a separate host fault: the Files
surface at `(-17,-23)` measured `972×1184`, while its visible image measured
`955×1161`. Capture now binds the verified root-screen intersection and image
origin separately from the full surface used by dispatch guards. The source
gate passed 2,512 checks. One unchanged production capture then passed against
the real owned Files window, with root `492` measuring `1440×2500`, image origin
`(0,0)` and pixel offset `(17,23)`. Root, surface, process start and native
coordinate mapping remained stable; semantic revision drift is recorded
separately. Evidence is under
`runtime-data/gpu1-acceptance/files-capture-acceptance-20261016T025152Z/`;
the accepted PNG SHA256 is
`da142163bf19a365adc614b2d86f31dae5a355ca6d1532d344585a493ac78491`.
There were zero hit queries, inputs or model calls, and the owned window closed.
This verifies capture geometry only. The Files native-hit fault and whole-task
completion remain unresolved; the suite denominator is unchanged.

One subsequent read-only Files diagnostic tested nine native-derived points
inside the same eligible Input cell. Every native hit reached the same panel
and then NULL. The target, mapping and ownership remained stable; the snapshot
contained 252 nodes without truncation, and the unchanged read epoch ended at
6,173 of 12,000 reads. No input or model call occurred, and the owned window
closed. This establishes unresolved native targeting, rather than physical
occlusion. Evidence is under
`runtime-data/gpu1-acceptance/files-native-hit-points-20261016T030415Z/`.

The resulting pre-selection handoff preserves the semantic handle and delegates
only an eligible requested pointer capability to pinned Moondream. Incomplete,
protected, stale or ambiguous evidence withholds that route. Grounding metadata
is excluded from semantic mutation counts. Both defects reproduced before the
patch; 355 focused checks and the full 2,568-check source gate passed afterward.
The unchanged LHN-047 replay on `24093a1` reached two real pinned Moondream 3.1
point requests, taking 14.062 and 3.720 seconds respectively, including loading,
inference and shedding. Both returned the same point inside the Input cell.
The action was rejected before dispatch when whole-frame revisions differed;
two later focus phases abstained. The run ended naturally after three phases
and 82.48 seconds, with zero inputs, outputs or recall. Owned windows closed,
inputs and unrelated windows were preserved, and queues and workers stayed
healthy. Evidence is under
`runtime-data/gpu1-acceptance/long-horizon/qwen-LHN-047-20261016T033016Z/`.

Paired production PNGs had identical semantic/probe/owner/geometry bindings.
Exactly 199 changed pixels were in the desktop clock outside the verified
application content rectangle; zero content pixels changed. A derived analysis
receipt was accidentally written after sealing inside the run folder. The
initial immutability failure is retained; only that extra file was relocated
unchanged to the sibling operator folder, restoring the original member set
without rewriting any original file or manifest. The incident audit and final
failure note remain outside the bundle. No solver input or workbench changed.

The next patch crops the accepted image to positively verified native content
bounds, retains original-source provenance, and compares the exact transmitted
region. It passed 441 focused checks and the full 2,601-check source gate.
The `7c9b79a` replay then made one actual pinned Moondream 3.1 point request
(HTTP 200, 3.968 seconds) and dispatched a physical double-click through guarded
XTEST. An independent native observation confirmed that the same owned Files
window changed from Workbench to Input and exposed all five input files. The
transmitted content image stayed identical across three captures while the
original root-image hashes changed. This verifies the content-bound grounding
and physical directory-opening milestone.

The full task still failed naturally after three phases in 74.21 seconds. The
first plan targeted Input but expected that disappearing directory cell to
remain focused; verification correctly remained unresolved. The next two plans
requested `switch_window` on the already-active Input scope. Its observed
operations did not include that verb, and both attempts abstained without input.
The workflow dropped the selector's typed status, leaving opaque abstentions
in replanning feedback. There was one physical input, no output artifact and no
memory recall. Evidence is under
`runtime-data/gpu1-acceptance/long-horizon/qwen-LHN-047-20261016T040026Z/`;
manual notes remain in its separate sibling operator directory. This is another
failed attempt of the same model/case cell, not another distinct suite task.

The next source, `bc33666`, retained factual per-step selection feedback and
passed 2,617 regression checks. Its full replay verified the physical Input
double-click, and the actual second planner request contained the corresponding
`semantic_unique` selection receipt. Qwen then opened `task-brief.txt` with a
second physical double-click, without requesting an active-window switch in
this trajectory. The unsupported-switch feedback path was not exercised live,
so this is not evidence that the feedback caused that planning difference.

The task still failed after 57.75 seconds: post-click native observation became
unavailable, and the third phase stopped at its observation callback. No output
or memory result earned completion credit. The exact historical native error
code was not retained. A later pinned read of the newly observed GNOME Text
Editor succeeded in 0.62 seconds with 375 nodes and truncation, establishing
current capability rather than the cause of the earlier failure. Its window
remains untouched because solver ownership is unproven. Registered windows
closed, queues became idle, and workers, lease, source, unrelated managed
windows and the sealed evidence bundle were preserved. Evidence and separate
operator notes are under
`runtime-data/gpu1-acceptance/long-horizon/qwen-LHN-047-20261016T041638Z/`
and its sibling operator directory. Source and transmitted PNGs are both
retained in this cohort.

The explicit GNOME Text Editor setup patch, `f0505fa`, passed 2,661 source
checks. Its next full LHN-047 attempt registered a new owned editor XID
`81788956`, sharing PID/start with the preexisting, unowned XID `81788932`.
Setup passed, but Qwen's first phase selected that older window. The native
controller offered the switch, and fresh verification showed that exact older
window in the foreground before the runner's passive guard rejected it. The
client cancelled after 22.33 seconds; the
entire run took 29.41 seconds and received no final response or completed tool
receipt. The sealed method index contains zero completed inputs, but the fresh
verification snapshot proves the foreground switched: zero indexed receipts
must not be reported as zero effects.

There were no deliverables or observed text mutations; all five inputs were
unchanged. The three registered windows closed politely, the preexisting editor
remained, and the original workers, lease and idle queues were preserved.
The full task failed; the separate host editor setup milestone passed. Opening
a file into a newly authorized editor window was not exercised. Evidence is
under `runtime-data/gpu1-acceptance/long-horizon/qwen-LHN-047-20261016T045525Z/`,
with immutable bundle SHA256
`bca7778ef4dd0bf9a9ec1da3a432ccadf57472c785a0cb17e00ef121be65230c`.
Manual notes remain in its sibling operator directory.

The next bounded hypothesis is that controller planning/candidates and the
runner must share the same immutable window authority. A portal-only scope
derived from fresh setup must exclude that preexisting same-PID window before
it can be offered or activated, with authorization repeated at dispatch. The
backup foreground guard remains. This addresses an authority mismatch; it
does not establish long-task completion, visual reading or Kev coverage.

The scoped source, `a4df402`, passed 2,723 checks with all 360 source
fingerprints unchanged during validation. Its single unchanged full LHN-047
replay verified the scope milestone: all four actual planner packets carried
the same digest/count, excluded the preexisting same-PID editor, and contained
no raw authorization list. Two real pinned Moondream-grounded XTEST
double-clicks opened Input and then `task-brief.txt`; independent observation
and the fourth planner packet confirmed the brief in owned GNOME XID
`81788968`. The backup foreground guard reported no failure.

The full task failed naturally after four phases in 110.31 seconds. Phase two's
step verified, but its terminal observation crossed from Files into the editor
and remained unresolved. Phases three and four requested `switch_window` on
the already-active editor. Both enumerations offered only observe/abstain.
The actual fourth planner packet already contained correct feedback identifying
the target as the current scope and listing its available operations without
`switch_window`; the model repeated the unsupported request. This is not a
missing-feedback claim. Later keyboard steps, including guessed selection
ranges, never ran. There were zero Kev calls, keyboard inputs, deliverables or
memory recall. No long-task or policy-selection pass is inferred.

All inputs and preexisting windows were preserved, registered windows closed,
and queues, original workers, source and the same GPU1 lease remained healthy.
Evidence is under
`runtime-data/gpu1-acceptance/long-horizon/qwen-LHN-047-20261016T052849Z/`,
with immutable bundle SHA256
`a464fada5d51743cf2807d3cdf26cf01dc93a5d40932fca495c5e1cee93c87f5`.
The separate operator note and actual phase/feedback diagnostic remain in its
sibling operator directory. Scope enforcement and owned file opening passed;
whole-task completion remains unresolved.

The already-active window patch, `b13bff5`, passed 2,870 source checks and a
before/after controller reproduction: an exact already-active request received
an explicit no-input receipt, then the following typed step ran and fresh
terminal verification passed. Its single full live LHN-047 replay did **not
exercise** that path. During the first compiler request, the CPU frontend's
watchdog raised `GPU health or quarantine blocks live acceptance` and stopped
its frontend. The failing discovery snapshot was not retained, so the precise
blocking condition is unknown. Fresh broker, worker and kernel checks showed
no current hardware fault or quarantine; this does not establish the historical
cause or make the stop a proven false positive.

The audit captured one planner request and one planning observation, with zero
phase replies, tool starts, point requests or controller dispatches. No output
was created. The runner exhausted its 300-second drain bound against the dead
frontend and sealed `cleanup_deferred` / `downstream_idle:false`, taking 322.67
seconds in total. That failure record remains unchanged. Separate operator
recovery proved frontend/native children gone, the original portal and trunk
idle, and adapter transport drained before politely closing exactly the three
registered setup windows. All preexisting windows, inputs, original workers,
source and the same GPU1 lease were preserved. No inference retry occurred.

Evidence is under
`runtime-data/gpu1-acceptance/long-horizon/qwen-LHN-047-20261016T055452Z/`,
with immutable bundle SHA256
`15707460477aff783f28253e2b61b27eb152f1222c8a722e692914b1461050b1`.
The separate watchdog diagnosis and operator cleanup receipt remain in its
sibling operator directory. Whole-task result is failed/incomplete; live
already-active verification remains unexercised. The next diagnostic change
must preserve the actual sanitized blocker evidence without relaxing admission.

The explicit native visual reader, `0922eba`, subsequently passed 2,912 source
checks with all 364 fingerprints unchanged. Its closed `visual_inspection`
phase delegates to the existing pinned `/observe` worker without input, raw
image promotion or mutation credit. Focused checks bind reading to fresh native
and document identity, exact transmitted pixels and producing-tool authority;
stale, protected, cancelled or invalid output is withheld. All three changed
schema objects passed the pinned llama.cpp CPU converter/parser with zero
warnings. These establish source and grammar compatibility only. No live reader
request or new desktop task ran at this checkpoint; local inference remains
pending watchdog evidence retention and fresh admission.

The admission evidence patch, `fbc381f`, passed 2,928 checks. The isolated
validation checkpoint and all 366 tracked/private-helper fingerprints remained
unchanged. Blocked admission now retains fixed blocker codes and bounded
boolean/count/UUID evidence from the same discovery snapshot; malformed
responses remain fail-closed. The watchdog serializer accepts only that trusted
diagnostic type and continues to redact generic exceptions. The guard, discovery
timeout, GPU scope and residency requirements are unchanged. Its saved-JSON CPU
check proves evidence retention, not GPU health or a desktop outcome. Receipt:
`/tmp/qwen-broker-evidence-post-receipt.json`, SHA256
`e6911be4c294eb00682ca9e8b0cff1e51f7bf01e6452ef6521675b32770a910f`.
The historical interruption's cause remains unknown; no inference or service
adoption occurred during this source validation.

The single CPU-only adoption at `84f3c55` then passed four read-only admission
samples over 30.873 seconds, preserving all eight original worker identities,
the same GPU1 lease and idle queues. At 06:33:00.925 UTC on 2026-10-16, its
watchdog stopped only the test frontend. The new trusted diagnostic retained
`unregistered_gpu_quarantine`, count one, with availability true, admission
unblocked and no missing selected GPU. LHN-001 subsequently failed preflight:
zero applications launched, zero model requests and no solver case attempted.

Bounded broker journal evidence identifies GPU0 and GPU2 quarantine intervals
in that event window, with neither including leased GPU1. Installed broker
policy and transition code block the affected GPU intersection; the acceptance
watchdog instead rejected any quarantine globally. This is a demonstrated scope
mismatch for this interruption. Its actor identity was not recorded, and it
does not establish the cause of the earlier 05:55 failure. Fresh quarantine had
cleared; original workers and lease remained intact, with no new hardware fault
evidenced. No reset, inference retry or frontend restart followed diagnosis.
The separate evidence and per-test note are under
`runtime-data/gpu1-acceptance/LHN-001-preflight-block-20261016T063519Z-operator/`.

The scoped quarantine correction, `ca863ab`, and inactive Flatpak mapping,
`2803b0e`, passed the combined final source gate: 3,019 checks, unchanged
validation HEAD and all 366 tracked fingerprints. The first mapping gate's
fixture-interface failure and a focused test's random substring collision were
retained and corrected in tests; production mapping stayed unchanged. Final
receipt: `/tmp/qwen-flatpak-inactive-b13/runtime-data/inactive-flatpak-design/final-full-post-receipt.json`,
SHA256 `360120c542edbfcd78a34767a4755a400e8428f0f57213fb9351860766eb5445`.

The watchdog now accepts only complete, canonical broker quarantine inventories
proved outside GPU1; affected, unknown, malformed or oversized scope remains
blocked, as do the original health and residency failures. Diagnostic records
retain sanitized quarantine UUIDs and scope-verification flags. Inactive
Flatpak windows require full namespace/XRes identity and complete, unique frame
geometry before optional scope filtering; private proofs bind revisions and
refresh before activation. Unknown mappings grant no handles. Existing current
window verification, cancellation, idempotency and native budgets remain.
These are source-verified capabilities. No new live task or remote switch ran
during the gate; owned Flatpak launch/cleanup remains a separate setup gap.

The next full Qwen LHN-001 run on `ec12821` reached nine phases in 263.65
seconds (269.12 seconds including the runner). It failed with no dossier,
matrix, archive or memory completion. Eleven outcomes dispatched real native
input. Two pinned Moondream-guided physical double-clicks opened Input and the
task brief. An already-active window request returned one independently
verified no-input receipt with absent effect; the workflow continued. The
model recovered from its incorrect selection-range expectation without help,
then switched to the owned Firefox window and physically typed/submitted the
official Python 3.10 URL in one compiled workflow. Committed DocURL verification
passed. Kev remained unexercised.

All three subsequent explicit visual-reading attempts failed the final pixel
key comparison after valid configured reader replies. Native/document identity,
semantic revision and capture geometry remained stable. This Firefox capture
had no native content inset, so its transmitted image included the desktop
clock in the window shadow margin. Saved pair differences were 135, 328 and
965 pixels; the corresponding counts inside the positively observed
DocumentWeb rectangle were zero, 200 and zero. The in-document change is a
visible scrollbar thumb. Therefore excluding the desktop clock alone cannot
be claimed to repair all three failures. The hook retained exact point HTTP
requests but not reader HTTP bodies/latencies; validated replies are established
by the control flow reaching each post-response capture, not fabricated packet
records. Zero readings were accepted.

Queues drained, exact owned windows closed, inputs and preexisting managed
windows were preserved, and the original eight workers, same GPU1 lease and
healthy CPU frontend remained. Evidence:
`runtime-data/gpu1-acceptance/long-horizon/qwen-LHN-001-20261016T065927Z/`,
immutable bundle SHA256
`4395177a187ff7c4a202516db50c12fbe621c5564320eaabe25467ccf4946719`.
The separate operator note SHA256 is
`c139d322633ce34f8b8d92bff2be3b12ff1ba043bc3e2b05075582cc0d0fda8b`.
Native execution and recovery milestones passed; complete-task and visual-reader
acceptance failed. The next reader criterion is separate from those verified
input capabilities.

The source correction retains a valid, source-bound document reading as
immutable captured-snapshot knowledge when only its document pixels change
during inference. It uses positively observed native DocumentWeb bounds to
exclude surrounding desktop pixels. Changed ownership, document identity,
semantic revision, geometry or other nonpixel bindings still reject the
reading; point/input checks remain exact. Even matching-pixel readings grant
neither current UI authority nor mutation/progress credit. The next live run
will additionally retain actual configured reader HTTP bodies and request
latency; historical reader text remains unknown.

That correction and the explicit [owned Flatpak Firefox setup](computer-use-flatpak-setup.md)
passed one combined full source gate: 3,122 tests in 104.54 seconds, aggregate
checks passing, unchanged validation HEAD `74f6a22` and all 370 tracked file
fingerprints plus the staged private reader hook unchanged. Receipt:
`/tmp/qwen-reader-flatpak-combined-post-receipt.json`, SHA256
`337c288fbb675ac9637d9a16357d959e3f4e0099d6faa4a18b08d00c25545d92`.
The isolated changes were adopted unchanged as `bdf82b6` and `a9d124b`.
This is source validation; fresh live reader acceptance and owned Flatpak
launch/cleanup qualification remain pending.

The next unassisted Qwen LHN-001 run on `eff4fa5` ended after 13 completed
phases: 383.27 seconds for the request, 394.83 seconds including the runner.
It failed with a transport read error and no authoritative model final. The
runner retained an outside-owned-scope foreground diagnostic, cancelled the
task, proved downstream idle and closed only its three owned windows. The
offending historical foreground tuple was not retained; its actor and cause
remain unknown. The CPU frontend stayed healthy, all original worker identities
and the exact GPU1 lease were preserved, and fresh fault checks found no new
GPU fault. No deliverables or reader calls occurred: reader acceptance is
unexercised in this run, not a failed reader response.

Independently of that interruption, phases 5–13 prove a generic repetition
defect. Each repeated the same already-active editor switch, cached focus and
cached Ctrl+A on identical observed state. All nine effect vectors were
`[absent, replayed, replayed]`, with zero new effects. The compiler's all-replayed
check excludes this mixed vector, so complete workflows reset its failure
counter. The separate diagnostic SHA256 is
`d56828818889733c21e5db2e96d65e2c03416bb1bca8e6bf1f5606464b8d735a`.
The immutable failed bundle at
`runtime-data/gpu1-acceptance/long-horizon/qwen-LHN-001-20261016T075430Z/`
has SHA256 `28bb43d4ea942750d6746562a0b3e99f1e829030d0fbe9f579483b4fbd6e01ac`.
No retry followed. A bounded typed repetition correction is a separate source
criterion; first legitimate no-ops and useful knowledge phases must remain valid.

The correction, adopted as `f1b9cf0`, annotates a repeated verified zero-effect
UI phase with `ui_phase_repeated_without_new_effect`, preserving its original
complete workflow and input/verification receipts. The next native-task compiler
receives that nonprogress reason through its ordinary compact result; repeated
failures use the existing three-failure bound. Effective action bindings and
complete existing observations determine repetition, rather than planner step
IDs or claims. Actual mutations clear the history. First valid no-ops, changed
state/actions and separate reader/memory phases remain valid. This applies to
the explicit native-task compiler, not ordinary chat/voice tool-loop routing.
All nine saved repeats produce one signature with no observed fields omitted.

One full source gate passed on frozen `035c03fd`: 3,139 tests in 107.35 seconds
plus aggregate checks, with all 370 tracked files and five reviewed private
helpers unchanged. Receipt `/tmp/qwen-native-phase-repeat-post-validation-receipt.json`,
SHA256 `1719fb1958305c3c823f0c06114ef3284a80a37dd22777afd34585c5bb71ddac`.
The first launcher attempt found no interpreter in the isolate and exited
before running the gate; the recorded passing run used the existing environment
with verified isolated imports. No live adoption or new model run occurred
during this gate. Fresh acceptance is planned on local GPU1 Ornith before the
next remote model-task runs.

The independent run-ledger audit at 07:32:49 UTC on 2026-10-16 found eight
canonical submitted attempts: seven Qwen LHN-047 attempts and one Qwen LHN-001
attempt. Those cover two of the 200 model/case cells, with six repeated attempts
and zero full passes. All 100 task IDs remain incomplete; 98 task IDs and 198
model/case cells have not been attempted. No canonical Ornith run is recorded.
The ledger keeps the preflight block, older pilot runs, remote setup diagnostics
and CPU source checks outside those counts. Its sealed-input audit is retained
at `runtime-data/gpu1-acceptance/operator-audit/long-horizon-ledger-20261016T073249Z/`;
`actual-run-ledger.json` SHA256 is
`6bd0fae81e15b062b271ea7817f8272c7cc6e2685c4b04642eebd341cc19d8fd`.

Layer evaluation follows the architecture guide's finite-action cascade.
Native observation supplies identity, capabilities and current state; the Omni
backbone plans and recovers; compiled workflows execute independently of the
backbone between steps; Kev arbitrates eligible ambiguous candidate sets;
Moondream supplies explicit visual readings or bound points; the executor and
fresh observer verify dispatch and effects. A unique valid semantic target may
bypass Kev, and an available semantic route may bypass vision. Model-call counts
alone do not establish either a delegation defect or successful coverage.
The current zero canonical Kev receipts require a source-and-trace eligibility
audit before claiming that layer is exercised. Accepted document snapshots must
remain knowledge, never current input authority or terminal mutation proof.

The subsequent delegation audit found 18 direct selector receipts and ten
first-step reconstructions: 21 unique executable matches, seven zero-eligible
matches and no ambiguous candidate set. Excluding the invalid-seal historical
run leaves 25 opportunities, with 20 unique and five zero-eligible. Interrupted
and incomplete evidence limits remain explicit. These counts use evidence
available at the audit's 07:54:47 UTC capture, before the later 13-phase run
completed; they do not count its subsequently retained selection receipts.
This does not demonstrate a
missed Kev dispatch or a successful Kev decision. Audit:
`runtime-data/gpu1-acceptance/operator-audit/kev-delegation-20261016T075447Z/report.json`,
SHA256 `1fd87730bf88ff7f580fcdbe6cdcb05a97c31932a36a05d48e14792d87857bfd`.

After the push, read-only SSH evidence established actual Egg41 adoption of
`eff4fa5`: reflog advance, core restart, new portal/adapter identities and the
matching deployed marker. Its Ornith trunk was healthy and resident; auxiliary
wrapper liveness alone did not prove their CUDA residency. Egg31 remained on
`ec12821`, with its previous worker identities and an existing request left
untouched. Neither observation is a remote task pass. Receipt:
`runtime-data/gpu1-acceptance/egg-auto-adoption-watch-20261016T075600Z/watch-summary.json`,
SHA256 `7c6e070d4fca88b8871d054dc4bf7f146f2b195e9c07e0dc278fd656d51dc0f2`.
The next model-test sequence is local GPU1 Ornith, then live Egg tasks after
their exact source adoption and idle readiness are verified.

### Local Ornith and remote adoption checkpoint, 2026-10-16

The supported local Qwen-to-Ornith handoff freed the old workers and lease
before acquiring the replacement lease on the same authorized GPU1 UUID.
The new Ornith trunk, Kev and pinned Moondream 3.1 stack passed readiness
checks. Moondream sheds its weights between requests; idle worker health does
not establish resident vision weights. Worker identity and residency receipts
are retained separately from task outcomes.

One unassisted local Ornith LHN-001 attempt on `476be1c` failed. The first
model-authored phase switched to its owned Firefox window, then stopped at
`no_matching_control`. The second compilation failed with the native parser
error `The model produced output that does not match the expected peg-native
format.` The task took 30.08 seconds, or 35.67 seconds including the runner.
No deliverable, reader call, point call, Kev decision or memory operation
occurred. The new reader and repetition guards therefore remain unexercised
in this trajectory. Cancellation, downstream drain and exact owned-window
cleanup passed; the new GPU1 workers and lease were preserved. Failed bundle:
`runtime-data/gpu1-acceptance/long-horizon/ornith-LHN-001-20261016T083450Z/`,
SHA256 `7c63d8de363ec1dc7b05bac51bff496062e1d253591b04e7ca2f0a91eb946f90`.
Its separate after-test note has SHA256
`48870ba91f9f7b6e70b36dcf9c339f8e0062c298a6e61e12e90290486499ca2d`.

A separate decoder-only replay increased only the actual output cap from
4,096 to 8,192 tokens. It reproduced the same malformed prefix and parser
failure, with no GUI actions or canonical attempt counted. A CPU-only
discriminator then reproduced the production chat template and grammar
without loading weights: the generated grammar accepts the malformed prefix,
while the native parser rejects it. The successful first phase passes both.
The responsible schema pattern `^[^\u0000]+$` becomes a character class that
also accepts unescaped JSON quotes and raw controls, allowing malformed
structure to be consumed as unfinished string content. Raising the token cap
does not repair this defect. The discriminator receipt is retained at
`runtime-data/gpu1-acceptance/long-horizon/ornith-LHN-001-20261016T083450Z-operator/cpu-grammar-discriminator-evidence.json`,
SHA256 `a03a8b76791e5bcb7f85de29d190522cd13ad371300c85ac56a58c6e05030772`.
A source correction and fresh live acceptance are separate pending criteria.

The actual-run ledger now records ten submitted canonical attempts across
three of the 200 model/case cells: Qwen LHN-001, Qwen LHN-047 and Ornith
LHN-001. There are zero full passes; 197 cells and 98 task IDs remain
unattempted. This includes the later Qwen and Ornith attempts through
08:38:53 UTC. Setup failures, decoder diagnostics and CPU checks are excluded.
The append-only update ledger has SHA256
`cb2e361fe3cbd974fbb77dbd865f94df2f59900f33e2774de0fa65bdf6bc3644`.

Egg41's automatic updater adopted `476be1c` after a natural retry passed all
3,139 tests and restarted the core service. Its earlier invocation failed one
background-task test; that failure remains retained and unexplained. Egg31's
isolated ARM gate also passed all 3,139 tests; source adoption was initially
held because an existing request remained active. Neither
source validation nor a service restart is a remote task pass.

After the local Ornith run, the Egg41 LHN-041 driver stopped at its initial
HTTP health preflight on 09:20:54 UTC. The canonical runner was never called:
zero applications, native inputs or model submissions occurred. The owned
deployment lock was released, the harness was untouched, and source,
configuration and worker identities were preserved. Its aggregate health
helper did not retain the exact timed-out endpoint; current health and
activity were unknown at that observation. This is a preflight block, not a
failed submitted model task. Receipt:
`runtime-data/gpu1-acceptance/operator-audit/egg41-LHN041-20261016T091053Z/after-test-note.json`,
SHA256 `f6fd97eac418389da1514ef087c7a1dfffd47381c979216a74a4064f3ba08250`.
The remote MD2/no-Kev condition remains distinct from local MD3.1/Kev.

At 09:29:43 UTC, Egg31 completed the separately authorized `476be1c`
adoption after becoming naturally idle. One supported Python reload took
3.32 seconds; new portal/adapter identities, six healthy idle observations,
unchanged loaded supervisor/model/pointing workers and restored harness were
verified. Configuration and all 18,030 preexisting untracked paths were
preserved. The absent full-deployment marker was not fabricated. Compact
adoption receipt:
`runtime-data/gpu1-acceptance/remote-egg31-qwen476-cutover-20261016T092834Z/adoption-receipt.json`,
SHA256 `195f78ec6b8cd4cde6d9d4d190805fd3011a67f404bdc404774bcf8c59786c3c`.

Egg41's subsequent endpoint-specific diagnosis found its core inactive, with
no runtime listeners or access-token file. Its installed unit explicitly
requires absence of `runtime-data/state/voice-service.paused`; that marker
was present and had been created before the recorded core stop. The actor
and reason were not established. The proposed supported start stopped at
this prerequisite: no marker removal, harness stop or core start occurred.
This actual pause remains respected pending the user's direction. Recovery
preflight note:
`runtime-data/gpu1-acceptance/operator-audit/egg41-supported-start-20261016T093113Z/after-test-note.json`,
SHA256 `5a58c477d50c99adcf8ebee4b4fc7451d4c2dfae860d2140a566522b7a278bfa`.

These are original task specifications, inspired by the cited challenge
families. Input fixtures and application/oracle readiness must be recorded
before execution; a specification is not a runnable or passed test. Do not give
the solver the procedure or sealed expected answers. All outputs remain in
owned scratch locations; no real purchases, bookings or messages are required.

| Case | Outcome | Independent completion checks |
| --- | --- | --- |
| LHT-01 Migration dossier | Reconcile three official versioned documentation pages, four behavior changes and supplied examples; write and package a migration note; recall one sourced finding. | Four correct version-bound facts, exact intended package, reopened note and source-bound memory. |
| LHT-02 Accessible visit | Combine official access/hours/collection information with group constraints; produce a Calc schedule/cost sheet and editor handout. | Feasible timing, all mobility/budget constraints, correct formulas and matching handout. |
| LHT-03 Programme repair | Repair a 12-session schedule using capacities and a revised speaker note; export and inspect a PDF. | Every session once, fixed slots preserved, no conflicts, correct change log and export. |
| LHT-04 Workshop kit | Reconcile attendance, stock and official specifications; prepare supply quantities and a handoff checklist. | Correct counts/rounding, budget, source-bound compatibility and consistent artifacts. |
| LHT-05 Mini exhibition | Catalogue six official collection objects, dates, provenance and distinct reuse conditions; organize permitted materials. | Six correct unique records, rights/source binding, matching assets and reopened catalogue. |
| LHT-06 Public handover | Produce public and internal outputs from fictional records, carrying only the specified fields into each. | Required rows/fields, confidential canary absent from public output, originals preserved. |
| LHT-07 Conflicting revisions | Resolve six document families using a ledger and embedded revisions; package current and superseded sets. | Correct authoritative version per family, exact membership, preserved old files and decision log. |
| LHT-08 Editorial pass | Apply ten corrections across three drafts while preserving quotations and similar non-target passages. | Exact intended span changes only, protected text intact, new files and complete change index. |
| LHT-09 Unit-aware data story | Clean a measurement table using official unit definitions; build derived values, a chart and an exclusions note. | Correct rows/units/formulas, chart ranges/labels and consistent exported narrative. |
| LHT-10 Tutorial transfer | Follow an official office tutorial to build a specified multi-series chart and report; reopen native and PDF outputs. | Actual chart structure, numeric values, layout and export content; existence alone insufficient. |
| LHT-11 Reading packet | Select four of six official references using version/language/topic constraints; download and annotate the packet. | Correct inclusion decisions, document identities, source/page references and exact files. |
| LHT-12 Incident handbook | Combine three official troubleshooting topics with supplied incident facts into a decision handbook. | Correct branches and version claims; observed facts separated from advice and uncertainty. |
| LHT-13 Production calendar | Reconcile tasks, staff availability and dependencies, including a predeclared late amendment. | All tasks/constraints, current amendment in both artifacts, no stale entries and correct totals. |
| LHT-14 Screening dossier | Build a VLC playlist and schedule from six licensed clips and a programme brief; save and reopen. | Exact media identity/order, measured durations, break arithmetic and persisted playlist. |
| LHT-15 Interrupted recovery | Finish research, notes and packaging across one declared owned-window interruption. | Fresh-state recovery, no stale dispatch or duplicate/lost output, complete original goal. |
| LHT-16 Handoff capstone | Reconcile three official sources, a 12-row manifest and six assets into a sheet, narrative, viewing copy and archive. | Complete consistent manifest/asset mapping, formulas, exports, archive roundtrip and memory recall. |

Begin with Gedit/Files/Firefox cases. Calc/Writer, Evince and VLC require real
setup proof before their cases. An installed executable is not sufficient.
Extend successful pilots to hour-scale tasks and richer state; do not label a
shortened task equivalent to its inspiration.

## Scoring and improvement decisions

Freeze criteria before each run. Strict completion requires all mandatory
outcomes plus method/preservation compliance. Report partial outcome credit
separately, with a fixed denominator; retain setup failures and unfinished
cases. Record source/checkpoint/quantization and auxiliary revisions, prompts,
think/context/output budgets, observations/actions, wall time, first useful
action, tool/model rounds, unique dispatched inputs/replays, tokens, phase
latencies, resource use and human interventions. Unknown metrics remain unknown.

Attribute the earliest blocking failure: setup/interference, missing observation,
target selection, policy abstention, pointing, guarded dispatch, constraint/state
loss, domain reasoning, saving/export, verification or transport/budget. Mark
later challenges untested if the agent never reaches them. Preserve failures
when a later revision succeeds. Improve one evidenced failure at a time, then
repeat that acceptance criterion with fresh state before broadening.

Development cases and locked evaluation cases must be separate. For local
reliability, predeclare at least three independent resets per model/case and
interleave model order; this is our protocol, not a universal benchmark rule.
Successful recipes may be retrieved only in a separately labelled condition,
with new handles and fresh verification, never stale action-ID replay.

Claim parity only after an actual reference system runs the same tasks with
matched inputs, observation/action authority, budgets, resets and evaluator.
Predeclare a non-inferiority margin and evaluate paired uncertainty; sixteen
tasks provide limited statistical power. Without a reference run, report local
Qwen/Ornith results only. Original benchmark comparisons additionally require
the original pinned environment, assets, evaluator and track. Never reset the
user's desktop to construct that environment.

The bounded CPU follow-up isolated the malformed-delimiter admission to decoded
regex atoms emitted directly inside JSON wire-string grammar. The pinned C++
converter patch uses its existing warned canonical-string fallback for unsafe
atoms, retaining the complete phase schema and host validation. Native engine
controls accept the captured valid phase and reject the captured malformed phase
at body byte 3,581, where the unchanged PEG parser rejects it. Safe character
classes and the other phase branches remain covered. This is CPU decoder proof;
it does not claim a patched native deployment or another completed desktop task.
The fallback may admit empty or decoded-NUL strings at generation time; host
validation rejects those before any step is dispatched. Python/JavaScript schema
converters are unchanged and were not claimed fixed.

### Patched native deployment and LHN-071 checkpoint

The exact correction commit `b8dca607` passed the full source gate on both
hosts: 3,147 tests in 106.74 seconds locally and 166.05 seconds on Egg31.
The local validation receipt has SHA256
`d8ca02a76ccd61596e78a07df77d9f0c48678c4baebf6d56fd3f3de29ef4e7d6`.
A separate native build preserved loaded files and produced common-library
SHA256 `59d7f67d375f16c2308e70c11c8f7b4374f1ec26e53c00a85fc233fa4d0f830b`.
After one supported GPU1 handoff, the new Ornith trunk's process maps proved
that exact library loaded, with the same model pins and 65,536-token context.

One unassisted LHN-071 CSV incident-handbook attempt then failed after
72.59 seconds and three phases. All three actual compiler replies returned
HTTP 200 and valid JSON, establishing live acceptance of the delimiter repair.
Canonical host validation rejected phases one and three before any of their
inputs: proposed text lengths were 3,135 and 2,283 characters, exceeding the
unchanged 2,048-character limit. Phase two dispatched one real XTEST click
grounded by pinned Moondream 3.1. Its expected focus remained false across
four verification observations; the effect remains unknown. The point HTTP
request took 13.797 seconds including worker load, inference and shedding.
There were no reader, Kev or memory calls, and all three deliverables were
absent. Task cancellation/drain, exact three-window cleanup, input and
preexisting managed-window preservation, unchanged source, the new workers,
lease and recorder checks passed. No retry or second source fix followed.

The immutable failed bundle has SHA256
`dd5dd5bc8eb772b8a69ae1c3c0b14cf9cabd5655f999b4d0dc7ed8f772a35586`.
Compact after-test note:
`runtime-data/gpu1-acceptance/operator-audit/native-grammar-b8dca607-20261016T105049Z/operator-after-test-note.json`,
SHA256 `b3fbbc508600b8f8e986da88ce76ab243c76258f774ee18fb952c0d3c3388362`.
The append-only ledger now records 11 submitted attempts across four of
200 model/case cells and three task IDs, with zero full passes. Thus 196 cells
and 97 task IDs remain unattempted. Updated ledger SHA256:
`cd750b3a7e5b438981c7b7774a8eb11f9c2eaf63dae810748585e65d328406d4`.

Egg31 separately qualified the actual owned Flatpak Firefox setup: native
`about:blank` document evidence, registered scope, exact polite close and
preexisting-tree preservation passed in 33.28 seconds. The sealed receipt
SHA256 is `ff715d2f2426733ffcf908db02ac22b1f32da07bed000bf3258b010738859ef8`.
Its private ARM native build also passed the actual-library CPU grammar proof,
preserving five host-specific media/TTS source variants; runtime adoption
remained pending. Neither result is a remote model-task pass.

After the user's continuation instruction, Egg41 resumed through its normal
service API. One start reached healthy idle readiness in 110.50 seconds,
with Ornith's Tegra device handles and context verified, the prior harness
state restored and source/configuration preserved. Recovery note SHA256:
`359fbba6153438aca94cc6f947fea4567600dc89f6490066854dab40e32bf300`.
It still ran `476be1c` at this checkpoint; the decoder patch had not yet been
pushed or adopted there. Remote long-task acceptance remains pending.

### String bounds, LHN-092 and remote observation checkpoint

A separate offline review of LHN-071's retained pixels and native receipts
confirmed that the pointer hit Input and its parent selection changed from
empty to `Input`. The table cell's focus predicate remained false. This proves
selection, without proving that the directory opened or changing the failed
task grade. The derived audit SHA256 is
`33db8f385e7bb67373c50f4a47422b5d9a62fc0cc692357393179c554b21fcfb`.

Commit `497c3ae` preserves explicit string bounds through the native regex
fallback and composes large finite repetitions without the pinned parser's
2,000-repeat limitation. The full source gate passed 3,149 tests locally and
on Egg31. The local receipt SHA256 is
`fc33ca74aff3d63dd19c9aaf88b4d3c0cc51c5b9f886e75d341f4cc0132f648b`.
An independently built library, SHA256
`372a44f8b1e83ddb539d274a662ffba9a487ca6504459e45de517b422a75d49f`,
passed actual-library CPU grammar controls and one supported GPU1 Ornith
adoption, with process maps, residency, pins and context verified.

The original whole LHN-092 duplicate-name recovery task then failed in
41.53 seconds. Phase one was schema-valid and exercised exactly 2,048
characters of proposed text. It dispatched one verified window switch, then
abstained because its focus target was a window advertising neither focus
nor typing. No text was entered. Phase two returned an unfinished JSON string;
the actual new-library CPU grammar accepted it as a viable, incomplete prefix.
The native log records exactly 4,096 generated tokens, matching the adapter's
clamp despite the caller requesting 8,192. An output-cap stop is an inference:
the native finish reason was not retained. The JSON parse exception terminated
the task after two phases. No source acquisition, physical pointer/typing,
reader, Kev or memory calls occurred, and all three deliverables were absent.

An independent review reproduced the immutable bundle SHA256
`d62163a9eb5188c57e0b74dcb2bf77fac61f9834c415c2b686c132b65b8d609a`.
Nine mandatory criteria failed; the full preservation assertion remained
unproven, with positive bounded checks for unchanged inputs, 16 preexisting
managed clients and three politely closed owned windows. Final evidence:
`runtime-data/gpu1-acceptance/operator-audit/native-grammar-497c3ae-20261016T114236Z/final-index.json`.
The ledger now contains 12 attempts across five of 200 local model/case cells
and four task IDs, with zero full passes: 195 cells and 96 task IDs remain
unattempted. Ledger SHA256:
`d3e37b454a517f0f22f0ad0a8b1293b8cff917f40d312d7a9303e0f4283e4815`.

Egg41 automatically adopted `76eaa64`, passed its 3,147-test source gate and
loaded the corrected ARM common library, SHA256
`7545df822fad9fc4bac0ee4375038f23134ab514eab768033f1d868a68fb55b8`.
Its original LHN-041 setup still failed before any model request. One private
trace proved an explicit null-child guard caused the native observation error;
no underlying GI read exception was recorded. Commit `93c95e4` changes that
guard to incomplete child coverage, preserving current siblings and identity
history without old input handles. Its validated tree passed 3,159 tests.

One corrected private-source observation on Egg41 returned a truncated current
window instead of aborting. It exposed only that window: no child control graph
or DocumentWeb/DocURL. Browser task readiness therefore remains blocked, and
no canonical model task followed. An earlier private launcher failed its
access-time assertion before observation; that instrumentation failure was
retained and corrected separately. The successful private acceptance receipt
SHA256 is `9d349ae1aa27065ace86689395691ed0be7c1af8a9b50062f291a1b5b15f7e34`.
Production source and services remained unchanged during both private tests.
Egg31's corresponding string-bounds library and source gate are prepared but
not adopted. Remote setup/build results do not add local model/case coverage.

### Bounded compiler recovery and second LHN-092 attempt

Commit `1595aa3` converts only JSON decoding failures from successful compiler
replies into a private typed error. The existing three-consecutive-failure
budget consumes it without input, then obtains a fresh native observation for
replanning. Cancellation/deadlines take priority; transport, HTTP, envelope and
unexpected callback failures remain fatal. No JSON repair, model replay,
schema relaxation or token-budget change was introduced. The full source gate
passed 3,182 tests; receipt SHA256:
`c0e59322b01efdebb1ec46c10a559a3b2e6159f07e8267cb68c311696d429685`.

One CPU frontend handoff adopted the exact tested planner imports while
preserving the Ornith GPU workers, library `372a44f8`, context and scoped lease.
A fresh original LHN-092 attempt started at 12:06:59 UTC and failed after
67.15 native-task seconds and four phases. All four actual compiler replies
were canonical-valid JSON: recovery was not exercised live. One accepted
Moondream reading described Files, without official-source acquisition credit.
The reader and point HTTP requests took 14.932 and 4.080 seconds respectively.
Two XTEST clicks were dispatched with unresolved expected effects. They targeted
the Workbench window and a Recent Files item while expecting keyboard focus on
the window root. The final pointer target referenced an older test's Recent
entry and was rejected before input as occluded. These are distinct from the
earlier Input-selection evidence. No typing, deliverables or memory followed.

The immutable failed bundle SHA256 is
`b6693980a674e6fca9debc60fef0b6135c5dec1c79815510410030c4ae93023e`.
Owned cleanup, unchanged inputs/source, preexisting managed-window checks and
GPU-worker/lease preservation passed. Final evidence:
`runtime-data/gpu1-acceptance/operator-audit/native-phase-compile159-20261016T121434Z/final-index.json`.
There are now 13 local attempts across the same five model/case cells and four
task IDs, with zero complete passes. Ledger SHA256:
`2d596cb694ae66969c730987fe2589347c60f73644bf2386648add306654d393`.

Egg41 automatically adopted `ae767d3`, passed 3,159 ARM tests and proved the new
mapped common library and healthy Ornith Tegra residency/context. A separate
owned blank-window experiment retained two initial one-node observations,
one normal frame-attribute read and three subsequent 232-node observations
with 41 eligible controls and one DocumentWeb. Identity/scope remained stable;
DocURL stayed unavailable. This supports a missing native accessibility
initialization step, with the one-window temporal/causal limitation retained.
It does not establish production browser readiness or a model-task pass.
Experiment bundle SHA256:
`48f08658bd763afe4dc0c70519699b69c52d04debf055b0920def554570f2d7e`.

Egg31 installed `ae767d3` and library `0504a2f4` through one supported cold
handoff. Its journal logged portal readiness after 86.59 seconds. The private
adoption monitor incorrectly required the on-demand pointing graph to be
resident while idle and failed to retain its loop samples; it stopped the
ready runtime at its 180-second proof deadline. This is an operator proof
defect, not evidence of model initialization failure. The private guard was
corrected with explicit cold/on-demand versus resident-worker checks and
bounded per-predicate samples. Recovery and remote model-task acceptance
remain separate, pending checkpoints.

### Focus contract and first original LHN-041 runs

Commit `c459b74` explains keyboard focus and foreground activation in both
canonical predicate schemas, which feed the model-visible phase packet and
decoder schema. It changes descriptions only: predicates, capabilities,
authority, token budgets and failure limits are unchanged. The full source
gate passed 3,194 tests. Actual-library controls reproduced identical grammar
bytes with and without the descriptions; that receipt SHA256 is
`19e852263e0700aacf7779cc35828b8b988bea262aedd5f6edc5653479a98473`.

One local CPU frontend handoff adopted this source while preserving the exact
eight underlying GPU-stack process starts, the scoped GPU1 lease and common
library `372a44f8`. The first frontend launch failed its doctor before spawning
because the private checkout lacked the expected binary path. An ignored link
to the already verified immutable binaries corrected that setup omission;
neither source nor GPU workers changed.

The first local original LHN-041 conflicting-revisions task failed after
53.87 runner seconds and three phases. All three actual compiler requests
contained the new descriptions in identical packet and decoder schemas.
Two window switches used `active_window` correctly. The first phase still
requested unsupported focus on an editor window root; the second misused an
already-active switch with unrelated visibility predicates. The third reached
an actual editable browser control and dispatched focus, physical typing of
the current owned Workbench URI, and Enter. A subsequent nonexistent control
target caused abstention. Five unique inputs were dispatched, but no task
brief, revisions or official sources were read, and no deliverables or memory
were produced. Proposed reconciliation prose was not evidence of reconciliation
and was never typed. Annotation adoption passed; planner adherence was mixed
and the whole task failed.

The immutable bundle SHA256 is
`3259b1c3717cc57cce11841d37f452fd981c38182934d544cd6e863f94d21a02`.
All 20 inputs, all 139 preexisting X11 tree rows, source and exact GPU workers
were preserved; three owned windows closed and the stack drained. The runner's
own worker snapshot was unavailable, so its `workers_preserved` field remains
null; the separate exact-process proof is positive. Final evidence:
`runtime-data/gpu1-acceptance/operator-audit/native-focus-c459-20261016T124944Z/final-index.json`.
There are now 14 local attempts across six of 200 model/case cells and five
task IDs, with zero complete passes: 194 cells and 95 task IDs remain
unattempted. Ledger SHA256:
`a7d1f3542c62401438be5efa689b0f2d17bcfb97a53de1d7138ad302a223996c`.

Egg41 automatically adopted `f808fb2`, passing 3,182 ARM tests and proving
Ornith Tegra residency, the mapped bounds-corrected library and 65,536-token
context. Its updater replaced the full core, even for these Python changes.
The first original whole LHN-041 model run then failed after 343.78 seconds
and three phases. The first malformed JSON reply produced typed nondispatch
feedback, followed by a fresh observation and a valid second-phase workflow:
bounded compiler recovery was exercised live. Reply length, finish metadata
and the next raw compiler packet were not retained, so no output-cap or exact
packet-delivery claim follows. A verified window switch and physical XTEST
click occurred; a subsequent focus call on an editable control left its focus
false while another child remained focused. The model also drifted toward
Qwen documentation instead of the supplied GNOME revision-selection task.
No keyboard input, acquired sources, artifacts, reader, Kev or memory calls
followed. Owned cleanup and preservation passed, including all 61 preexisting
X11 rows. Its immutable bundle SHA256 is
`2eeef23f9473c72c604265628ceb075bfa060a9eb59908105d453f407b2b68a3`.
This Tegra/Moondream-2/no-Kev result is a separate remote cohort, not one of
the 200 local A100 results.

Egg31 recovered its healthy runtime in 86.02 seconds, then passed 3,182 ARM
source tests and adopted `f808fb2` with one supported Python reload. The daemon,
trunk, native libraries, GPU workers and existing X11 tree were unchanged.
Its first LHN-041 setup selected an unavailable editor and stopped before a
model request. A corrected setup using the installed Gedit observed all three
owned applications, but the Flatpak Firefox ownership helper subsequently
returned EOF during registration. Gedit and Files closed; Firefox cleanup was
withheld because its proof channel had failed. That case has no model-task
coverage and cleanup remains incomplete. A separate private-Xvfb CPU
discriminator reproduced fatal default-Xlib handling of an invalid window,
but the earlier helper's stderr and exit status were not retained: its actual
historical cause remains unknown. No existing browser window was killed.

### Programme repair and X11 ownership diagnostics

Egg41 subsequently adopted `1ba1850` automatically. Its 3,194-test ARM gate
passed, new workers proved Tegra residency and the existing 65,536-token
context, and the live read-only tool catalog exposed both exact focus/activation
annotations for both native tool aliases. Configuration, native library and
all 62 preexisting X11 rows were preserved. Adoption receipt SHA256:
`c37e41f57816b086a1298fede767fab810e79134867004ebb42fdbe5481acc95`.

The unattempted LHN-011 museum-visit task could not freeze fresh primary source
bytes: the museum's public endpoints returned HTTP 403. Cached search content
was not substituted for fresh evidence. No model request or canonical attempt
was credited. LHN-021 programme repair was selected instead, with unchanged
inputs and goal, five blank native apps and eight current official TDF/GNOME
source captures kept outside the solver's inputs.

The original Ornith LHN-021 run failed after 47.875 seconds and three valid
compiler replies. All five owned apps passed setup, but no input was dispatched.
One phase had no eligible control; two pointing requests returned points outside
the bound target. Offline inspection verified the exact transmitted PNG/crop
and production floor transform. Both points landed near Recent Files, outside
the requested breadcrumb button; its full basename was not painted. This
supports correct pre-input rejection, not a global coordinate-transform defect.
The two HTTP 200 grounding requests took 3.924 and 4.052 seconds; neither
produced an accepted point receipt. No source, keyboard, artifact or memory
criterion passed. All four inputs, the exact GPU workers and lease, and all
139 preexisting X11 rows were preserved; all five owned windows closed.

The immutable bundle SHA256 is
`a3cb89169358359d1a63c280ed9ee54d109d401f0bcf18b5bb96fc5ac5d9a1b4`.
Independent review passed preservation and failed the other eight criteria;
review SHA256:
`37a0df2724342c50840d2643c754f9229e58f796249129d0208c07e736a4c0ff`.
The local ledger contains 15 attempts across seven of 200 model/case cells
and six task IDs, with zero full passes. Its SHA256 is
`50433ea05827ffab336fe99327f07a99576c9a57f5f5d388ac05704263027257`.

Commit `9107b3b` converts protocol errors on the ownership helper's own X11
connection into typed failures instead of default-Xlib process termination.
Incomplete tree/subscription/event evidence permanently invalidates that
connection. Other connections retain their prior/default handler. Nine real
private-Xvfb controls passed, including destroyed resources, asynchronous
selection errors, foreign-handler delegation and a real 4,096-window boundary;
the full source gate passed 3,203 tests. Code/test bytes match the frozen gate;
the explanatory documentation was added afterward and was not represented as
part of that gate's frozen tree. Control receipt SHA256:
`553009aa4f743022cfdb5be0e1e4cf9e5f3689b23fbf6049807888fc2168da64`.
This establishes the reproduced protocol-error behavior, not the unknown cause
of Egg31's historical EOF or a live browser/task success. Its blank-page RPC
verification requirement is used during setup; cleanup already permits a
navigated page. No unused generic-verifier change was made.

### Matched Qwen run and bounded prefix recovery

One controlled GPU1 handoff replaced Ornith with the installed Qwen audio-bridge
tag, preserving context 65,536, common library `372a44f8`, policy/vision pins and
the fault recorder. All eight old processes and CUDA allocations disappeared,
Kev stopped before the old supervisor, and the old lease released before the
new scoped lease. The new Qwen parent was ready in 102 seconds. A separate
`c459b74` CPU frontend bound the actual new parent, model and module paths.

The first Qwen LHN-021 attempt failed after 82.036 seconds and three valid
compiler replies. It correctly targeted Input, but all three Moondream requests
returned HTTP 500 `FileNotFoundError` before input. The private checkout lacked
its default weights/tokenizer paths; installed pinned files existed in the main
checkout. This is an operator asset-path configuration failure, not evidence
of grounding accuracy, planner inability or a hardware fault. Independent
review failed eight criteria and passed preservation. Immutable bundle SHA256:
`8e69148c3edbb8f5b2d8b4afdeefcaf01b379529abd28904f0c657d804cc0401`.

Exactly two absent ignored asset links corrected the configuration. Production
artifact validation accepted the real checkpoint/tokenizer/CUDA-runtime hashes
and inodes without loading a model or initializing CUDA. Source, workers and
lease stayed unchanged. One original Qwen LHN-021 retry then exercised actual
Moondream 3.1 reading and pointing successfully: two guarded physical
double-clicks opened Input and `task-brief.txt`, with fresh native effects.
The task still failed after 220.673 seconds and eight phases. Later selection
requested 2,048 characters from a 1,261-character buffer and was rejected;
repeated focus received no-progress feedback, and a subsequent selection
remained unverified. No text entry, output files or memory result followed.
The four unique dispatched actions include two physical double-clicks, one
semantic focus and one selection; they are not four physical clicks.
Immutable retry bundle SHA256:
`168584d638b78e283c6a29db72db95753c21ccd524e54f9fcc8de1b78b57a87f`.
Both Qwen attempts preserved all four inputs and 139 preexisting X11 rows,
closed all five owned windows and drained the stack. The corrected retry also
preserved the new eight-worker graph, lease and installed artifact inodes.
The local ledger now contains 17 attempts across eight of 200 model/case cells
and six task IDs, with zero full passes: 192 cells and 94 task IDs remain
unattempted. Ledger SHA256:
`2029981b87721d46f2b7035d37bff049e0a3fd48e7828957aae065086dbeb250`.

Commit `3855b4b` grants another bounded replan when a failed workflow has a
verified prefix containing a new dispatched, producer-bound applied effect.
The workflow stays failed and whole-task verification stays false. Novelty
uses semantic action/effect transitions, not opaque IDs or expectation wording;
stable window cycles cannot repeatedly buy recovery. Three unsuccessful phases
without a new eligible transition, cancellation, deadlines and the 40-phase
ceiling still stop execution. No token, schema or input-authority limit changed.
The combined full source gate passed 3,242 tests. POST receipt SHA256:
`319904f0d7de594fb66c06066ed3368aaef23e066b6b12641113c8b20d0e9040`.
The exact saved LHN-041 results reach a synthetic fourth planning phase, while
the original zero-effect Ornith LHN-021 results still stop at three. Those are
CPU replay controls, not live task completion. Live adoption remains a separate
checkpoint.

### Prefix-recovery adoption and tutorial-transfer setup

The local `ef6403f` Python frontend passed its bounded readiness window while
preserving the Qwen GPU workers, native library, lease and model pins. Its first
original LHN-059 tutorial-transfer run failed after 64.884 task seconds. One
Moondream-grounded physical double-click opened Input with all three declared
filenames verified. A second compiler phase reached grounding, then the stream
disconnected with `foreground_identity_outside_owned_scope`; no completed
dispatch/effect receipt or authoritative final was received for that phase.
That second input's outcome is unknown. No required workbook, Writer document,
PDF, acquired official source or memory result existed. The new prefix-recovery
condition was unexercised: phase one completed, and phase two had no completed
workflow result.

Post-terminal metadata identified a preexisting GNOME Text Editor window as
foreground. The frozen five-app fixture had no text editor, and the host's
actual `text/plain` handler was GNOME Text Editor. The earlier LHN-021 fixture
included an owned window in that application and opened its brief successfully.
These observations support a separate host-setup hypothesis; the historical
guard record did not retain the foreign identity and does not prove that actor.
All three inputs, all 139 preexisting X11 rows, the exact GPU workers and lease
were preserved, and all five owned windows closed. Foreign editor contents were
not inspected, so full unrelated-content preservation remains **unknown**.
Immutable bundle SHA256:
`d99e65b87e8d931127de8f02ce2fcfec6299e5cdfc3e6e8c4bdb44db89f46a30`.
The local ledger contains 18 attempts across nine of 200 model/case cells and
seven task IDs, with zero complete passes. Ledger SHA256:
`6ced0706aa7e867ea05c51cebbd54e2889783c56aef2a063ef402106df9fd993`.

Egg41 automatically adopted `ef6403f` after its actual ARM gate reported
3,233 passed and nine skipped. Its host lacked Xvfb; the individual skip reasons
were not emitted, so no claim follows that the nine private-Xvfb controls ran
there. The updater performed a full core restart. Fresh source/catalog hashes,
the mapped native library, actual Ornith Tegra residency, 65,536-token context,
idle health, config and all 62 preexisting X11 rows passed the adoption review.
Review SHA256:
`ba3d9745fe34c8c40826b2efd421022b2d839b9d1b1de75e40b658dc3a9e41ba`.

Egg31 separately passed all 3,242 ARM tests before adopting the same source.
The first private doctor invocation mistakenly checked its default Qwen tag,
which was not installed on this Ornith host; that attempt performed no reload
and remains retained. Explicit configured-Ornith doctor checks passed before
the single supported Python reload. Adapter and portal changed, while the
daemon, trunk, pointing/TTS controls, tunnel, native libraries, config and all
71 preexisting X11 rows were preserved. The continuation took 9.858 seconds;
adoption receipt SHA256:
`e27834ae30d69b3271323101d1fa4bed8858f33e906fe513805bea217aa35e14`.
These are deployment results, not desktop-task completion or live acceptance
of the prefix-recovery condition. Both remote configurations remain separate
Tegra/Moondream-2/no-Kev cohorts.

Both subsequent original LHN-021 remote setups stopped before model submission.
Egg41's owned Calc window exposed no usable native scope or nodes, after a
partial browser observation. Its 8.166-second setup failure has no model,
phase or task-criterion coverage. Canonical cleanup initially reported false;
a separate later complete-tree/process proof found both task windows absent
without replaying close. All 61 rows present after harness quiescence remained
unchanged, and the final 62-row tree matched the pre-quiescence baseline.
The original cleanup field remains false. Canonical bundle SHA256:
`157acf9eda0d35c74a2053cf73b6845116f52a246efbe7a339f420ebaf625a49`.

Egg31's 1.591-second setup received a valid negative helper `create` reply after
the ordinary Flatpak launcher exited successfully. It created no ownership
ledger; `close` then correctly returned `flatpak_launch_ledger_missing`.
This was not the earlier transport EOF. The generic error code does not identify
the failed internal substage. All 71 original X11 rows were preserved, but four
new mapped resources remained and no affirmative cleanup authority existed.
No second close or setup was attempted. Its canonical bundle SHA256 is
`ddd3503246fcd51773ce41b1bbedfbca202c5ca153245b937a3f69a7ee6545d3`.
Neither setup-only result enters the local 200-cell model denominator.

### Owned text-handler qualification and typed helper errors

Commit `673c09a` adds a default-off, explicit host setup option to own one blank
selected text editor when the frozen fixture lacks an editor slot. The selected
native editor must match the actual `text/plain` handler; mismatches stop before
app launch. Existing slots are not duplicated, the 16-window ceiling remains,
and only the fresh proven XID joins scope and cleanup. Goal, inputs, rubric,
task signature, MIME defaults and model instructions are unchanged. The 159
focused checks and full 3,265-test gate passed. This is a changed host setup
condition, not a matched replay of the original five-app fixture.

One original-goal Qwen LHN-059 qualification used the new runner with the
existing `ef6403f` frontend and `c459b74` GPU supervisor; neither was restarted.
Moondream grounding and physical double-clicks opened Input and the task brief.
Fresh observation bound the brief to the supplemental owned editor XID
`81789121`, excluding preexisting `81788932` despite their shared process.
The guard stayed clear throughout. This passes the host brief-opening criterion.
The whole task failed after 252.224 task seconds and nine phases: selection
`0..1` was dispatched but never verified, and two cached unresolved retries
reached the three-failure stop. Eight unique dispatches include semantic and
unchanged operations; they are not eight physical clicks. There was no authored
keyboard input, required output, acquired official source or memory result.

The real brief-opening phase had a verified new applied prefix but a failed
terminal predicate. Replaying all completed receipts in original order with
the actual novelty history verifies prefix-recovery eligibility and a reset
there. The live run stopped at phase nine; the old counter arithmetic on this
same saved trajectory would stop at eight. That arithmetic is not a separately
executed old-controller comparison. All three inputs, all 139 preexisting X11
rows, eight worker starts and the lease were preserved, all six owned windows
closed, and downstream work drained. Unrelated document contents were not read;
the preservation evidence is scoped input, guard, identity and metadata evidence.
Immutable bundle SHA256:
`23d249a6bfb8a210c539c6e05e551b2425a305d230746a6dbf1520af2cbb6d19`.
Independent review scored eight failed criteria and preservation passed, based
on unchanged inputs, exact before/after trees and managed identities, six owned
close receipts and the immutable scope excluding the older editor window.
That preservation verdict retains the uninspected-content limitation above.
There are 19 local attempts across nine model/case cells and seven task IDs,
with zero whole-task passes. The newest attempt has the explicit additional
host-condition label. Ledger SHA256:
`08970e223a0360c4d628d65250432eaccd03c65833ae6338a67dfe44923ef4c8`.

Commit `b25476d` fixes a separately reproduced helper-protocol defect: its
validator rejected digits in its own `flatpak_x11_*` codes. Producer and client
now preserve bounded ASCII identifier codes, while malformed codes are still
redacted/rejected and never confer ownership. Actual CPU pipes reproduced three
failures before the patch. Afterward 237 focused controls and the combined
3,291-test source gate passed with 378 unchanged fingerprints. POST receipt
SHA256:
`69d690e9b6c7effe27a3c6e7aaf0df109a119b29c30e0d583e8a7a3b70d3e694`.
The committed executable/test tree matches the validated isolated tree; this
evaluation narrative was updated afterward. This does not identify Egg31's
historical underlying error or restore a missing ownership ledger.

### Ornith LHN-001 and loaded decision-head audit

After adopting `7ddf3ca`, Egg41 ran the original LHN-001 Python migration
dossier task with three owned applications and unchanged inputs and rubric.
The live Ornith request completed four phases before blocking. Switching to
the editor verified, but the model requested unsupported window-root focus;
a later advertised New Tab focus dispatched without a verified effect, then
two further window-root focus requests found no eligible control. The whole
task failed: no required output, source acquisition or memory result. Review
scored nine failed criteria and preservation passed. The original browser
cleanup result remains false; a separate saved later tree proves the three
owned visible windows absent and all 61 prior rows unchanged. Three new
unmapped resources remain unclaimed. Canonical bundle SHA256:
`fa1b9397c645b76c2b2354079cb7509cc316e43a79b200c2814b36220c79e149`.

Egg31's original LHN-001 setup stopped before any model call. The ordinary
Flatpak Chromium window was registered using reported PID `2` and start `0`,
which were not a positively verified caller-owned host identity. Observation
then failed generically; that error does not identify a provider-read cause.
A later complete tree proved the newly created visible window absent and all
75 prior rows unchanged. No task criterion was exercised. Canonical bundle
SHA256:
`c820ac0ec26caa5c5bf31f6ce55dc2eb6cda5326bb08d99d92e97704ee2245d0`.
Both remote results remain separate Tegra/Moondream-2/no-Kev cohorts.

A separate, frozen 24-case local audit completed 24 actual Qwen-27B calls and
24 actual Kev-4B head calls on the existing GPU 1 workers in 265.081 seconds.
Every response passed its schema/protocol check; worker identities, lease,
source and final idle state were preserved. Inputs were explicitly historical
and conferred no current desktop input authority. Qwen returned 14 blocked,
seven complete and three visual-inspection phases, with no UI phase. This
authority condition confounds capability-planning evaluation; it supplies no
positive current-desktop action-quality result.

Kev's raw choices matched 13 of 18 nonzero-candidate target labels and all six
zero-candidate safe labels. Every score remained below the unchanged 0.95
threshold, yielding 24 abstentions. These hypothetical target labels are not
GUI effects or completion grades, and unique/zero-candidate probes do not prove
production policy routing. No threshold, prompt or label was adjusted.
The actual language adapter retained its 4,096-token output cap despite the
request's 8,192 option; upstream finish reason and usage were not exposed and
remain unknown. There were no GUI inputs, executor calls, visual model calls,
memory operations or whole-suite credits. Immutable audit bundle SHA256:
`8bf4cfa32139e8e812c233dea3466b8ee58106a6127f9ed3a287f019500ba565`.

The next source checkpoint adds optional native text counts and captured-prefix
coverage, preserving unknown values separately from observed zero. It keeps
legacy native revisions and idempotency unchanged. Review reproduced two
prepublication regressions: metadata could displace an existing planner-visible
node or make a repeated zero-effect phase appear new. Legacy controls now receive
the existing projection budget first; optional metadata uses spare space.
Repetition fingerprints use a metadata-free state copy and bound full-state
hashes, while raw evidence hashes and other authority fields remain intact.
The actual endpoint source fixtures now stop those loops rather than accepting
an unverified completion claim. They are source tests, not live model runs.

The same checkpoint requires positive host process ownership before ordinary
window registration and supplies an explicit `flatpak-chromium` setup option.
A namespace-local WM PID needs a complete authenticated host binding before
publication; foreground and scope checks share that resolver. Ordinary valid
host processes need no namespace/XRes query. Existing browser launchers and
profiles are preserved. Installed Chromium latency and live task behavior are
not established by the CPU proof fixtures.

The composed focused gate passed 776 tests. Its first full gate failed 27 old
fixture cases because two fixtures lacked the new physical-host identity
boundary; production remained unchanged. Explicit fake-process identities and
an asserted launch-boundary double repaired those fixtures without bypassing
scope or namespace checks. Both affected files then passed 59 tests, and the
second full gate passed all 3,432 tests in 103.96 seconds. All 381 source/test
fingerprints and five reviewed private helpers were stable during each gate.
The passing receipt SHA256 is
`24aa74cb7e137da2af85168298b2a4e3b256cf49e4a2ebd0e13187ad94d2a040`.
This evaluation narrative was amended after the source gate. Deployment and
whole-task live qualification remain separate, pending results.

### Live qualification of the text and browser-owner checkpoint

Published revision `8c3926c` passed all 3,432 tests on Egg31 and 3,423 tests with
nine skips on Egg41. Egg41's automatic updater adopted the revision within its
600-second window: fresh Ornith workers, actual Tegra residency, native library,
65,536-token context, health and idle checks passed. All 62 desktop metadata rows
and configuration were preserved. Individual skip reasons were not exported;
auxiliary-head warmth was not established. Adoption receipt SHA256:
`5fcbafa95d80099e94a12c6851784ae1a688698a5725993c292cdef39e4c235f`.

The local CPU frontend adopted the revision while preserving the eight model
workers, GPU 1 lease, mapped native library and 139-row desktop baseline.
A newly added private import observer initially rejected an already wrapped
method; that failed frontend startup is retained. An unwrapped-method proof
passed with the real metric hooks, then recovery startup passed its preservation
window. This was instrumentation repair, with no production change or model
request. Egg31's single CPU reload also passed. Its earlier pre-adoption hold
retains an unexplained unmapped resource disappearance; the successful reload
preserves a declared fresh 75-row post-quiescence baseline, not the older 76.

One full original-goal Qwen LHN-059 run used the same explicit owned-editor host
condition. Two real Moondream-3.1/XTEST double-clicks opened Input and the brief
in owned editor XID `81789161`, excluding older `81788932`. The whole task failed
after five phases. Native selection `0..1000` remained unresolved, then a new
`0..1077` attempt also remained unresolved; only the final phase replayed the
cached second attempt. Five unique dispatches comprise those two double-clicks,
one native focus and two native selections. There was no physical keyboard
input, required output or memory result.

The raw brief node positively reported 1,077 characters, complete captured
coverage and zero selections. The current planner view had 11,998 of 12,000
compact characters and omitted that optional metadata. Crucially, both later
compiler packets retained the typed facts twice in action feedback. The changed
full-range request used the observed length; a blanket feedback-loss claim is
unsupported. All three inputs, model workers and lease were preserved; all six
owned windows closed and work drained. One prior X resource disappeared and nine
metadata rows changed, with actor and unrelated contents unknown. Independent
grading is eight failed criteria and C08 unknown. Canonical bundle SHA256:
`28c8c41c282f7a6f73c97d632a5623ba075bf12e330db97df56ddd80432c8448`;
peer review SHA256:
`b2b69ecf976ed29a7d9c16abe50d4e7a1c926b8960e0ab9dfb1371a62dd798ba`.
The reconciled local campaign has 20 unique attempts across nine model/case cells
and seven task IDs, with zero whole-task passes. Remote setup results remain
outside the 200-cell local denominator.

Egg31's original LHN-001 qualification used the explicit owned Flatpak Chromium
setup. The real ordinary-profile launcher started, then the helper returned
`flatpak_x11_protocol_error`; no portal or model submission occurred. Cleanup
returned `flatpak_launch_ledger_missing`. All 75 preexisting rows remained, but
five new resources lack an affirmative ownership ledger and remain untouched.
Solver criteria are unexercised and preservation remains unknown. The exact
failed X request, resource and stage were not retained, so no causal diagnosis
follows from the stable code. Canonical bundle SHA256:
`1c93554681bc7516a8edd774d91aeabd3d77280bac28255dacf312efcc95dbfc`;
independent peer review SHA256:
`702f04cab4c01d1649ce406ee82339250b4c2e8ee0b2527a207863bc980b11e6`.

### Live full-text selection and a repeated-window cache fault

Revision `d8a4bf5` passed 3,531 source tests and adopted locally without changing
the eight GPU workers or GPU 1 lease. Egg31 passed 3,531 tests and one CPU-only
reload. Egg41 automatically adopted it after 3,519 passes and twelve skips;
individual skip reasons were not logged. Both remote runtimes were healthy and
idle. The proposed read-only X11 discriminator was not applicable: all five
specified resources were absent. Their historical disappearance remains
unattributed, and no invalid query or new browser was used to manufacture an
error.

One original full Qwen LHN-059 qualification failed before reaching selection.
Three actual Moondream calls returned points, but all attempted double-clicks
were rejected before input with `native_observation_unavailable`. The precise
provider read was redacted and remains unknown. A separate bounded read-only
Files diagnostic did not reproduce it. The task retained zero dispatched
inputs, five successful owned closes and one negative close, with four new
editor/dialog resources left untouched. Independent grading is eight failures
and C08 unknown. Its sealed bundle SHA256 is
`79b3d08c94590eb3bd869ef954cba093ff9c5ceb44463d17210745aafb2755fa`.

The next, different original task, Qwen LHN-072, ran for 619.06 seconds and
eighteen phases. It verified physical double-clicks opening Input and the brief.
After unresolved partial selection `0..1200`, the model authored the exact
complete range `0..1412`. One guarded Ctrl+A was chosen before any native setter,
and a fresh native observation verified that exact range. The runtime did not
expand the earlier partial request or retry its uncertain dispatch.

The same trajectory navigated ordinary Firefox to the official Git-status page
through focus, physical URL typing and Return; the committed DocURL was
verified. One source-bound Moondream reading was accepted as captured-snapshot
knowledge, with limited section coverage and no current-input authority. Across
the task there were thirteen unique dispatch receipts: two physical
double-clicks, four keyboard inputs and seven native pattern/activation attempts.
Two point calls and one reading call exercised Moondream; Kev and memory were
unexercised. All three inputs and three owned closes passed their checks. No
required output existed, and the whole task failed. Its sealed bundle SHA256 is
`83ccf97b02b19ccb03562f6f1235b8048f2f6a5158cc5b8ea643f78c25baaf6f`.

The terminal failure exposes a separate cache problem: after intervening
verified window changes, newly planned editor switches reproduced an earlier
state/step binding and returned its old verified receipt. They sent no new input
while fresh terminal observations remained in Firefox. A new source cycle must
distinguish that new intent from retries of the original uncertain input. The
live full-range selection milestone does not establish general task completion
or repair this separate fault. Independent grading records six failed literal
criteria, three passed occurrence/scope criteria (C02, C07 and C08), and C04
source-section coverage unknown. The stronger whole-goal authored-output and
physical Save/delivery requirements remain unmet. The 26-member bundle was
unchanged before and after review. The campaign now has 22 distinct attempts,
ten model/case cells, eight task IDs and zero whole-task passes. The following
source checkpoint separates first-action grounding from cache renewal after a
later verified mutation; whole-task live qualification of those corrections
remains pending.

## Three-host qualification of `3f2d845`

The local CPU frontend adopted this checkpoint while preserving the eight GPU
workers, GPU-1 lease and mapped native library. Egg41 adopted it automatically;
Egg31 passed its isolated ARM gate and one CPU-only reload. Source gates passed
3,584 checks locally and on Egg31; Egg41 reported 3,572 passes and twelve skips
without individual skip reasons. Adoption does not establish task success.

The next different local task, Qwen LHN-042, retained its original twenty inputs
and ten criteria. Seventeen unique dispatched inputs comprised two pointer,
nine keyboard and six native attempts. Two Moondream point calls and two
accepted snapshot readings were exercised; Kev and memory were not. The model
selected all six correct filenames, but four claimed embedded revisions were
wrong, and its submitted actions did not read the authority ledger or document
versions before authorship. No required output, official-source navigation,
archive, reopen or physical Save/delivery completed.

Five multiline typing attempts remained unresolved. Two requested 370
characters including six line feeds; the observed value contained exactly the
other 364 characters. A later cached selection receipt was consumed even though
the fresh input prestate had no selection: a new 549-character write appended
543 characters to the existing 364, producing 907. These are separate input and
workflow-consumer faults. The operator requested supported cancellation at
18:36:30 UTC on 2026-10-16 after reproducing systematic line-feed loss. A late
compiler reply was retained without subsequent completed-phase/input evidence;
this was not natural termination at the consecutive-failure limit.

All twenty inputs and 138 prior window metadata rows were preserved. The owned
editor's polite close was negative and four new editor/dialog resources remained
at the canonical checkpoint. Independent grading records eight failed criteria,
literal keyboard/mouse occurrence C07 passed, and C08 unknown. The unchanged
43-member bundle is
`7f48b578683e1cd18d391f7b0c76b95d357edd0c3d925ae52beca956c3986891`.
The local campaign is 23 sealed attempts, eleven model/case cells, nine task IDs
and zero whole passes; subsequent cleanup must not rewrite this negative seal.

Both Eggs received the unchanged original Ornith LHN-072 task. Egg31 stopped
during Flatpak setup before a model submission, retaining BadWindow 3,
request 2, resource 67108865 at `events:post_sync` and a poisoned connection.
No launch ledger or cleanup authority was established. Egg41 ran four phases
with three unique input attempts: a verified switch, an unresolved switch and
unresolved Ctrl+L. An older switch receipt was replayed without input; the fourth
phase returned invalid JSON. No source acquisition, output or memory completed.
Its three owned windows closed, but three new unmanaged resources had unknown
owners. Independent grading is eight failures, literal C02 passed and C08
unknown. These Tegra/Moondream-2/no-Kev cohorts remain separate from local cells.

Actual private-Xvfb experiments reproduced both the lost line feeds and a
CreateNotify/destroy-before-drain subscription fault. The multiline input
candidate's literal physical sequence passed read-back controls, including the
exact 370-character fixture. The cached-prefix candidate blocks the saved
364-to-907 append before input while preserving the stored historical outcome.
These are source regression proofs, not new loaded-model task passes. A separate
XRes lifetime experiment showed that a destroyed resource ID can identify a
different current client after its original connection closes. Current range
identity cannot be used as historical transient ownership; that proposed
Flatpak repair remains withheld.

On source `29e2071`, the original local Ornith LHN-091 recovery task ended
naturally after six phases and 273.41 seconds at 19:39:02 UTC. Its authoritative
terminal status was blocked after three consecutive unverified phases; the runner
records failed independent checks. Six unique dispatched attempts included one
physical pointer and five native patterns, with no keyboard, Moondream, Kev,
reader or memory calls. No deliverables were produced. All three inputs and 138
preexisting window metadata rows were preserved, and all three owned windows
closed. Its unchanged 28-member bundle is
`f7561ced380f8c892fb4c1c7afdb4a4b9f460f4b25868a2be1f8808f9f019ca9`.
Two cached intermediate focus outcomes were freshly verified before reuse with
zero input and zero progress credit. Multiline typing and false cached-prefix
rejection were not exercised by this attempt.
Independent grading found nine failed criteria and preservation C08 passed under
the frozen input/window oracle; unrelated document contents were not inspected.
The sealed local campaign now contains 24 attempts, twelve model/case cells,
ten task IDs and zero whole passes. Remote cohorts remain separate.

That run also supplied an exact private-planner contract failure: phase 2 switched
to Files while expecting the other editor window to be active. The observed
Files activation could not satisfy that predicate. The causal switch candidate
binds the private planner effect to `$target` at every phase position, leaving
public definitions and the native executor unchanged. Source and decoder proofs
for this restriction do not establish a new loaded-model task pass.
