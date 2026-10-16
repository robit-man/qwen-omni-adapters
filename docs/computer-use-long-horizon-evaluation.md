# Long-horizon native computer-use evaluation

Status on 2026-10-15: evaluation specification; no SOTA parity established.
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
