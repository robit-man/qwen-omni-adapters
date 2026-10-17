# Computer-use reliability implementation plan

Status: implementation started; no full-task success claimed.

Research basis: [broader primary-source and implementation review](computer-use-research-review.md).
The original guide is a hypothesis, not a required architecture. The review
adds decision-interface evaluation, explicit failed-strategy suppression,
pre-input refresh versus uncertain-input reconciliation, retrievable context,
and independently verified workflow outcomes. Keep these as separate measured
cycles; published results do not transfer to our pinned models or hardware.

## Objective and execution rules

Enable Qwen and Ornith to perform real, long-horizon desktop work through
native input, Kev finite decisions, and Moondream visual grounding. Use the
existing 100-task live suite and the original documentation/dossier task as
acceptance work. Text instructions are the current input scope. Local model
work stays on broker-leased GPU 1; remote Ornith targets are Eggs .31 and .41.

Follow each cycle through a concrete failing criterion, bounded hypothesis,
small patch, relevant tests, full repository validation, one deployment and
bounded live evidence. Commit and push verified fixes to main, observe Egg
updates and test live there. Preserve failed attempts and distinguish unique
tasks, retries, primitive diagnostics, and whole-task passes. The user has
authorized this full sequence; each cycle gets its own checkpoint.

Starting revision: `8184f0e345ea291eefcc03a0280692598c13e03a`.
The local causal checkpoint is
`runtime-data/gpu1-acceptance/operator-audit/root-causal-checkpoint-8184-provider-confirmed/`.
Its evidence establishes an ineffective native setter, focus-only verification,
replayed recovery, whole-snapshot freshness rejection from hidden UI changes,
and redundant independently generated objectives. The exact internal Firefox
setter failure remains unknown. These facts are not benchmark passes.

## 1. Mandatory operation verification

The controller derives mandatory postconditions from operation arguments.
Model expectations may add constraints but cannot replace required effects.
Start with `set_value`: the target must expose the requested value; an already
focused target and a provider acknowledgement do not verify a text write.
Extend the same rule to checked state, focus, text selection, and window
activation where the operation supplies an unambiguous effect. Do not invent
semantic effects for arbitrary clicks or key chords.

Acceptance: an acknowledged ineffective setter with `expect: focused` remains
unresolved; a successful setter verifies its exact value; wrong, missing or
truncated text cannot produce a false pass. Independent terminal checks remain.

## 2. Explicit text-entry strategies

Add typed text-entry intent with target, text and replace/insert/selection mode.
Resolve and focus the field; inspect value, caret/selection and coverage;
execute the explicitly selected native or physical strategy; verify the text.
Keep native setters and physical input distinct. A fresh, established no-effect
result can make physical entry an explicit recovery candidate. Uncertain effects
require reconciliation before another write. Remember failed strategies within
the control/application session, with generation/version invalidation.

Acceptance: real native fields and browser address bars accept replacement,
insertion, selection replacement, multiline and Unicode text without duplication.
The original ineffective-setter fixture can recover through verified physical
input. No hidden fallback changes the meaning of `set_value`.

## 3. Fresh action revalidation

When a snapshot changes before input, obtain fresh state and re-enumerate the
candidate rather than reusing stale authority. Recheck application/window scope,
target identity/generation, eligibility, focus, modal state and relevant geometry.
Allow a bounded fresh decision when unrelated hidden content changed. Never
blindly retry an operation that may have dispatched. Preserve final native input
guards and serialize focus-dependent input.

Acceptance: hidden browser updates do not permanently block Return; changed
target, foreground window, modal, geometry or authorization still prevent stale
input. Post-dispatch uncertainty cannot trigger duplicate actions.

## 4. Executable recovery

Maintain target, operation, strategy, intended effect, observed effect and
dispatch certainty in the execution ledger. Historical receipts stay historical.
Candidate generation excludes a known ineffective repeated strategy as a fresh
action. Recovery chooses refresh, focus restoration, another input strategy,
visual inspection, modal handling or planner escalation according to typed
capabilities and observed conditions. Keep bounded retry and cancellation rules.

Acceptance: unchanged setter/replayed outcomes cause a different applicable
strategy or an honest bounded stop; editing narration or weakening a terminal
predicate never creates progress.

## 5. Single objective source

For phases that already own an objective, generate it once and copy its exact
value into persistent planner state with source provenance. Other phases keep
one explicit commitment objective. Remove duplicate generation and cross-field
string-equality rejection. Preserve original user goal and independently checked
requirements separately from current subgoal and model completion claims.

Acceptance: valid subgoal and boundary plans are admitted without duplicate
wording. Invalid structure, stale epochs, unauthorized scope and empty objectives
still reject before input. Original acceptance requirements cannot be erased.

## 6. Layer ownership and bounded information

| Layer | Responsibility |
| --- | --- |
| Qwen / Ornith | Interpret requests, plan subgoals, reason over documents and recover from unexpected situations |
| Kev | Select among currently valid finite action/recovery candidates |
| Moondream | Read or locate visual content when native semantics are insufficient |
| Native executor | Resolve targets, move the real pointer, click/type/select/drag/scroll and observe effects |
| Task controller | Maintain task state, verify milestones, reconcile effects and govern replanning |

Fast decisions receive current subgoal, relevant controls, candidate effects and
recent receipts. Planner requests receive milestone state and unresolved failures.
Detailed source observations remain available for bounded expansion. Known
deterministic sequences do not require a planner call for every primitive.

Acceptance: actual receipts show the invoked layer, candidate set and outcome;
pixel-only cases really invoke Moondream, ambiguous finite choices invoke Kev,
and observations or policy prose cannot be counted as input or task completion.

## 7. Reusable, preconditioned sequences

Build verified sequences for text editing, navigation, menus, file dialogs,
saving and archives. Check preconditions at each transition; return only the
executed prefix and fresh state on failure. Cover physical movement, single,
double and right click, selection, drag and scroll. A click does not establish
that a file was saved or that a task completed.

Acceptance: sequences work across held-out applications/layouts and stop at
broken assumptions; final artifacts and reopened contents match the task.

## 8. Live validation and release

1. Reproduce each diagnosed defect against its focused repair.
2. Run the original GUI documentation → dossier → matrix → archive → reopen →
   memory task with unchanged task requirements.
3. Execute the existing 100 distinct tasks, covering text editing, pointer use,
   windows/dialogs, real websites, files, archives and cross-application work.
   Include held-out layouts/apps, focus changes, delayed content and modals.
4. Test Qwen and Ornith locally on GPU 1, and applicable Ornith cases on .31/.41.
5. Record outcome, actual layer calls, physical input, observed effects, artifact
   checks, latency and recovery after each task. Independent read-only evaluation
   judges the result and never solves the task for the model.

Pass criteria concern complete tasks, not service readiness, unit-test counts,
model narration, or one successful primitive. Report unsupported cases and
unresolved faults explicitly. No arbitrary-task or SOTA parity claim precedes
held-out live evidence.

## Checkpoints

| Cycle | State | Evidence |
| --- | --- | --- |
| Plan and research persistence | Written and read-back verified | This document and [108-file archive manifest](research/computer-use-2026-10-17.manifest.json); raw corpus outside runtime cleanup |
| Baseline validation | Passed | 3,997 tests; `/tmp/qwen-reliability-baseline-20261017.log` |
| 1. Mandatory verification | Local gate and live provider passed | 4,005 tests; real Firefox setter unresolved, physical typing verified; [evidence](computer-use-reliability-evidence.json) |
| 5. Single objective source | Source and grammar gates passed; live branch unexercised | 4,008 tests; 9 native cases per model; original live task failed at native scope observation before open-file dispatch |
| 3. Native workflow freshness | Source gate and real Firefox provider passed | 4,027 tests; real page mutation rejected the old Return offer, one fresh decision dispatched once and verified the destination; three setup failures retained |
| 2a. Full text replacement | Provider passed; planner discovery defect exposed | 4,055 tests, 28 new regressions; real Firefox physical replacement and navigation passed. One live Kev experiment exposed probability splitting between equivalent transports. Host now selects transport; Kev selects ambiguous targets. Original Ornith task failed: compressed observations omitted optional text coverage, incorrectly hiding the new family. |
| 2b. Text-family discovery | Source, grammar and live discovery passed | 4,060 tests; actual compressed phases 3–5 now offer text entry, while executor coverage remains mandatory. Original task still failed; all five responses generated a generic commitment objective before selecting a phase. |
| 5b. Phase before commitment | Decoder restriction removed; whole task failed | 4,060 tests and 4 native grammar cases per model passed. Live replies used assessment → phase → commitment, but all three still selected primitive workflows with incorrect effects. |
| 6a. Decision-interface evaluation | Nine actual Ornith calls completed; no task credit | Removing raw workflows did not help. A compact typed-family projection selected the correct Firefox text entry; file-navigation decisions remained weak. All requests/responses and failures retained. |
| 2c. Keyboard/scroll input boundary | Contract and native grammar passed; live pending | 4,076 tests; 16 new regressions and 7 native grammar cases per model. Private guarded input returns an unknown-effect observation without fabricated success predicates. |
| 6b. Compact planner | Live text replacement passed; whole task failed | 4,101 tests and 9 native grammar cases/model passed. Opt-in interface generated eight live phases, including four physical URL replacements with exact readback. It did not submit navigation, falsely claimed page acquisition, then repeated three low-confidence pointer attempts. Full task failed in 75.73 s with no deliverables. |
| 7a. Text preparation and submission | Live navigation passed; whole task failed | 4,121 tests and 9 native grammar cases/model passed. Ornith selected the sequence, which verified the URL and pressed Return once. Fresh AT-SPI Document.DocURL proved Python 3.12 navigation. The unchanged task then stalled: schema-budget fallback removed all desktop actions, and seven inspections repeated unusable Moondream readings. Cancelled with all owned windows closed; no deliverables. [Durable experiment](research/computer-use-text-submit-2026-10-17.manifest.json). |
| 6c. Capability-preserving compression | Contract, grammar and loaded decision passed | Reproduced the lost controls at 58,434 bytes. Sharing identical target definitions reduces the working schema to 53,666 bytes within the unchanged 57,344-byte limit. All 4,130 tests and 6 native grammar cases/model passed. Actual Ornith selected the missing 3.10 URL submission from the captured state; no input or whole-task credit. [Durable experiment](research/computer-use-capability-compression-2026-10-17.manifest.json). |
| 6d. Native document reading | Live source delivery passed; whole task failed | Seven Moondream probes failed reliable document reading; cropping corrected language recognition but not the error condition. Native traversal instead skips off-screen subtrees without granting absence/input authority, and a separate 4,000-character excerpt survives control compression. All 4,139 tests passed. The original live Ornith task received the exact visible 3.12 documentation, then cycled between applications without producing outputs. [Diagnosis](research/computer-use-native-reading-2026-10-17.manifest.json) and [live record](research/computer-use-native-reading-live-2026-10-17.manifest.json). |
| 4b. Repeated applied transitions | Contract passed; loaded model recovery failed | Complete workflows now enter recovery when their verified semantic transition was already observed; actual receipts remain verified. All 4,142 tests passed. Replaying the real first six phases produced the correct recovery packet, but loaded Ornith still chose another switch. No additional desktop run. [Durable record](research/computer-use-transition-recovery-2026-10-17.manifest.json). |
| 6e. Historical observation expansion | Contract and decoder passed; model recovery not reliable | Task-owned historical references preserve exact native excerpts across switches, with bounded storage, session isolation and no input authority. All 4,155 tests and 6 native grammar cases/model passed. Loaded Ornith still repeated a switch; generation order, reasoning and sampling comparisons did not establish reliable recovery. No new desktop run. [Durable research and experiments](research/computer-use-observation-recall-2026-10-17.manifest.json). |
| 6f. Prerequisite-first subgoal handoff | Live target selection improved; whole task failed | All 4,171 tests and 9 native grammar cases/model passed. Three captured-state comparisons selected useful opening controls. In one unchanged live Ornith task, six planning/action requests selected the Input folder across three phases; Kev abstained each time. Zero inputs, no deliverables; all owned windows closed and inputs unchanged. [Durable record](research/computer-use-subgoal-handoff-2026-10-17.manifest.json). |
| 6h. Primitive uncertainty escalation | Live recovery passed; whole task failed | All 4,192 tests and 8 native grammar cases/model passed. Actual Ornith resolved two Kev abstentions with physical double-clicks, opened Input and task-brief.txt, and received its actual text. At phase five, accumulated abstentions halted the task before the next recovery; no deliverables. All windows closed, inputs unchanged. [Durable evidence](research/computer-use-pointer-recovery-2026-10-17.manifest.json). |
| 6i. Deferred-choice accounting | Live handoff passed; whole task failed | All 4,196 tests passed. One pending fast-policy deferral no longer spends the failure budget; ignored or failed recovery remains bounded, and legacy controllers retain their prior limit. Actual Ornith reached 13 phases, five physical inputs and a cached repetition; it read the brief but failed to open the next input. No deliverables; inputs preserved and owned windows closed. [Durable record](research/computer-use-deferral-budget-2026-10-17.manifest.json). |
| 6j. Narrow primitive resolution | Contract/model comparisons passed; live path unexercised | All 4,206 tests and 8 grammar cases/model passed. Twenty-six captured-state model comparisons are preserved; a narrow named-operation choice selected folder double-click and single Open activation, while window selection remained imperfect. The unchanged live task failed before recovery because Kev requests timed out/rejected; zero inputs. [Comparisons](research/computer-use-control-interpretation-2026-10-17.manifest.json), [implementation/live record](research/computer-use-narrow-pointer-2026-10-17.manifest.json). |
| 6k. Policy request deadline | Contract and live escalation passed; whole task failed | All 4,207 tests passed, including a real HTTP reply beyond the old deadline. The unchanged live task received three valid Kev replies and three narrow resolutions; two clicks opened Input and the editor Open popover. The Main menu move reached the recorded target coordinates but failed binding verification; a later window switch lost native active-window observation. No outputs; all owned windows closed and input hashes unchanged. [Durable record](research/computer-use-policy-deadline-2026-10-17.manifest.json). |
| 6g. Sampling transport | Contract and loaded transport passed | All 4,161 tests passed. An actual call through the patched adapter implementation matched all ten requested sampler values in the GPU 1 worker’s native slot parameters. Caller defaults and `think` remain unchanged; no whole-task credit. [Durable record](research/computer-use-sampling-transport-2026-10-17.manifest.json). |
| 6l. Popup switching prerequisite | Corrected native provider passed; whole task failed | All 4,221 tests passed. Fresh native popup relations and exact accessible child membership withhold window switching until a separately guarded dismissal. The real provider rejected the blocked switch without input, dispatched Escape, restored switching and passed three subsequent Files readbacks. The unchanged Ornith task made 19 model requests, one pointer click and four window switches, but opened no popup and produced no outputs. Its failure does not exercise the guard. [All failed candidates, provider proof and live record](research/computer-use-popup-prerequisite-2026-10-17.manifest.json). |
| 6m. Subgoal continuity | Context repair and two captured-state choices passed; live criterion not reached | All 4,223 checks passed. Six actual model comparisons supported retaining the prior plan in two Files states, with no improvement in the Firefox state. The original live LHN-041 run failed earlier with one breadcrumb click, no brief read and no outputs. Separate LHN-041 and LHN-047 baselines also failed; Egg .41 completed a bounded LHN-047 attempt but lost foreground identity. [Requests, responses, failed tasks and host evidence](research/computer-use-planning-continuity-2026-10-17.manifest.json). |
| 2d. Unicode delivery | Local and ARM gates passed; repeated Firefox readback incomplete | All 4,242 checks passed locally and on Egg .31, including delayed real keyboard delivery and restoration. Ordinary Firefox verified one exact accented/CJK replacement; a second acknowledged replacement lost independent observation. A→B→A and full tasks remain incomplete. [Rejected candidates, source and final evidence](research/computer-use-unicode-delivery-2026-10-17.manifest.json). |
| Remaining 2, 4, 6–8 | Pending | Insertion and partial selection replacement, broader recovery/delegation, context expansion, reusable task sequences and whole-task acceptance remain incomplete. Separately bound primitive-choice boundaries retain their existing freshness behavior. |

Observed deployment blocker: Egg .31 deployed 31fc3a3 after 4,060 checks passed, then reproduced the actual XTEST Unicode failure (`café\nnaïve` becomes `café\nnave`) during the a13f410 update, with 4,059 other tests passing. The isolated nine-case recheck had passed; no transport fix is claimed. Its updater stopped before replacing the active runtime. Egg .41 adopted a13f410 while preserving its paused core.

Subsequent .31 diagnosis captured real GTK key events in an isolated X server:
three of four ordinary xdotool trials lost accents and delivered keyval zero.
Holding distinct unused mappings for é and ï through recipient readback passed
all four matched trials, including two deliberately delayed recipients. The exact
original keymap was restored. This isolates mapping lifetime/reuse as the repair
target; production transport remains unchanged. [Durable evidence](research/egg31-unicode-mapping-2026-10-17.manifest.json).

## 6l. Popup prerequisite for native window switching

Acceptance: switching applications while a current owned popover is open must
not leave X11 activation and the accessible input owner inconsistent. Keep the
current observable state usable and allow a separately guarded dismissal before
switching. Do not manufacture an active accessibility window or retry input.

Evidence: `pointer-delivery-cycle-20261017/fixed-switch-diagnostic` reproduced
ten failed reads with only 29 visited nodes after an editor popover remained
visible over Files. `focus-binding` reproduced three failures, then three
successful reads after reactivating the owned editor, sending one guarded
Escape, and switching to the same Files window. Ordinary X11 keyboard-focus
queries followed Files even during the failure; they do not detect this grab.
GTK's Popover documentation describes input grabs and Escape dismissal. The
acquired documentation and pinned GTK 4.14.5 source are retained with the audit.

Bounded hypothesis: derive open popup evidence from native HAS_POPUP /
CONTROLLER_FOR relations with visible ancestry in the current owned window.
Require exact current parent-child membership as well: retained hidden GTK
objects can report visible flags and parent pointers. Withhold switch
capabilities while positive popup evidence exists and recheck
before dispatch. Preserve popup controls and a separate guarded Escape choice.
No application-name rules, unscoped input, implicit dialog cancellation, or
relaxed ownership checks. Validate negative controls and real editor switching,
then run one unchanged loaded-Ornith LHN-001 window. The earlier intermittent
pointer-window rejection remains a separate, unproven cause.

## 6m. Preserve the previous plan at subgoal selection

Acceptance: after reading the brief and returning to Files, subgoal selection
must retain the unmet authority-ledger acquisition instead of restarting the
already observed brief. The original LHN-041 task, input bytes and evaluator
remain unchanged; complete-task delivery remains a separate required outcome.

Evidence: the 192.29-second LHN-041 run opened Input and the brief, then switched
between Files and the editor until it stopped without outputs. Its phase 9 and
11 subgoal requests omit the retained commitment and unmet constraints that the
following action-selection request receives. Exact captured-request comparisons
reproduced brief rereading in both states. Adding only the existing commitment
changed both replies to authority-ledger acquisition. A third comparison in
Firefox still reread the brief. All six requests and results are retained under
`breadth-evaluation-20261017/continuity-comparison`; they dispatched no input and
earn no whole-task credit. The separate editorial baseline also failed, with
three physical inputs and no outputs.

Bounded hypothesis: preserve the exact existing commitment in the subgoal
projection as `retained_plan`, including its planner-claim authority and lack of
input authority. Keep current observations, historical-source expansion, the
original goal and all execution guards unchanged. Check first-turn/continued
task isolation, run the full gate, then one original LHN-041 live window.

## 2d. Unicode mapping lifetime on X11

Acceptance: real keyboard delivery on Egg .31 preserves `café\nnaïve`, including
a recipient deliberately delayed by 250 ms, and restores the original keymap.
The updater's 2834d7f gate reproduced `café\nnave` with 4,220 other checks passing.
Earlier controlled trials preserved all characters when distinct unused mappings
remained bound through recipient readback. Their source and event evidence are
already in the Unicode mapping archive.

Bounded hypothesis: reserve distinct unused keycodes for the requested Unicode
characters, address those codes explicitly, and hold their mappings through a
bounded recipient readback. Keep ASCII and newline typing, foreground/ownership
guards and independent effect verification. Allocation or capacity failure must
send no input; partial input must remain unresolved and must never be replayed.
Restore only mappings still owned by this attempt, including normal cancellation.
Test delayed delivery, cleanup and exhausted capacity before one local provider
window and the affected Egg's actual gate. This is a transport repair, not proof
of model planning or whole-task completion.

The first ordinary Firefox probe invalidated a single unacknowledged command
chain: mixed ASCII/Unicode arrived out of order despite passing private GTK
checks. A controlled segmented probe preserved every prefix through accented
and CJK text. The revised hypothesis therefore also requires recipient prefix
acknowledgement between segments, with fresh dispatch guards and a five-second
total budget. The recorded IME sources explain possible asynchronous processing;
the exact Firefox reordering mechanism is not proven.

Separate follow-on criteria, not silently folded into this transport change:

- Define a proven provider-specific mapping between Firefox ATK's supplementary
  character/U+FEFF representation and user text, including selection offsets and
  legitimate U+FEFF. Do not strip invisible characters globally.
- Capture the exact accessibility read predicate that failed after a second
  Unicode replacement. The transport acknowledged that prefix, but independent
  full observation failed; A→B→A is incomplete and must not be marked passed.
