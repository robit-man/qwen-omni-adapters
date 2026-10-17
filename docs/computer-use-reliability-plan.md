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
| 4b. Repeated applied transitions | Next bounded repair | Existing receipt validation recognized repeated window transitions in phases 6–8, but complete workflow status reset the failure counter anyway. Preserve verified input receipts while requiring recovery on an already observed transition. Evaluate actual model recovery from those exact states before another desktop run. |
| Remaining 2, 4, 6–8 | Pending | Insertion and partial selection replacement, broader recovery/delegation, context expansion, reusable task sequences and whole-task acceptance remain incomplete. Separately bound primitive-choice boundaries retain their existing freshness behavior. |

Observed deployment blocker: Egg .31 deployed 31fc3a3 after 4,060 checks passed, then reproduced the actual XTEST Unicode failure (`café\nnaïve` becomes `café\nnave`) during the a13f410 update, with 4,059 other tests passing. The isolated nine-case recheck had passed; no transport fix is claimed. Its updater stopped before replacing the active runtime. Egg .41 adopted a13f410 while preserving its paused core.

Subsequent .31 diagnosis captured real GTK key events in an isolated X server:
three of four ordinary xdotool trials lost accents and delivered keyval zero.
Holding distinct unused mappings for é and ï through recipient readback passed
all four matched trials, including two deliberately delayed recipients. The exact
original keymap was restored. This isolates mapping lifetime/reuse as the repair
target; production transport remains unchanged. [Durable evidence](research/egg31-unicode-mapping-2026-10-17.manifest.json).
