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
| 2b. Text-family discovery | Next bounded repair | Offer the family from positive editable capabilities; execution still requires fresh full text coverage. |
| Remaining 2, 4, 6–8 | Pending | Insertion and partial selection replacement, broader recovery/delegation, context expansion, reusable task sequences and whole-task acceptance remain incomplete. Separately bound primitive-choice boundaries retain their existing freshness behavior. |

Observed deployment blocker: Egg .31 retains an actual XTEST Unicode failure (`café\nnaïve` becomes `café\nnave`), with 4,004 other tests passing. Its updater stops before replacing the active runtime. Egg .41 adopted cycle1 successfully while preserving its paused core.
