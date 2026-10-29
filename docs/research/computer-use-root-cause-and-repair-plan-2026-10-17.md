# Computer-use root causes and ordered repair plan

Status: active, incomplete. Zero verified whole-task passes. This current
checkpoint supersedes the historical sections below. Source tests and saved-image
inference are not desktop acceptance.

## Current diagnosis

The deepest supported architectural cause is an unreliable immediate-intent →
observed-target → dispatched-input → intended-effect contract. Information is
lost between layers, and some checks establish a weaker or different fact than
the next layer requires. There are several reproduced mechanisms, not one proven
internal model defect. Model size alone is not an established explanation.

1. **Execution and verification mismatch.** A focused empty field was credited
   as a completed text write. A native setter repeatedly left it empty while
   physical typing worked. Recovery kept the ineffective strategy available.
   The exact Firefox internal setter failure remains unproven.
2. **Native discovery boundary failure.** A real owned Save As dialog passed
   X11 process/start/UID and direct-transient checks, but twelve paired native
   observations reported zero ACTIVE accessibility roots. The native reader
   rejects an inactive window boundary before visiting its descendants. The
   code already pumps GLib events and clears accessibility caches. Actual069
   now proves the provider itself returns ACTIVE=false for the dialog: seven
   direct GetState reads agree exactly with all refreshed accessibility reads.
   Plain editor ACTIVE=true before the shortcut; both roots lose ACTIVE after
   Save As opens while a descendant remains FOCUSED. This is a compatibility
   failure in our activation requirement, rather than a stale-cache-only fault.
   Exact installed GTK source and the mapped provider library corroborate the
   role-dependent ACTIVE behavior. A focused descendant alone is not enough
   authority;070 must prove its current owned foreground root before input.
   Actual070 now observes the real dialog and184 complete controls with raw
   ACTIVE=false. Its first write then fails a cumulative observation budget;
   setter dispatch/effect remain unknown. Repeated full proof work is the next
   bounded cause to measure, without relaxing current authority or effect checks.
3. **Lossy semantic translation.** Actual GPU1 predicate compilation preserves
   complete meaning in 39/56 cases. Quoted literals, OR/AND, negation and some
   qualifiers change. A valid schema or copied source substring does not prove
   preservation of the requested meaning.
4. **Misassigned visual judgments.** After Moondream pointing, the candidate
   asks Qwen for another spatial/identity verdict. Removing numerical point
   text in066 still fails10/12 with the same two false blocks. Target-blind
   fact reporting in067 passes only1/4 complete strict cases. These are real
   perception/contract failures, not output-format failures.
5. **Evidence retention and operation gaps.** Complete component traces can
   exceed the65,536-byte inline limit and prevent retention of recovery
   evidence. Visual keyboard/text, selection, scrolling and dragging also
   remain unimplemented in the new visual route. These are implementation
   defects, not merely missing tests.

## Concrete repair and acceptance order

1. **Recover the native dialog handoff (070).** The068/069 provider comparison
   is complete. Implement a separate private activation witness combining the
   exact current host window/process/scope/transient relationship, provider
   credentials, strict geometry and a complete fresh focused-control ancestry.
   Preserve the existing ACTIVE path. Bind the witness into the native revision,
   rederive it before content release and at every input authority boundary,
   including immediately before a pointer button. Share existing deadlines and
   read/node bounds; reject ambiguous, incomplete, stale and foreign roots.
   Preserve mandatory mouse-button release after a dispatched down event.
   The isolated070 revision2 implementation passes346 focused controls,
   independent review and its5,052-test full gate; a070-only release patch
   separately passes4,322. Actual native dialog discovery passes, but the write
   returns unresolved on native_observation_budget_exceeded; readback/Cancel
   remain unrun. Diagnose071 cumulative cost before another source change or
   fixture. Do not publish this partially passing candidate.
   Then return to the unchanged whole task; a dialog pass is not its completion.
2. **Preserve one typed operation and its intended effect.** Bind the original
   request to an immutable target claim and source-backed operands. Represent
   ALL/ANY/NOT explicitly; preserve exact strings, content/label/spatial/state
   distinctions and row/parent relationships. Models choose observed IDs;
   host code performs exact expansion and admits only implemented complete
   expressions. The operation determines its minimum verification: a write
   requires value readback, navigation requires destination state, and save
   requires independently observed artifact/content. Focus alone cannot
   satisfy a write. Keep the56-case semantic regression and evaluate operand,
   operator and logic selection with actual weights.
3. **Finish private expandable evidence (063, in parallel).** Retain complete
   immutable traces in a session/task-owned durable archive and give the
   controller bounded references. Verify bytes, hashes, ownership and lifecycle
   on expansion. Keep TTL/Trash, quotas and cancellation; native operations
   must not depend on archive filesystem health. The isolated implementation
   passed377 focused CPU controls after repairing read-after-close, removal
   count, atomic lease-renewal and live-session TTL continuation faults. Its
   independent storage/lifecycle review and full5,897-test source gate pass on
   refrozen source. This storage component is verified in isolation; integration
   and actual desktop acceptance remain open. Stop storage edits here.
4. **Rebuild visual fallback around appropriate layer ownership.** The recent
   Omnius demo genuinely completed four live actions with Qwen/Kev/Moondream.
   It assigns semantics to Qwen, bounded choices to Kev, grounding to Moondream,
   and coordinate mapping/current ownership/input/effect checks to the host.
   It has no second Qwen spatial verdict. Reuse that separation, while retaining
   our stronger distinction between visual hypotheses and native facts. Do
   not blindly remove the failed gate, invent enabled/visible native nodes,
   lower Kev thresholds, or dispatch unvalidated points. Any replacement needs
   its own independently evaluated positive/absent/ambiguous/wrong-point
   contract. Stop iterating point-verdict prompts under the failed conditions.
5. **Complete operation coverage and useful recovery.** Implement missing
   visual text/key, selection, scroll and drag through typed operations and
   separate post-action readback. Keep completed subgoals and failed strategy
   history. A failed setter must lead to a supported alternate primitive or
   fresh evidence, not another cached success claim or repeated failed input.
   Preserve uncertain-dispatch locks and use fresh observation before retry.
6. **Prove the original outcome locally, then publish and test Eggs.** Require
   a real edit/save/reopen workflow with correct content, native→visual→native
   handoff where needed, and independently verified effects. Run unchanged
   long tasks with local Qwen and Ornith using text goals and GPU1. Publish
   individually accepted repairs to main; observe actual auto-update/restart
   and rerun live Ornith tasks on.31 and.41. Continue the original100-case,
   200-model-cell,960-criterion scope. A component or fixture pass does not
   replace full desktop and long-horizon acceptance.

## Current measured checkpoint and limits

066:5,748 source tests pass;28 actual GPU1 requests; original panel10/12,
including two false blocks. 067:4 actual GPU1 requests; exact pointed facts2/4,
derived relations2/4, explanation factuality2/4, all required checks1/4.
Together these latest two experiments made32 actual model requests and zero
GUI inputs. The healthy eight-process CUDA cohort remains unchanged. Neither
failed candidate is adopted, committed or pushed. No verified whole-task pass
has been established.

068 then completed one owned physical Save As shortcut and three metadata
samples in3.679 seconds. The native reader still rejected the dialog. Direct
GetState is unknown because all seven probe attempts failed; the diagnostic's
capture pass does not close that required comparison. Both created windows
closed,12 protected identities and preexisting managed windows stayed unchanged.
See `computer-use-native-dialog068-results.md` and the separate069 probe repair.

069 then completed one separately bounded owned Save As shortcut in3.746 seconds.
All seven direct provider reads succeeded. The plain frame returned ACTIVE=true;
three modal samples returned ACTIVE=false for frame and dialog, exactly matching
four GI/cache stages. Fresh focused-descendant ancestry was observed, but the
128-node diagnostic scans were truncated and cannot establish production-wide
uniqueness. Both created windows closed;12 protected processes and14 previous
windows remained unchanged. No model calls occurred in068/069. See
`computer-use-native-provider069-results.md` and the070 plan.

070 then ran one owned physical fixture in7.8279 seconds. Current provider
ACTIVE=false was observed with184 complete native nodes, but mandatory write
verification failed on an observation budget. Its setter effect remains unknown;
no Cancel click followed. Two new windows closed once and all15 current prior
windows plus12 protected processes remained unchanged. See070 results. The
earlier14-window precondition abort is preserved separately. No model request,
source adoption, new commit or push occurred.

The correct recent Omnius controller and Oct11–14 commits were inspected.
Its genuine three-model success was a four-action host-enumerated browser task;
its longer cross-application desktop fixture used zero model requests. Those
results are useful implementation references, not evidence that arbitrary
planning or our current stack already works.

Fresh read-only Egg checks find healthy resident Ornith trunks and working
Omni auto-update on both hosts at24f1691. Both currently have older MD2
configuration and no observed Kev policy worker. New repair source is not
adopted there. Remote long-task acceptance remains owed.

Durable evidence:

- `computer-use-point-projection066-results.md` and
  `computer-use-point-observation067-results.md`: latest actual visual results.
- `computer-use-predicate-family062-results.md`:56-case semantic results.
- `/home/roko/Documents/qwen-omni-research/2026-10-17/omnius-current-layer-independent-comparison-20261017/review.json`:
  source and actual-demo comparison, SHA256
  `4efddef1795a2bed0debfade4605cfa38502781324b98cdd97c6527ec05fd020`.
- `/home/roko/Documents/qwen-omni-research/2026-10-17/point-observation067-independent-controls-20261017/independent-final-grade.json`:
  independently graded actual067 result, SHA256
  `210a68f287bbaf3c86dda79a49327e610abeabb8b59becad380211848ef79a16`.

## Earlier checkpoints (historical)


Latest implementation evidence: separate062 passed5,798 source tests, then
completed56 actual GPU1 compiler calls. Independent semantics is39/56 with four
wrong admitted quoted literals and one false block. C20/C22 are repaired, but
full predicate acceptance fails; see `computer-use-predicate-family062-results.md`.
Source copying alone does not identify whether quotes are delimiters, and a
conjunction-only representation still loses OR/negation and some qualifiers.
No062 desktop adoption occurred. The separate063 real CPU surface counterexample
proves65KiB trace-retention overflow; its private durable expansion proposal is
not implemented. Point-view061 revision2 passed5,748 source tests, then its first
unchanged original12 actual panel failed10/12 on060 compiler source. Both point
positives remain falsely blocked; no held-out or physical run followed.

Earlier checkpoint: CU.TARGET.REPRESENTATION.060 passes its5,696-test source
gate and12/12 original actual compiler checks. Overall actual semantics remains
failed10/12: both positive point queries are rejected, including one enum that
contradicts its own correct containment explanation and one wrong row location.
Exact original images and point packets reached the worker. Numeric-point
grounding and decision consistency are the current spatial bottleneck; the
internal model cause remains unproven. Button-inventory completeness claims
also overstate observed coverage. See060 results and the separate061 plan.

Additional current evidence: the separate060B actual compiler breadth run
passes all18 supported selectors but simplifies two unsupported requests:
relative position becomes any button, and document content becomes a filename.
Three repeated-source copies are guarded by host ambiguity rejection. Overall
expected interpretations match19/24. The next predicate-family062 hypothesis
moves capability admission to host contracts over explicit semantic predicate
families, retaining unsupported requirements for bounded discovery. This is
separate from061 and neither is implemented at this checkpoint. No source-bound
copy or schema alone proves that the original constraints were preserved.

Prior checkpoint: CU.VISUAL.FACTS.059 passed its5,657-test source gate, then
failed the actual GPU1 panel. Seventeen native component requests produced
zero correct complete target compilations and two correct final assessments
out of twelve. The compiler chose the expected broad kind/operator but mixed
the role word into the label, generated incorrect source offsets, or emitted
fields incompatible with its chosen operator. The output schema admitted
combinations the semantic decoder rejected. No point assessment was reached.
This makes the current repair target a simpler, internally consistent target
representation; it does not establish useful desktop input. Details are in
`computer-use-visual-facts059-results.md`. Earlier checkpoints below remain
historical evidence, not descriptions of the current implementation status.

The concrete next work order is in
`computer-use-target-representation060-plan.md`. It preserves the original
whole-task criteria. Accepted fixes should be published after local source and
actual acceptance, then observed and tested on both Eggs; the full breadth
matrix must not become a reason to indefinitely withhold an accepted repair.

Latest checkpoint: CU.VISUAL.NATIVE.057 repairs the backend grammar failure
and completes the unchanged twelve actual GPU1 cases. Structure passes 12/12;
semantics passes 8/12 with three incorrect positive gates. Target identity,
ambiguity and point membership remain unreliable, so no desktop adoption is
permitted by this result. Details and the next isolated assessment work are in
`computer-use-visual-native057-results.md`. Earlier sections below preserve
the chronological diagnosis and original owed acceptance scope.

Follow-up CU.VISUAL.ORDER.058 changed only constrained property generation
order, then repeated the same twelve actual cases. The result remains8/12 with
three incorrect positive gates. Correctly enumerating five candidates still
produced a uniqueness claim. Field ordering is insufficient; the next bounded
contract design separates typed observed candidates and host-computed matching/
cardinality from model verdicts. Target identity and spatial membership remain
separate failures. See `computer-use-visual-order058-results.md`.

Follow-on checkpoint: the P054 snapshot repair now passes independent controls
and its source gate. An eight-call actual GPU1 pilot demonstrates absent-target
false coordinates and different semantic-reading behavior; full results and
remaining limits are in `computer-use-perception054-results.md`. No source
candidate has yet passed an actual-host visual action or whole task.

## Direct answer: deepest verified cause and current repair

The system does not yet preserve a reliable immediate intention → observed
target → physical operation → intended effect loop. This is an execution and
orchestration defect: different layers have been asked to establish facts their
outputs do not establish. It is not evidence that all remaining failures have
one cause, or that the backbone's size is the limiting factor.

- A prerequisite (focus) was credited as the requested write effect.
- Semantic information and immediate operations were lost or changed at
  planner/policy boundaries, while unsuccessful strategies remained selectable.
- Vision was gated by the native discovery failure it needed to recover from.
- Reading a retained image was rejected by an irrelevant current pixel change.
- MD point outputs were implicitly treated as target-existence evidence. In the
  actual eight-call GPU1 panel, both absent targets still received coordinates.
- The visual planner received a textual account rather than the captured image;
  the generic MD account was wrong on both measured scenes.

The bounded repair, CU.VISUAL.ACTION.055, is now implemented in an isolated
candidate and passed combined source review/gates. Omni receives the exact full
captured image and original goal. A structured Omni target-presence assessment
precedes MD pointing. A separate latest-image Omni assessment must confirm both
unique target presence and that the fixed proposed point is on that target.
Kev retains its finite choice and existing confidence/encoder limits. The host
then permits at most one scoped physical attempt. A click acknowledgement is
still not a verified requested effect.

Its actual twelve-request GPU1 panel then exposed a new concrete blocker: only
two replies matched the required assessment structure; ten were prose. All
seven fixed-point assessments failed the output contract. The adapter appends
a generic visual-description directive and does not constrain this image
route to the assessment schema. One prose answer also mislocated a point, so
format repair cannot be claimed to fix spatial judgment. The detailed failed
result and next bounded schema-constrained repair are recorded in
`computer-use-visual-action055-results.md`. No physical adoption occurred.

The next acceptance order is: combined source gate; actual GPU1 positive,
absent, ambiguous and wrong-point controls; one bounded physical desktop action
with independent outcome verification; the unchanged long task on Qwen and
Ornith; broader operation/matrix coverage and live Egg adoption. Visual
keyboard/text, selection, scrolling and dragging remain separate missing
capabilities. This candidate has not yet been adopted or pushed. Zero verified
whole-task passes remain the current result.

## What is established

The recurring architectural problem is a mismatch between what the controller
requires as evidence and what each layer can actually establish. Some boundaries
reject useful observations; other boundaries previously credited an action's
prerequisite as its successful effect. These are several reproduced faults,
not a proven single explanation for every failed trajectory.

1. **Vision depended on accessibility succeeding first.** In the actual owned
   Save As dialog, twelve complete paired observations had zero ACTIVE
   accessibility roots, although host window ownership and direct parentage
   were positively established. All four old visual entry points required the
   native semantic observation. The visual layer could not inspect this dialog.
   The toolkit's internal reason for the missing ACTIVE root remains unknown.
2. **The new visual path also demands whole-image equality for read-only
   inspection.** Its first actual Qwen trial made three planner requests and
   one Moondream request, then stopped after three inspection failures. The
   same 955-by-690 window, geometry and ownership differed in one pixel:
   RGB (36,36,36) became (37,37,37). Inspection was rejected both before and
   after inference. Zero model-owned input occurred. Exact-image equality is
   being used for both interpreting an immutable captured image and admitting
   an input against the current desktop. That conflation caused this failure.
3. **An earlier write predicate checked focus instead of the requested value.**
   A focused but empty Firefox field passed. The native setter repeatedly
   left it empty; physical typing produced the intended value. The browser's
   internal setter failure remains unknown. Effect contracts now have targeted
   source repairs, but these do not establish whole-task acceptance.
4. **Planning and recovery have lost or misused actionable information.**
   Reproductions found oversized Kev requests, omitted controls, lost immediate
   operations, repeated failed strategies and historical-content identity
   errors. Source repairs and bounded checks exist. Repeated source acquisition
   and failure to author the final outputs remain unresolved even with retained
   source text available to planning.
5. **The actual Moondream reading was wrong.** It described content absent from
   the captured dialog. The saved request contains the correct screenshot;
   model/tokenizer pins and retained worker file hashes match. Source inspection
   established no image-drop or stale-cache defect. Its JPEG95 representation
   also differs between the two saved pixel states. Neither representation
   equality nor the internal cause of the hallucination has been established.

The latest failed action was therefore stopped by controller behavior before
pointing or Kev was exercised. This run cannot establish that those models
would have selected or completed the action correctly. Model adequacy remains
an empirical question after the execution path is usable.

## Ordered work and acceptance

### 1. Separate snapshot perception from current input authority

Register CU.PERCEPTION.054 as a new condition; preserve VISION053's original
failure and criteria. Start from the frozen, independently reviewed 552-file
candidate. Make the smallest change to the read-only path in visual_surface.py:
interpret exact retained image bytes, retain original capture time/window/frame
bindings, and return the model reading as untrusted evidence about that image.
Desktop changes during inference must be recorded rather than erase the reading.
Cancellation, deadlines, invalid identity and protocol errors still apply.

Acceptance: a changed-pixel saved-frame control retains the correctly bound
reading; changed owner, cancellation and forged source controls still reject;
no read operation acquires input authority or task-success credit. Pointer
offers and dispatch retain their existing admission rules during this repair.
This milestone fixes perception availability only.

### 2. Establish useful perception before choosing a visual action

Use the actual preserved screenshot to make bounded, actual-weight comparisons
of the visual layer's reading/grounding and the Omni backbone's interpretation.
Record exact input bytes, model configuration, replies and independently graded
visible targets. Compare task-directed inspection with the current generic
observation request. A direct image route to Omni is a candidate to evaluate,
not an assumed fix; preserve cache_prompt:false and current-media provenance
if that route is implemented. Never let an unsupported caption become a fact.

Acceptance: the chosen configuration identifies visible controls and abstains
on absent/ambiguous controls across positive and negative examples. One saved
dialog result is insufficient to select a production strategy.

### 3. Replace whole-window pixel equality with an evaluated action contract

Design this separately from read-only inspection. Keep task/window ownership,
geometry, source age, cancellation, one-use decisions and uncertain-input
accounting explicit. Evaluate target-specific fresh grounding/revalidation so
an unrelated rendering change does not stall every action. Moving, removed,
occluded, ambiguous or foreign targets must invalidate the proposal. Do not
introduce a hardcoded pixel mask, guessed global similarity threshold or a
blanket freshness bypass. The current local point API supplies centers, not
control bounds; do not invent bounds or claim an unimplemented detector exists.

Acceptance: one bounded live goal reaches actual Moondream grounding, Kev
selection and physical pointer movement/input, and independently observes the
requested outcome. Input acknowledgement alone does not satisfy the goal.

### 4. Complete operations and recovery as one usable loop

Omni owns the objective and immediate operation; the observation layer supplies
source-bound native evidence and images; Moondream grounds visible targets;
Kev chooses among finite valid alternatives or abstains; the host executes;
an independent readback checks the intended effect. Preserve full source values
for planning while keeping Kev's action choices within its evaluated budget.

Add the still-missing visual keyboard/text, selection, drag and scrolling
capabilities. Use operation-specific verification: exact text/value for writes,
selected content for selection, destination/application state for movement and
saved/reopened contents for file operations. A failed or uncertain strategy
cannot be renewed by paraphrasing. Recovery must acquire relevant evidence,
change an executable strategy, or report the concrete missing capability.

### 5. Return to whole-task acceptance and release

After the lower path works, run unchanged LHN-047 on local Qwen and Ornith with
GPU1. Require all ten original criteria and all five deliverables, including
reopened contents, archive, source/documentation use, memory and cleanup. Then
execute the frozen 100 cases/200 local model cells and all 960 mandatory
criteria. Preserve notes and failures per run. Publish verified fixes to main,
observe both Eggs updating, and run actual Ornith tasks on .31 and .41.

Each code repair follows the repository's focused/full gate, one adoption and
one bounded live-window discipline. CPU test totals never substitute for model
actions or whole-task success. No claim of arbitrary-task reliability or parity
with frontier systems is supported yet.

## Durable evidence

- Frozen candidate/gate: runtime-data/gpu1-acceptance/operator-audit/
  execution-sop-20261017/vision053-combined-gate-20261017/receipt.json
  (5,131 tests; source inventory unchanged).
- Actual failed trial and supplemental cleanup: runtime-data/gpu1-acceptance/
  operator-audit/execution-sop-20261017/vision053-host-diagnostic/runs/
  first-frozen-20261017T1309Z/. Original failure is unchanged. All new windows
  subsequently closed; fourteen prior windows and ten workers were preserved.
- Independent failure archive: /home/roko/Documents/qwen-omni-research/
  2026-10-17/vision053-first-frozen-failure-independent-20261017T131048Z/.
- Pixel diagnosis: /home/roko/Documents/qwen-omni-research/2026-10-17/
  vision053-first-trial-image-diagnosis-20261017/.
- Moondream audit: /home/roko/Documents/qwen-omni-research/2026-10-17/
  vision053-md-image-source-audit-20261017/.

No production/main adoption of VISION053 and no new main push occurred. A
separate CPU diagnostic frontend used the frozen candidate against the existing
GPU cohort; GPU workers were not restarted for this trial.
