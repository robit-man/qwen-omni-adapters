# Computer-use research review: decisions, recovery and evaluation

Reviewed 2026-10-17 UTC against runtime revision `8e845a9`. This extends the
architecture guide and the earlier Agent S3 / UI-TARS-2 / LocalLSTC / CoAct-1
review. It does not establish arbitrary-task reliability or SOTA parity.

## Sources and implementation findings

Sources were acquired before analysis. Both research corpora (108 files,
58,162,922 bytes) are durably archived outside runtime cleanup at
`/home/roko/Documents/qwen-omni-research/2026-10-17/`. Every archived file was
read back and SHA-256 verified; all 30 broader-review acquisition hashes also
matched their original capture records. The tracked
[archive manifest](research/computer-use-2026-10-17.manifest.json) identifies
every file, size and hash. Copies of the manifest exist in the archive and
repository. The archive includes papers and extracted text, pinned source
snapshots, repository metadata, acquisition URLs, earlier induction records
and the earlier report. The original runtime-data copies remain available.
The code review covers the named control-loop components, not every repository
file. Paper inspection covers methodology, evaluation, failure analysis and
limitations; this is a focused engineering review, not full paper induction.

| Source | Pinned reference | Finding and limit |
| --- | --- | --- |
| Jev-Ultrafast | [1231850](https://github.com/browser-use/jev-ultrafast/tree/1231850a0bf1a0c0341fe408ef1668dbbfdfac46) | `model.py:48–148` derives compatible operation/target banks from observed controls. Speculative heads share state; only the selected head resolves an action. `agent.py` consumes decisions before input, records execution before post-observation, and refreshes stale decisions. Its short browser demonstrations do not establish long-horizon native desktop performance. |
| Open Computer Use | [02e875a](https://github.com/max1874/open-computer-use/tree/02e875ac614492c6ca3c5104bd3aa8a9911b6a79) | `agent.py:340–478` separates refusals from uncertain/already-executed input, preserves failed attempts and suppresses refused operation/target pairs. `axbridge.swift:1196–1280` checks text readback and distinguishes unverified writes. Its unconditional erase, implicit setter fallback and label substring identity checks are unsuitable imports here. Native macOS behavior is not Linux evidence. |
| CUA-JEV | [f22e41e](https://github.com/ZJU-REAL/CUA-JEV/tree/f22e41e24f8eaca09888862aa115c0d28ee7eb91) | Typed candidates carry intent, route and preconditions; runtime records decision, commitment, guard, receipt and verification separately. Published open-task examples still make a planner call per action and use DOM/API/CLI routes. `verify.py` defaults to `receipt.success`; that is insufficient for our ineffective-setter case. Its scope is explicitly bounded. |
| Kev | [1f04cfa](https://github.com/jaredpalmer/kev/blob/1f04cfa28e6a708485b28a9585010b9e436b776e/docs/model-cards/kev-4b.md) | The model card describes document/decision training and recommends thresholds frozen on the user's workload. Its calibration data does not establish native GUI decision calibration. This current card describes a different published revision from our pinned `6cfce5c2` runtime; do not transfer its numbers to our weights. |
| OpenJev | [da6a68f](https://github.com/alanhuangyoo/OpenJev/tree/da6a68fb9195958cc0a6cdb7b35587acec436a65) | The authors train on the exact typed browser interface and audit terminal labels. Their README distinguishes offline step accuracy from live-task results; the latter use a final-page LLM judge. Candidate-limited evaluation, test-selection exceptions and browser-only training constrain generalization. This motivates workload evaluation, not an untested model replacement. Paper acquisition was not completed; these are repository claims and code observations. |
| OSWorld 2.0 / 2.1 | [paper v2](https://arxiv.org/abs/2606.29537v2), [code d995367](https://github.com/xlang-ai/OSWorld-V2/tree/d9953673daf55381237e058fd9388cc2d74c5538) | Evaluation separates strict completion from partial progress. Failure analysis identifies missing constraints, observation/action lag, skipped corrections and lost early evidence. Interdependent workflows and independently checked final state are more informative than concatenated primitives. Release 2.1 pins task/assets/environment versions; task classes are gated. Our suite is not an official OSWorld score. |

The exact code supports strong statements about implementation. Benchmark
numbers remain author-reported and have not been reproduced on this stack.
The earlier review also found Agent S3 uses a flat policy: research does not
unanimously prescribe our guide's hierarchy.

## What changes in our engineering approach

1. **Test the decision interface before trusting delegation.** Measure whether
   the correct action is offered, whether the policy selects it, and whether
   the executor delivers it as separate outcomes. Evaluate the pinned Kev model
   on actual native task states, paraphrases, distractors, missing evidence and
   changed layouts. Preserve probabilities, abstentions and latency. Keep the
   current threshold until a held-out native evaluation supports a change.
2. **Make useful choices, not a larger verb list.** Add typed effect-oriented
   families, starting with text replacement/insertion. Planner supplies intent
   and content; the controller derives compatible strategies and mandatory
   postconditions. Kev arbitrates genuine alternatives. Deterministic verified
   transitions need no policy call. A primitive's advertised capability does
   not establish its desired semantic outcome.
3. **Recovery must change execution.** A rejected pre-input decision may be
   discarded and re-resolved under fresh state. An attempted write needs
   reconciliation. Failed strategies stay in the ledger and affect subsequent
   candidates; another narrative assessment alone is insufficient.
4. **Compression must retain access to facts.** Keep original constraints,
   current subgoal, resource identity and unresolved transitions explicitly.
   Add task-bound expansion of retained observations when omitted details are
   needed. A hash with no retrieval operation is not expandable context.
5. **Separate artifact quality from visible activity.** Grade original
   requirements independently: correct documents, exact data, archive contents,
   reopened files and memory provenance. Record partial milestones separately.
   Preserve native-only execution for this acceptance lane; other projects'
   API/CLI successes cannot substitute for visible desktop input.

These are engineering deductions to test, not guarantees supplied by papers.
Do not add all capabilities at once or lower guards to improve a score.

### Local decision-interface experiment

The first live text-entry experiment offered two equivalent transports for the
same Firefox address field. Pinned Kev assigned native setter 0.486, physical
replacement 0.3425 and abstain 0.1715; the unchanged 0.95 threshold admitted no
input. This is one measured example of a poorly factored decision, not a
calibration study. The repaired interface assigns transport selection to the
executor using capabilities and execution history, and leaves genuinely
different control choices to Kev. A unique target therefore needs no model
call. Subsequent real Firefox execution verified physical focus, selection,
replacement and navigation. It is provider evidence, not a backbone-model
or whole-task pass. Detailed failures and receipts are retained in
`runtime-data/gpu1-acceptance/operator-audit/text-entry-cycle-20261017/`.

The next live task exposed an information-boundary defect: optional native text
coverage was dropped by the bounded planner projection, incorrectly hiding the
entire text-entry family. Capability-based discovery repaired that omission;
fresh executor coverage remains required. The family was then offered in three
real Ornith planner phases, but none selected it. All replies generated a
generic commitment objective before phase selection, which excludes phase-owned
objectives in that grammar. Native decoder checks confirmed the exclusion.
Reordering generation to assessment → phase → commitment removed it, but the
next live task still chose primitive workflows in all three phases and failed.
In particular, Ctrl+L opened a location field while the planner checked window
focus. The next experiment evaluates compact delegated action interfaces with
the loaded model on these captured states before another full desktop rerun.
No task-completion improvement is attributed to ordering alone.

### Input transport reference

The [pinned xdotool release](https://github.com/jordansissel/xdotool/tree/f11c44edd4d426d69bcff479b0d643a4da8bc99b)
temporarily maps unavailable key symbols and removes the scratch mapping after
key dispatch. That motivates investigating keymap/event timing for the Egg .31
Unicode loss; it does not prove the cause of the deployed failure. The upstream
source is archived separately with a [three-file manifest](research/xdotool-unicode-2026-10-17.manifest.json).
Distro patches and application event delivery remain unverified. A fresh
isolated nine-case input check passed on .31; one passing recheck does not
establish that the earlier intermittent loss is fixed. No delay change or test
skip was introduced.

The subsequent bounded repair reserves distinct unused keycodes through actual
recipient readback. Private GTK checks passed, but one ordinary Firefox attempt
reordered mixed Unicode/ASCII input. A segmented diagnostic preserved each prefix
through accented and CJK text. The candidate now acknowledges each prefix before
the next segment; it stops, restores its mappings and reports uncertainty instead
of replaying partial input. This establishes a useful delivery intervention, not
the precise internal reason for reordering. The installed IBus 1.5.29 source and
[Mozilla's asynchronous composition report](https://bugzilla.mozilla.org/show_bug.cgi?id=1810406)
support investigating asynchronous input-method processing.

An emoji probe exposed a separate representation issue: Firefox reported an
extra U+FEFF and two native positions. Mozilla's
[ATK conversion change](https://bugzilla.mozilla.org/show_bug.cgi?id=1346535)
explicitly uses this convention to align offsets. Generic stripping would corrupt
legitimate content and selection positions; conversion remains a separate open
criterion. Seven source/induction artifacts (473,507 bytes) were saved, fsynced and
hash-verified in the [native input-method archive](research/native-input-methods-2026-10-17.manifest.json).

## Mapping to observed failures and bounded next work

| Local evidence | Current status | Next acceptance criterion |
| --- | --- | --- |
| Firefox setter acknowledged but field remained empty; focus-only predicate passed | Mandatory value effect fixed in `44c5643`; real provider diagnostic passed | Typed replacement must reconcile a failed setter and explicitly choose a verified physical strategy without duplicated text |
| Hidden Firefox nodes changed while the address field stayed identical; Return rejected before input | Native workflow refresh implemented and checked in real Firefox with a controlled page update | Preserve one fresh re-enumeration and strict target/scope/focus/dispatch checks; separately bound primitive-choice boundaries need their own decision protocol |
| Original LHN-001 narrated double-click but emitted scroll; four planner phases produced one window switch | Full task failed; no Kev primitive decision in that run | Typed intent families must produce applicable choices and receipts for actual layer calls, with original artifact checks unchanged |
| Native bridge rejected the owned editor although all 406 passive guard samples allowed it | Exact read failure unknown; no open-file dispatch proved | Capture the failing ownership/read predicate and fix that cause; no scope relaxation based on a guessed dialog |
| Historical row hashes cannot be dereferenced by native planner | Context expansion still missing | Bounded read-only retrieval returns exact task-owned historical evidence, without presenting it as current UI |
| Egg .31 lost a Unicode character in real XTEST input | Deployment gate fails; previous core remains running | Reproduce transport/readback cause and pass exact Unicode/multiline input on ARM before another deployment |

The first code cycle from this review addressed pre-input freshness. All 4,027
repository tests passed. A real Firefox page update invalidated the initial
Return offer; a fresh decision dispatched once and verified the destination.
Three earlier probe setup failures remain recorded. This provider diagnostic
used no model calls and earns no whole-task credit; it does not close delegation,
native-scope or ARM faults. See the [implementation evidence](computer-use-reliability-evidence.json).

### Loaded-model decision-interface comparison, 2026-10-17

Nine actual Ornith requests used three captured native desktop states on the
already leased GPU 1. They dispatched no input and earn no whole-task credit.
Removing primitive workflows alone did not repair decisions: the three reduced
replies still selected pointer boundaries, including an attempt to write a URL
by activating its field without text. A compact control/family projection then
selected the correct text-entry family and exact URL in Firefox; the other two
replies switched windows, including one switch to the already active window.
This is narrow evidence for simplifying the interface, not proof of general
planning improvement. The compact calls took 1.787, 1.832 and 2.515 seconds;
corresponding broad-interface calls took 12.865, 16.686 and 19.166 seconds.

The broad Firefox schema occupied about 56,000 characters versus 2,400 in the
compact comparison. Full goals and historical result/ledger evidence remained
provided. A new comparison epoch replaced working-state data omitted by the
original capture; this is declared in the saved notes. One comparison-helper
error (empty definitions after pruning) occurred before inference; the remaining
calls resumed without replaying the completed request. Every request, response,
source digest and per-comparison note remains in
`runtime-data/gpu1-acceptance/operator-audit/decision-interface-cycle-20261017/`.

The immediate contract repair separates keyboard/scroll dispatch from claims
about its effect. Real prior Ctrl+L input created a location field, while the
planner checked focus on the old window and reported failure. The private
input-and-observe path retains unknown consequences until observed and grants no
completion credit. The compact planner was subsequently implemented as an opt-in
projection with exact snapshot-bound expansion. Its actual desktop trial
verified physical URL replacement but failed the full task. The subsequent
`text_submit` sequence proved real navigation with independent Document.DocURL
readback. Its failed continuation exposed a compression defect: a 55,454-byte
grounded schema exceeded the 56 KiB limit after adding working-state fields,
silently replacing all desktop actions with non-UI choices. Compact projection
then inherited that loss despite its much smaller actual decoder schema.

The same run captured seven repeated Moondream inspections of readable Python
documentation, returning truncated invented text. The exact reader PNG and all
requests, responses and controller snapshots are [archived durably](research/computer-use-text-submit-2026-10-17.manifest.json).
The planner's reading objective never reached the visual reader; its current
endpoint uses a fixed generic prompt. That is a proven information loss, while
the model's precise reading failure remains unproven. The controller also resets
its failure counter on each accepted inspection, regardless of repeated evidence.
These are separate repair criteria; neither navigation nor protocol success is
whole-task completion.

### Ornith sampling and planning comparison, 2026-10-17

The [upstream Ornith 1.5 9B model card](https://huggingface.co/ornith-ai/Ornith-1.5-9B/blob/489cb97981b8654bcfcf30ce1f94ed1b62e07b53/README.md) recommends sampled decoding; the live native runner had forced greedy decoding. The installed adapter also drops top-k, min-p and penalty values placed inside Ollama `options` when translating to its OpenAI backend. These are concrete configuration/transport differences, not proof of the task failure’s sole cause.

Actual GPU 1 comparisons preserved the same captured state and original goal. Generation order alone did not fix selection. Reasoning sometimes exhausted the effective 4,096-token output cap. With upstream general-task sampler values confirmed in native slot parameters, one of four samples selected historical recall; the others repeated a switch, inspected an empty editor, or produced no final decision. An independent subgoal-planning probe identified reading the supplied brief as the prerequisite and Files as the next operation. This supports a bounded handoff experiment, not a broad performance claim. All requests, responses, failed attempts and pinned upstream sources are in the [106-file durable archive](research/computer-use-observation-recall-2026-10-17.manifest.json).


### Kev uncertainty escalation, 2026-10-17

The [author’s Kev model card](https://huggingface.co/jaredpalmer/kev-4b) describes a typed decision model with confidence-based deferral and workload-specific threshold evaluation. That supports separating fast classification from slower recovery; it does not prove a particular desktop threshold or primitive policy works. The fetched checkpoint differs from the loaded pin, so its calibration is not silently transferred. Pinned source files, revisions, checksums and the actual loaded-policy comparisons are [saved durably](research/computer-use-kev-escalation-2026-10-17.manifest.json).

In the unchanged local task, the prerequisite-first planner selected the Input folder, but Kev abstained three times. Reconstructing the exact request reproduced the result; adding visible context changed probabilities without clearing the existing threshold. The controller also routed explicit planner pointer preferences back through Kev. The next repair therefore supplies a private one-use escalation after a valid abstention: the slow planner chooses among the same eligible operations, while native target freshness, scope and input guards remain authoritative. Dispatch returns an unknown effect and earns no task-completion credit. This is an engineering inference from the measured dead end, subsequently supported by two actual native double-click recoveries. The unchanged whole task still failed when the controller counted deferred decisions as consecutive failures; see the [live archive](research/computer-use-pointer-recovery-2026-10-17.manifest.json).


### Local primitive resolution comparison, 2026-10-17

Twenty-six actual GPU 1 Ornith requests compared captured failure states. Explicit
window-membership flags and extra primitive descriptions in the full planner did
not establish improvement. A narrow finite choice with named operation labels
retained a folder double-click and chose single clicks for Open controls. An
unnecessary window click and overstated predicted effects remained; predictions
therefore grant no verification credit. This supports a bounded context-reduction
experiment, not general desktop competence. Captured JSON ordering limits and
all failed variants are [preserved in the durable record](research/computer-use-control-interpretation-2026-10-17.manifest.json).

## Policy timing and live pointer evidence

The [deadline repair record](research/computer-use-policy-deadline-2026-10-17.manifest.json)
preserves 77 more files, including the real HTTP regression, exact model
requests/responses, native states and nine read-only post-tool screenshots. The
client now waits beyond the backend compute watchdog. The full task still
failed: three valid Kev deferrals reached slow-model resolution, two actual
clicks occurred, and no deliverables were created. The failed Main menu move
was independently observed at its intended coordinates; its pointer-window
binding and the later active-window loss need separate diagnosis. This evidence
does not establish the cause of the earlier transient policy slowdown.

## Popup grabs and window switching

The [GTK source archive](research/computer-use-gtk-popup-2026-10-17.manifest.json)
contains official Popover documentation and the GTK 4.14.5 implementation,
matching this host's installed library series. GTK describes default popover
input grabs and Escape dismissal. In the native reproduction, activating Files
while the editor Open popover remained visible left all ten reads without an
active accessibility frame; only 29 nodes were visited. X11 active-window and
keyboard-focus queries followed Files and did not diagnose the grab.

Reactivating the owned editor, issuing a guarded Escape, and switching to Files
changed three failed reads to three successful reads. A first guard using visible
flags and parent pointers falsely classified retained hidden popups as open.
Requiring current accessible parent-child membership corrected that false
positive. This is a verified provider mechanism, not whole-task or model credit;
the source and live experiments remain separate from long-horizon acceptance.
The [262-file experiment archive](research/computer-use-popup-prerequisite-2026-10-17.manifest.json)
retains the rejected candidate, corrected provider check and unchanged Ornith
task failure. The latter did not open a popup, so it supplies no end-to-end
exercise of the new guard. All archive bytes were flushed and hashed after
readback outside runtime cleanup.

### Auxiliary-model accounting correction

The separate native pointing HTTP recorder contradicts earlier zero-Moondream
counts for cycles 6h, 6i, 6k and 6l. It records respectively 2, 2, 1 and 1
successful Moondream3.1 responses, with exact request-image and response hashes.
The [capture and correction archive](research/computer-use-pointing-accounting-2026-10-17.manifest.json)
preserves those HTTP records and PNG bytes. The structured evidence index is
corrected; original archived notes remain intact. This establishes actual layer
calls and does not change any failed whole-task outcome.

### Planning continuity and broader live evaluation

The [continuity and breadth archive](research/computer-use-planning-continuity-2026-10-17.manifest.json)
contains two separate local task families, a captured-state comparison, one
candidate run and an Egg .41 run. The subgoal projection omitted the previous
plan and unmet constraints that the action selector received. Adding that exact
metadata changed two actual Ornith replies from rereading an already observed
brief to acquiring the unread authority ledger; a third Firefox state did not
improve. These are correlated diagnostic states, not a general performance
estimate. The candidate's whole live task failed earlier and never exercised
ledger selection. None of the four whole-task attempts passed.

All exact model request bodies and hashes are retained. The candidate's auxiliary
parsed-packet recorder omits the new `retained_plan` field and reports that
omission; the authoritative wire bodies include it. Egg .41's run used actual
GPU-resident Ornith with Moondream2 and no Kev endpoint. Its owned-foreground
monitor became unable to map the active window, after three input dispatches.
The run drained, closed its owned windows and restored the prior harness. Egg
.31 kept its earlier deployment after the actual accented-text regression failed
again; that known transport defect remains unrepaired by this continuity change.
