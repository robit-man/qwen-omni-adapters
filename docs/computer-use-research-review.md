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
completion credit. A compact planner remains a separate implementation cycle.
