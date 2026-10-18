# Source-bound immediate action diagnostic

This new panel preserves four actual decision sources from the failed Ornith
LHN-047 run: phases 1, 5, 6 and 7. Each source retains the exact bind/expand
capture and original proposal/action request bytes. No original task, prior
panel, acceptance oracle or source observation is edited.

The failing acceptance criterion is narrow: a downstream action must preserve
its proposal's source-bound immediate operation family and observed selector,
or explicitly return to planning without input. In the actual phase 1, a
proposal to activate Text Editor instead became a pointer action on the active
Workbench window. In phase 7, a proposal to open Input became an action on the
`tmp` breadcrumb. Phases 5 and 6 supply actual matching switches to different
windows as positive controls.

Old proposals did not have typed immediate selectors. Every added
`immediate_action` in the nine host probes is therefore an explicitly
counterfactual reconstruction, using only controls already present in its
exact source. It is never represented as an actual model reply. The source's
free operation prose is interpreted by the independent evaluator for this
small diagnostic; production must not use lexical rules to infer an anchor.
The implementation must require the model's explicit typed selector.

The host probes include matching actions, both actual mismatches, a different
observed inactive window, a different source-offered operation on the same
control, missing anchor, stale source phase and explicit no-input replan.
A fresh valid binding remains permitted after a stale one is rejected. An
alternative strategy may explicitly replan; it cannot silently execute a
different action under the old anchor.

`fixture_api.py` performs only bounded file reads. Load it by file path with
`importlib.util.spec_from_file_location`, then use:

```python
spec = fixtures.load_frozen_panel()
inputs = fixtures.model_inputs("SRC.PHASE01")
probe = fixtures.host_inputs("ANCHOR.HOST.030")
```

`model_inputs` supplies private compiler material and exact baseline requests,
not oracles or reconstructed anchors. Never send its entire private envelope
to the solver. For an actual candidate chain, call the production
`ui_decision.subgoal_context(inputs["compact"], atomic_anchor=True)`, obtain the
model's actual proposal, bind that actual output with `bind_subgoal`, and send
the actual resulting compact context/schema for the downstream decision.
Do not fill a model's proposal from the host probe's golden anchor.

The model matrix has 16 independently retained cells: four sources, Ornith and
Qwen, baseline and atomic-anchor conditions. Every cell starts unrun. Each
full chain has proposal and downstream calls. Preserve native `think:false`,
temperature 0, requested output 8192 and observed state. Baseline system
messages stay exact; candidate system/schema changes must come from the actual
frozen production atomic route and be recorded with source hashes and wire
deltas, never evaluator-written action hints. Record every failure and unrun cell. If a newly sampled baseline
proposal differs, its downstream request must consume that actual output;
do not silently replay the old proposal's downstream request as its result.
Candidate generation can change wording/strategy, so chain outcomes alone do
not isolate which generated text caused an improvement. The fixed host probes
independently test containment.

Report source validity, handoff consistency, explicit replanning, semantic
task alignment and effects separately. Decoder-constrained family/target
consistency cannot demonstrate good strategy, file opening, acquired content,
task progress or whole-task completion. The original 100-task and all-device
acceptance scope remains unchanged.

CU.FACT.045 remains separate and unpatched. The old captured handoff already
labels `current_view` as a planner claim; merely adding another label is not a
repair. The source still contains an observed Input row while the model
repeats “contents not yet visible.” Test actual false-premise resistance
separately from provenance/no-effect-credit properties. Seeing Input does not
prove its entire inventory or any file's readable contents.
