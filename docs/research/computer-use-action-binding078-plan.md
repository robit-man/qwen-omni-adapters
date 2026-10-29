# 078: Test action binding at the layer that selects the concrete operation

Status: proposed separate causal experiment; no method, source change or model
call yet.077 is frozen as a failed applicability experiment.

## Evidence and bounded hypothesis

077's two actual navigation expansions correctly rejected inconsistent fixed
targets. In one rejection the later planner described a candidate navigation
primitive, but the early family/target lock prohibited selecting it. The same
weights therefore showed useful semantic judgment in that context while still
failing the registered actionable result. The cause of the first-stage wrong
choice remains unproven.

Test whether assigning exact action selection to the second planner yields
useful supported operations for the identical retained semantic plans. Do not
regenerate those plans, change task goals or inject preferred commands. This
directly tests delegation at the operation boundary rather than another name,
mask or prompt-wording adjustment.

## Source and experimental conditions

Use unchanged gated582-source077 inventory
`9c44a3f40d695929db02b5974c7f769a5f34c0d60c4ad3eea433aa0f8a06c102`.
No application patch is needed to test its existing two helper modes. Revalidate
all source bytes against the passed5,240-test gate before the experiment.

Use all five actual077 first-stage semantic plans exactly once per condition,
with their same captured source packets, original goal, history, current UI,
model profile and samplers. The independent oracles remain unchanged and never
enter the request constructor.

- Baseline: existing `bind_subgoal(..., atomic_anchor=True)`, original full
  proposal and production atomic handoff. This is a fresh actual second-stage
  call for each of the five plans, not reuse of a historical response.
- Candidate: remove only `immediate_action` from the retained proposal, use the
  existing `bind_subgoal(..., atomic_anchor=False)` and its existing
  `SUBGOAL_HANDOFF`. Keep the objective copied exactly once. The second planner
  selects from the original finite source-backed action schema, then the
  existing host binding and canonical validator compile its proposal.

The candidate is explicitly a counterfactual helper protocol, not the current
production `create_app` request. Freeze every necessary handoff/schema difference
and prove all unrelated wire bytes and source facts unchanged. Do not rewrite
source receipts, hidden hashes, state or semantic claims to manufacture parity.
The early planner family/target agreement is intentionally absent in this
condition. This is a coherent schema/claim/policy comparison, not a single-field
ablation or a claim that every planner guard is identical.

## Gates, actual comparison and next work

Freeze independent acceptance and method before implementation or calls. Verify
objective constancy, full immediate-operation/prerequisite continuity, unique
source target/finite-family binding, canonical validation and no fresh input
authority. Candidate failures and abstentions must remain failures. No native,
Kev, Moondream, confidence, ownership, freshness or effect guard is relaxed.

Preregister one serialized A/B comparison: five fresh baseline and five fresh
candidate second-stage calls, max10 POSTs,60 seconds each,360 seconds aggregate,
no retry or resampling. Preserve each first response and raw transport/token/GPU
join. Require useful supported concrete operations on both navigation cases and
all three positive controls; source/JSON success or replan alone cannot pass.
There is no GUI, effect, current-native-freshness or whole-task credit.

If it fails, preserve the negative result and stop this binding hypothesis.
If it passes, treat it as evidence for a separately gated application-route
change, then verify genuinely generated plans and one unchanged whole task.
Collect actual local Ornith evidence on GPU1 through the existing graceful
profile handoff before further speculative Qwen presentation changes. Both
backbones, both Eggs, all original long-horizon criteria and the breadth matrix
remain the goal; a five-plan comparison cannot satisfy them.
