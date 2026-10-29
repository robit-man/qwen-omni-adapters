# 082: Compile observed provenance on the host for direct planning

## Failing criterion and evidence

The original081 Ornith whole task failed all functional criteria. Its first
model reply selected the unique observed Firefox window correctly but supplied
the whole-observation hash as the target fingerprint. The expected per-target
fingerprint was absent from the actual request, captured packet and schema.
The optional full `action_handoff` therefore exposed a host-computed value the
model could not copy from its input. The unchanged validator correctly rejected
that claim before input. The later Open-selector failure and PEG backend error
are separate observations and are not explained by this finding.

Evidence: research root `direct-planner081-fingerprint-diagnosis-20261017`,
report SHA256
`d39d7b399b15dae2d40486f56d756f65a2a1f3d29afb2b52ae45c296711eda2b`.
The original whole-task grade remains1/10, preservation only,0/5 outputs.

## One bounded hypothesis

Direct planning can preserve a model-selected operation, target and immediate
explanation without asking the model to regenerate opaque host provenance.
Project a model-facing semantic contract; expand its validated response into
the unchanged canonical handoff on the host using the captured observation.
The existing compact planner's host compilation is precedent, not evidence
that the direct route will then complete the whole task.

## Implementation boundary

Start a separate source-only candidate from exact081/077 source5829c44.
Retain all failed replies and080 source unchanged. Run the full baseline before
edits. Reuse existing venv/vendor links with explicit private imports; copy no
weights or runtime payloads. Prefer changes limited to `portal/ui_decision.py`
and its direct-route caller in `portal/app.py`, plus focused tests.

Keep the canonical working envelope, `validate_action_handoff`, native
executor, effect contracts, policy, verified prefix and recovery guards intact.
Expose a direct model schema without host-owned handoff/transition metadata.
The model continues choosing its phase, operation and referent and supplies
the exact immediate-operation/prerequisite explanation through a small typed
semantic intent. Objective remains supplied once through the existing phase
or commitment path. Preserve the explanation verbatim; never replace it with
inferred keyword intent or claim it as observed evidence.

Compile source/session/scope/phase/goal/target hashes from the captured packet
after the semantic reply passes its own schema. Reuse existing handoff
construction/validation where possible. Bind only the exact selected positive
referent; ambiguity, missing targets or unsupported operations must use the
existing observation/discovery/replan boundary rather than silently selecting
another target. Preserve historical predecessor and freshness requirements for
recovery; never renew an old claim from a fresh observation.

Do not repair a model-supplied invalid legacy receipt in place. New model
output cannot supply host-only fields. The original invalid081 reply stays
invalid. The host-compiled claim remains non-authorizing; all normal fresh
input guards and independent effect verification still run. This patch does
not adopt080 visual intent or change model/settings/budgets/parser behavior.

## Acceptance order

1. Freeze external acceptance before editing. Cover positive unique binding,
   exact semantic continuity and negative stale epoch/goal/source/target,
   ambiguity, missing referent, tampered semantic reply, injected host fields,
   recovery predecessor and protected input cases. Validate existing compact
   behavior and canonical admission remain unchanged.
2. Focused tests, independent review and the required full regression gate
   against exact private source. Historical-response transformations used for
   CPU counterfactuals must be explicitly derived and never called actual model
   output or actual input execution.
3. Preregister one bounded actual-weight comparison on frozen source cases,
   retaining original task/settings. Measure semantic target/operation quality
   separately from host provenance correctness. No GUI/input credit from this
   generation-only comparison; no response injection into a live task.
4. If that named criterion passes, run one unchanged original whole task with
   all ten criteria/five outputs. Its success remains required separately.

The current whole task, both models/devices, complete desktop breadth and later
speech loop remain owed. A source or provenance milestone cannot close them.
