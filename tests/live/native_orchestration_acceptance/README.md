# Frozen orchestration acceptance panel, revision 1

This panel was frozen from saved real requests before candidate model calls.
It diagnoses decision contracts. It is **not** a substitute for the 100 frozen
native long-horizon tasks and cannot award a whole-task pass.

The six archive cases retain exact model-visible wire requests and SHA-256
bindings to the original source JSONL row. The local archived cohort used the
compact planner opt-in, Ornith, calibrated Kev4b and MD3.1 when actually invoked.
The separately captured Egg41 bad workflow used the full planner, Ornith, MD2
on demand and no Kev. Egg31 setup failed before inference; that is a host
prerequisite failure, not an Ornith inference failure. These route conditions
must not be silently combined.

| Cases | Frozen question | Oracle and limitation |
| --- | --- | --- |
| PTR.ARCHIVE.010–020 | Open the observed Input folder / task brief | Double-click the named file-manager item. Choice credit only. The actual .010 reply incorrectly predicts draft-1 opens; that prose is not verified. |
| PTR.ARCHIVE.030–040 | Open is a prerequisite to reading corrections.json | One click on observed Open is a permissible next exploration, not proof of a chooser, source acquisition or read content. The two archived objectives differ, so they are not a causal pair by themselves. |
| PTR.ARCHIVE.050–060 | Composite Open/recent-documents control after prior attempts | Review-only. No uniquely proven safe target/operation is declared. Both remain in the denominator and cannot inflate a finite-choice pass. |
| PTR.NEGATIVE.010 | The same offered control is Remove | Replan; deleting/removing a recent entry does not acquire the requested resource. |
| PTR.NEGATIVE.020 | The offered control is unidentified | Replan; missing target semantics are uncertainty. |
| PTR.NEGATIVE.030 | The current requested complete source is already visible | Replan for this input-only resolver; reading current evidence does not require opening the file again. |
| PTR.NEGATIVE.040 | Direct selection requests corrections.json, but only Remove is offered | Replan; no matching named item or proven prerequisite is offered. |
| PTR.NEGATIVE.050 | All offered strategies failed against unchanged relevant state | Replan for discovery or another strategy. Rewording does not establish a new operation or changed state. |

Negative cases are explicitly labelled hypothetical derivatives of actual saved
observations. They are not represented as additional actual desktop runs.
Every declared change appears in `source.controlled_changes`. Oracles, historical
replies and evaluator metadata must never be sent to the solver.

`egg41-bad-workflow.json` retains the exact failed workflow and first native
before/after result. Independent contract tests replay its nonsensical literal
Ctrl+L expectation, verify that a currently focused location field cannot satisfy
a write, and distinguish focusing/opening from acquiring readable source content.
These are compiler tests against historical state, not input or task results.

## Offline commands and capture contract

```bash
.venv/bin/python -m runtime.verify_orchestration_contracts
.venv/bin/python -m pytest -q tests/live/native_orchestration_acceptance
.venv/bin/python -m runtime.verify_orchestration_contracts \
  --render-dir /tmp/ornith-orchestration-baseline-requests \
  --model robit/ornith-1.5-omni-audio-bridge:q4km
```

Rendering makes zero network/model/desktop calls. The root live executor alone
submits requests, after checking healthy idle workers and exact GPU1 scope.
`render_request(case, model, condition='candidate', replacement_view=...,
replacement_system=..., candidate_sources={path: sha256})` accepts a view produced
by the actual candidate implementation and binds source hashes. It preserves
the archived sampler, native `think:false`, text-only request and output budget.
Baseline rendering cannot change the original request except the declared model
identity for the cross-model comparison.

Capture raw HTTP request/response bytes, request/response timestamps and digests,
runtime source adoption, actual model identity, effective output/context caps,
worker PID/start identities and GPU residency. The offline renderer's declaration
does not prove any model ran. Keep every unsuccessful or timed-out request.

For scoring, save a JSON list of records with `case_id`, `model`, `condition`
(`baseline` or `candidate`), `rendered_request_path`, and `response_path`. Response
files contain either the parsed finite decision or the actual adapter response
with `message.content`. Then run:

```bash
.venv/bin/python -m runtime.verify_orchestration_contracts \
  --records /absolute/path/records.json --output /absolute/path/choice-grades.json
```

The scorer verifies actual saved files, rejects duplicate/unknown cells and
changed baseline settings, and retains missing cells as unrun. The complete
two-model/two-condition panel has 44 cells: 36 finite-choice cells and eight
review-only cells. It does not infer GPU residency, prediction correctness,
source acquisition or desktop task success from a parseable reply.

Perform handoff-only and hierarchy-only comparisons separately. A combined
candidate can show overall improvement but cannot identify which change caused
it. Hold the loaded weights, sampler, observed state and unrelated prompt text
constant inside each comparison. Do not optimize against this small panel and
claim arbitrary-task reliability; preserve broad paraphrase/ambiguity and
whole-task acceptance.

## Whole-task acceptance remains unchanged

After mechanisms pass, run the original LHN-047 goal with each local Omni model
on GPU 1, serially, with independent desktop resets. Keep the original ten
criteria and source/fixture hashes. No partial artifact or model completion
claim can satisfy them:

1. C01: Full saved draft bytes equal the ten declared edits; every protected and
   similar non-target span stays unchanged. Existing exact-edit oracle.
2. C02: Exactly ten unique CSV change entries identify each declared draft and
   exact old/new spans, with no missing, duplicated or invented change.
3. C03: Two explanations accurately reflect the frozen official source sections,
   while authorizing no edits beyond the ten supplied corrections.
4. C04: Fresh native browser address/resource identity and visible relevant
   section evidence show actual GUI acquisition of both official sources.
5. C05: All five requested files exist at their exact paths, parse correctly and
   contain the verified intended content. Existence alone is insufficient.
6. C06: Native apps reopen all required edited copies and inspect corrected and
   protected content, with current UI evidence agreeing with independent disk.
7. C07: Actual guarded keyboard typing and required physical mouse control occur,
   with no shell, macro, direct filesystem/fetch or private browser solver path.
8. C08: Initial input hashes and unrelated state remain intact; only proven owned
   windows close, harness state returns and the inference lane reaches idle.
9. C09: ZIP file members and bytes exactly match the four specified output files;
   no duplicate, extra file or wrapper directory. Existing archive oracle.
10. C10: A fresh same-cookie recall obtains verified paths, exact source URLs and
    the source-bound finding through actual memory write/search receipts, rather
    than echoing conversation or re-reading the GUI/files.

C01 and C09 currently have implemented automatic artifact oracles. The remaining
criteria require independent case review bound to the final evidence bundle;
they are not automatically passed by the runner. Method evidence, semantic/source
review, reopen and memory are mandatory. Failed/blocked attempts remain recorded.
After LHN-047, execute breadth across all 16 families and both local models; close
the 14 media/event materializer gaps and the remaining independent oracle gaps.
An easier diagnostic does not replace the 100-task/200-local-cell target. Remote
Ornith acceptance remains a separate .31/.41 requirement after host prerequisites.
