# Current-document negative, version 2

This separately frozen diagnostic has one case and four initially unrun cells:
Ornith and Qwen, each with baseline and source-bound candidate conditions. It
adds no results to the original 44-cell panel and replaces none of its rows.
The original 100-task whole-task acceptance contracts remain unchanged.

`PTR.NEGATIVE.V2.010` uses the actual Text Editor pointer request captured for
`PTR.ARCHIVE.030`. The full 1,810-character task brief is already present in that
request. Its active editor title and exact readable textbox value agree with
controller observation row 12. The raw original request, original packet and
full native snapshot are retained here with manifest digests.

There is exactly one controlled semantic change: the immediate objective is
replaced by the last sentence of the unchanged original user goal:

> Read Input/task-brief.txt for the supplied official source starting URLs,
> declared input inventory and exact delivery/archive requirements.

The same change is serialized into the model request. Global goal, system
policy, full visible context, active window, previous no-input deferral, Open
choices, response schema, sampling, output budget and native `think:false` stay
unchanged. No instruction telling the solver which operation to choose is added.
The private oracle is `replan`: this input-only resolver has nothing to click in
order to read the currently supplied brief. Replan verifies only this finite
choice; it does not demonstrate comprehension or satisfy the entire task.

The native provider reports Document.DocURL evidence unavailable. The case
claims current readable textbox text and observed editor identity, not a
provider-verified resource URL, saved-file identity or reopen result. The
captured pointer text is the existing production projection; it is not the
raw `state.text` serialization, but includes the exact complete textbox value.
The retained `previous_attempt` describes the original abstention under the
earlier objective, not a newly executed failed input strategy.

This fixture was designed after inspecting the version 1 baseline and frozen
before its own first inference. It is a held-out diagnostic input, not an
unbiased unseen benchmark. Version 1 `PTR.NEGATIVE.030` and its incorrect score
remain intact: that earlier fixture inconsistently claimed a complete current
brief while actually providing a file listing, so its mechanism interpretation
is inconclusive. This version does not rewrite that historical result.

The existing offline evaluator supports this separate manifest without source
changes:

```python
from pathlib import Path
from runtime.verify_orchestration_contracts import load_panel, render_request

manifest, cases = load_panel(Path("tests/live/native_orchestration_acceptance_v2"))
case = cases["PTR.NEGATIVE.V2.010"]
unexecuted = render_request(case, "robit/ornith-1.5-omni-audio-bridge:q4km")
```

`render_request` exposes only `case.input.wire_request` to the solver, never the
oracle, source annotations or historical replies. Use `grade_records(cases,
records)` for captured responses and retain all four cells, including unrun
ones. Candidate calls require the actual candidate view and implementation
hashes. Only the root live executor may submit requests or operate the desktop.
This directory contains no inference or desktop runner.

```bash
.venv/bin/python -m runtime.verify_orchestration_contracts \
  --panel tests/live/native_orchestration_acceptance_v2
.venv/bin/python -m pytest -q tests/live/native_orchestration_acceptance_v2
```
