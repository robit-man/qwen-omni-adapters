# Native long-horizon task runner

`runtime.verify_long_horizon_native` packages the 100 original task specifications
in `tests/live/native_long_horizon`. The 200 Qwen/Ornith first-attempt cells are an
intended evaluation matrix. The specification manifest's zero counters are
frozen design metadata, not a live run ledger. Retained Qwen LHN-047 attempts
and their source cohorts are recorded in
[the evaluation report](computer-use-long-horizon-evaluation.md); none has
passed the full task. Other cells remain pending. CPU tests of this harness do
not execute those cells.

The runner submits one outcome-only text goal to the existing authenticated
`/api/chat/stream` endpoint with the `native_task` profile. The server compiles
bounded phases and the existing controller owns native selection, dispatch and
verification. The solver tool scope is `native_ui`, `memory_write` and
`memory_search`. There is no audio, TTS/ASR, shell, background executor, web-fetch
tool, CDP browser or private browser profile. The exact goal changes only by
substituting the newly owned absolute `WORKBENCH` path. Sealed evaluator data,
expected answers and an action plan are never added to the request.

Every case materializes a read-only `Input/task-brief.txt` containing its public
source starting URLs/authority/topic labels, input inventory, required output
paths and prefrozen family-specific archive conditions. These are supplied task
facts, not a solution. The reconciliation tasks publish the authority rule and
path convention while keeping the evaluator's computed edition choices private.
All 75 ZIP cases specify their exact member paths or selection rule before an
attempt; no member list is invented after observing the model's choices. Archive
files never include themselves. Public releases exclude internal files, and the
capstone's separately edited archive copy is not a member of its earlier ZIP.

The user has already authorized the evaluation and owned setup. Technical
readiness checks are automatic checks of evidence, **not another approval
step**. There is no interactive approval or `ready:true` override. Pending
semantic grading does not block a technically valid attempt.

## Setup and invocation

Use a normally deployed portal and its exact loaded model. This runner never
starts or restarts the runtime or acquires a GPU. It checks authenticated health,
natural queue idleness and the `/api/status` model before opening apps. The
endpoint must be loopback HTTP without embedded credentials or capability URLs.

An offline preparation creates only declared `Input/` bytes in a new directory:

```bash
.venv/bin/python -m runtime.verify_long_horizon_native \
  --case LHN-047 \
  --output runtime-data/long-native/LHN-047-preparation-001 \
  --prepare-only
```

The preparation records exact missing prerequisites. Exit 2 means that at least
one remains missing, even when input materialization succeeded. Each live run
uses a **different, new output directory**. Existing directories are never
resumed or overwritten.

For a source-frozen case with valid host evidence:

```bash
.venv/bin/python -m runtime.verify_long_horizon_native \
  --case LHN-047 \
  --model robit/ornith-1.5-omni-audio-bridge:q4km \
  --endpoint http://127.0.0.1:8920 \
  --token-file runtime-data/state/access-token.txt \
  --readiness-file /absolute/host-evidence/LHN-047/readiness.json \
  --output runtime-data/long-native/LHN-047-ornith-001
```

`--think` is an explicit native boolean, default false. `--browser chromium`
selects an ordinary installed Chromium instead of Firefox. The launcher never
adds a remote-debugging flag, custom profile or headless mode. Token content is
loaded only into the HTTP client; it is not copied to the evidence bundle.

The client action deadline is at most 1,800 seconds. Requested sampling is
temperature 0 and `num_predict:8192`. Server phase/deadline limits remain those of
the deployed profile; this runner does not override them. At the reviewed base,
the profile allows at most 40 phases and uses the smaller of the portal timeout
and 1,800 seconds. A proposed 500-input ceiling is **not implemented** and is
recorded as such. Changing any budget defines a separate cohort.

## Source and application prerequisites

`readiness-requirements.json` identifies the source receipts needed for the
selected case. Predeclared nonempty assertions, expected-value maps and evaluator
descriptions satisfy rubric readiness. A criterion can require later semantic
review without an already implemented automatic scorer.

The host source pack must have this shape, bound to the exact packaged task
signature and current checkout revision:

```json
{
  "schema": "robit.omni.long-native-readiness.v1",
  "case_id": "LHN-047",
  "task_signature_sha256": "<exact packaged task signature>",
  "source_revision": "<exact checkout revision>",
  "receipts": [{
    "id": "source:govuk",
    "path": "govuk-proof.json",
    "sha256": "<SHA256 of proof bytes>",
    "authority": "host_independent_evaluator"
  }]
}
```

Each receipt points to a bounded, owner-owned regular JSON file beside or below
the pack, outside the solver Workbench. It is not accepted merely because it
contains a successful status. Its schema is
`robit.omni.long-native-prerequisite-proof.v1`, with the same `case_id`,
`task_signature_sha256`, `source_revision`, `receipt_id`, authority, and
`status:"verified"`. Its `source` object has exactly `url`, `topic_pin`,
`content_path`, and `content_sha256`. The HTTPS URL must have the declared
primary-source hostname; the nonempty topic pin identifies the exact section,
version or capture date. The referenced nonempty source bytes are independently
hashed and must also stay outside Workbench. All source IDs declared by the case
need receipts. Arbitrary status JSON, a changed file, another origin, a model
assertion or solver input cannot satisfy this check.
Adding the public briefs and prefrozen archive contracts changes task signatures;
an earlier source pack must be rebound to the current exact task signature and
checkout revision before use. Historical packs cannot silently qualify a new
specification cohort.

The host evaluator freezes source facts for later scoring. It may acquire its
reference snapshot independently; the solver must still navigate and read the
real public sources through GUI. The runner neither fetches those sources nor
supplies their content or answers to the model. File ownership and digests
establish evidence identity, not semantic correctness; source accuracy remains
an explicit independent criterion. This is a protocol boundary rather than an
OS sandbox against a same-user GUI deliberately opening host evidence.
After validation, the runner copies the exact source pack, proof documents and
primary-source bytes into owner-only `host-source-evidence` outside Workbench.
Their identity and later immutability are part of the captured run bundle.

For apps, setup uses the existing owned launcher guards and opens only the
declared blank windows. Files opens the owned Workbench. Every new blank app
must immediately produce a nonempty native observation bound to its exact XID,
PID and process start before another app is opened or a task is submitted.
Missing apps, inaccessible controls, ambiguous windows, namespace/provider
identity mismatches and unsafe LibreOffice handoff fail setup explicitly. Calc
and Writer retain the per-child OpenCL-disable and exact executable guards.
There are no automatic installs, accessibility-setting changes or browser
identity workarounds in this runner.

Fourteen specifications remain implementation-blocked: LHN-077–082 require
released-amendment injection; LHN-083–088 require the declared real media assets
and player materializer; LHN-089–090 require controlled owned-window interruption.
The other 86 have a transport/setup path, conditional on actual source and app
proof. This is not a claim that any one of them is currently host-ready or that
its full office/browser workflow succeeds. Additional app feature, export,
source-content and reopen failures remain observable task outcomes.

## Evidence, cancellation and cleanup

The bundle retains the exact raw NDJSON bytes, receive timestamps, reset events,
tool phases and native outcomes. The canonical event bound is 256 MiB and the
per-stream bound is 512 MiB; complete lines are processed before the pending
fragment limit. This accepts the measured legitimate >4 MiB final event while
remaining bounded. Full cumulative tool traces stay in raw NDJSON, rather than
being copied into summary; a compact native-outcome index binds before/after
state digests to that raw evidence. Each completed tool phase gets fresh read-only
native and filesystem evidence. Initial/final observations, all setup ownership
records, output hashes, ZIP member hashes, pointer samples and a per-case
after-test note are kept separately. The initial and final managed-window
inventory contains XIDs/PIDs/start times only; it does not read unrelated titles
or UI content. Unreadable identity remains unknown rather than a preservation
pass. Worker snapshots read existing identity files only and make no memory or
GPU-residency claim. Checkout hashes do not prove that an already running Python
service adopted those sources; deployment adoption needs its own receipt.

Foreground authority accepts only a registered XID with its current exact
PID/start, or a positively linked transient of that XID with the same identity.
A different top-level window sharing a PID is not automatically authorized.
Unexpected foreground identity or read uncertainty stops this client. Newly
opened unregistered windows are retained as uncertainty; they are never closed
by guessing ownership. This can expose a missing application/window contract
during a long task.
This is passive foreground monitoring and receipt auditing, not a server-side
XID authorization allowlist. It cannot guarantee that cancellation precedes every
input into an unexpected window. Such uncertainty cannot become a method pass.

On an error or missing final event, the runner sends at most one cancel for its
own cookie-scoped native controller. That endpoint sets the Store cancellation
flag; the current native-task planner has a separate stop event and can renew
the Store on a later observation. Neither that cancel receipt nor HTTP abort is
proof that all further input has stopped. The runner waits for actual downstream
natural idleness
before closing anything. Failure to prove idleness keeps all owned windows
open and reports `cleanup_deferred`. Cleanup requests polite closure for exact
recorded windows only; it never kills a browser or shared application. A save
dialog or unknown launch can therefore leave cleanup explicitly incomplete.
Observer/guard/client shutdown errors retain typed teardown failures and the
summary/note; unconfirmed observer shutdown keeps windows and marks closure
unknown rather than skipping the evidence journal.
Do not start another evaluation on that desktop until the retained attempt has
drained and its owned cleanup is resolved.

If the case requires memory and the main turn finishes with an actual successful
`memory_write`, a separate fresh same-cookie text turn requests recall with
`memory_search` as its only task tool. Its evidence is independent of the main
answer; it cannot navigate a browser or reopen files to reconstruct a memory
claim. The client drains again before cleanup.

`metrics.json` records measured wall/phase/input information and requested
sampling. Token counters, prefill latency and GPU resource usage that the public
profile does not expose are explicitly unavailable. `portal_turn_attempts`
counts this client's attempted turns, not unknown upstream weight generations.
`upstream_model_request_count` remains null; no transport count is presented as
an inferred model invocation count.
Delegation metrics deduplicate actual policy/point receipts by content digest,
retain their identity/status/model/revision fields and count new inner input
receipts by phase. Configured auxiliary availability is not a call. The exact
semantic-unique and planner invocation counts remain unknown when not exposed.

## Independent grading

Model completion and the profile's `planner_claim` remain model claims. File
existence, model prose, an exit code or a configured auxiliary model cannot
create a task pass. The runner independently checks input preservation, exact
required output paths, actual typing/pointer receipts, pointer movement and
owned-scope evidence. Physical receipts do not alone prove that no GUI terminal
or application macro was used; that method audit remains part of independent
grading.

All 75 archives have exact membership/byte readback checks. The six reconciliation
cases additionally verify the prefrozen authoritative Current set. The six
editorial cases LHN-047–052 have a full-text exact-edit
oracle: all ten specified edits must occur and every other byte—including
protected quotes, similar distractors and final newlines—must remain unchanged.
Its expected text is computed in memory solely for grading and is never written
as a solver output. A failed automatic check cannot be overridden by a later
all-true review. Other source semantics, formula/chart/layout properties,
source-acquisition/reopen provenance and memory accuracy require the predeclared
case-specific independent rubric. Missing graders remain pending, not passed.

After reviewing the raw bundle, run an offline evaluation:

```bash
.venv/bin/python -m runtime.verify_long_horizon_native \
  --output runtime-data/long-native/LHN-047-ornith-001 \
  --evaluate --review-file /absolute/host-evidence/LHN-047/review.json
```

The run freezes an `evidence-manifest.json` with path/size/SHA256 for every file,
including source proofs, raw main and recall streams, pointer/native/setup
evidence, final artifacts, preservation records, summary and after-test note.
Symlinks, nonregular or oversized files fail evidence freezing. Only this derived
manifest and `independent-verdict.json` are excluded to avoid self-hashing.
Any later added, removed or changed evidence invalidates grading.
Evidence files are bounded at 512 MiB each and the complete bundle at 1.5 GiB.

An evaluation without a review verifies the manifest and writes
`independent-verdict.json` with its bundle SHA256. An independent review must be
outside the run bundle and has schema
`robit.omni.long-native-independent-review.v1`, exact `case_id`, that
`bundle_sha256`, `authority:"host_independent_evaluator"`, and `criteria` mapping
**every** predeclared criterion ID to a native boolean. The evaluator re-reads
the actual files, requires them to match the captured final inventory and
checks source stability, confirmed downstream idleness, exact cleanup and
preexisting managed-window preservation. It preserves the original summary and
failed/unresolved action statuses. A review is a grading record, not permission
to execute or a substitute for evidence.

Exit codes distinguish the boundary: 0 for successful offline preparation or a
fully passed independent review, 1 for a request/check failure, 2 for setup or
readiness blocking before submission, 3 for unconfirmed drain, and 4 for pending
or unsuccessful independent review. No live task is marked passed by its initial
run. Report blocked, attempted, failed, pending and passed cells separately;
retain every attempt and correlated family aggregate. These are original tasks
inspired by saved primary-source benchmark protocols, not a reproduction of
their environments, evaluators or scores.
