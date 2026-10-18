# Independent review of the actual Ornith baseline

The saved run reproduces **7 correct finite choices out of 9**, with 2 additional
review-only responses. All 11 raw request/response bindings pass independent
checks. The original 44-cell matrix still has 33 unrun cells. These text-only
adapter calls earned **zero whole-task passes**.

The complete machine-readable audit, per-case raw-byte hashes, predictions,
source checksums and reduced GPU observations are in
`baseline-independent-review-20261017.json` (SHA256
`b768b610c9dd61806deac74e131e098218562741ac8ea762d679026ec8c7fe7b`).
The reviewed run is
`runtime-data/gpu1-acceptance/operator-audit/execution-sop-20261017/actual-ornith-baseline-20261017T090234Z`.

Every actual request parses to its frozen rendered request. All receipts match
the raw HTTP byte hashes and the run manifest; the saved grades reproduce
exactly. Replies name the expected Ornith tag, return HTTP 200 and `done:true`,
and contain no thinking channel. Requests preserve native `think:false`,
temperature 0, requested output budget 8192 and text-only/no-tools settings.
The audited executor calls the existing adapter directly and dispatches no GUI
input or service change. This does not independently monitor unrelated user
activity on the desktop.

The incorrect finite choices are `PTR.ARCHIVE.030` (replans instead of taking
the permissible Open prerequisite) and `PTR.NEGATIVE.030` (double-clicks the
brief). The latter has a fixture inconsistency introduced by this verifier: it
claims a complete current brief while actually retaining a file listing.
Its frozen incorrect score and raw input remain intact, but its proposed
mechanism is inconclusive.

Several correct choices have unsupported or incorrect explanations:

- ARCHIVE010 predicts draft-1 opens after choosing the Input folder.
- ARCHIVE040 predicts a file picker; the archived actual transition opened
  recent documents. This new inference call verifies neither transition.
- NEGATIVE020 denies visible brief text even though the full text was supplied.
- NEGATIVE050 calls the Open composite a Remove button from another task. Its
  synthetic failed-strategy record also sits beside an original no-input
  abstention receipt, limiting recovery-mechanism interpretation.

ARCHIVE050 and 060 remain review-only and have identical request and response
bytes. ARCHIVE030 and 040 have different objectives, so they do not isolate
the handoff change causally.

Saved broker observations prove the trunk and Kev PIDs were resident only on
the reserved GPU1 UUID under an active scoped lease. The expected release
horizon is overdue and advisory; it is not heartbeat expiry. Current read-only
checks through the accelerator module corroborate all eight PID/start
identities and GPU1 residency for trunk, Kev and point. TTS is currently not
resident and was not used. These calls exercised language inference only, not
Kev decisions, Moondream, speech, ASR or mouse input.

The executor matched the loaded model path to the installed Ornith model-layer
digest, and current trunk arguments corroborate that blob association. A new
hash of the 5.6 GB tensor file was not computed. Raw `/props` and GPU command
output were not retained; reduced before/after observations were. The recorded
65536 context and requested 8192 output budget do not independently establish
an effective output-token cap. `baseline_source_revision` belongs to the idle
CPU frontend and does not prove adapter or decoder source adoption.

The separately frozen version 2 current-document diagnostic is in
`../native_orchestration_acceptance_v2/`. It uses the actual full brief/editor
state and changes only the immediate objective within the original user goal.
All four version 2 cells start unrun; it does not replace the historical row,
the original 44-cell panel or the 100-task whole-task requirements.

No model call, desktop input or GPU/service mutation was performed by this
reviewer. The raw run remains under cleanup-managed `runtime-data`; root must
archive it durably outside that directory before closing evidence retention.
