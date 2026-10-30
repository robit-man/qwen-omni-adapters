# 075 revision3: verified recovery of the retained test run

The preceding goal turn made progress: it exercised real Ornith weights, sealed
the failed whole task, isolated a seven-file cleanup release and ran its baseline,
focused and full gates. Its first real fixture failed before close, revealing an
observer-contract mismatch and a distinct sticky-reconciliation gap. Neither
failure is completion. The user has reiterated that actual computer-use task
completion is the priority; this repair is only the bounded prerequisite to
resume those runs.

## Failing acceptance, evidence and hypothesis

Run19d7eceab4594a2bb626dbf2d932f1dc retains one registered blank test window81790304.
The default ledger has pending1/closed5. In the frozen455-source cleanup release,
`reconcile` rejects `cleanup_uncertain` before checking absence. No supported
operation can resolve that state after independently verified settlement/cleanup.
Original failure review: a900ad82dd3557b49b8bcb3643c3fdb4cc6889abd43b67a114a041ea097b1bd3.
Independent pre-edit acceptance:7229d0d9885f389910eaca98335da39c0c9ce53084a7af0a0cbf600b13d1b993.

Hypothesis: a separate explicit host-only reconciliation operation can resolve
exactly this uncertainty after all original runner, launch, ownership, private
file and current inventory checks plus fresh host settlement verification.
Ordinary reconcile must continue refusing uncertainty. No GUI/model authority,
force/reset flag, original-evidence rewrite or automatic cleanup is added.

## Exact source scope and isolation

Create a new private worktree derived from published54b plus the exact frozen075
seven-path patch. The baseline455-file inventory is
0a8043adf201c46954e57eac1adeb1a6710ccc3e02bdef990934769e3f1d0b21.
Run validation before editing. Change only runtime/native_run_ledger.py,
tests/test_native_run_ledger.py and
docs/research/computer-use-fresh-run-cleanup075.md. Preserve the prior release,
dirty MAIN, planner/provider/oracle files, worker services and prior failed runs.

## Frozen private API and proof contract

`NativeRunLedger.reconcile_verified_settlement(inventory, *, settlement_ref)`
consumes exactly one currently pending uncertain record. settlement_ref is an
absolute private-file path plus its raw SHA256. Existing guarded `_read` rules
apply to the receipt, method source and two evidence files; each is bounded at
64KiB. This is the trusted Python host-caller boundary used by finish(settled),
not authentication of arbitrary JSON and not a model-facing tool.

The closed receipt schema is robit.omni.native-verified-settlement.v1. Required
fields: schema, run_id, session, pending_body_sha256 (canonical),
pending_record_sha256 (raw sealed file), mirror_sha256 (raw original mirror),
scope_sha256, launches_sha256 (canonical), inventory_sha256 (canonical),
producer (PID/start/UID), method (path/raw SHA256), observed_at_unix,
expires_at_unix, settlement_evidence (path/raw SHA256), cleanup_evidence
(path/raw SHA256). Reject additional/missing fields and malformed bindings.

The producer must equal this ledger instance's actual live runner identity.
The method reference must equal the actual __main__.__file__ and current source
hash of the running host operator. This supplies auditable host provenance;
the trusted operator remains responsible for actually observing settlement.
Times must be finite numeric values excluding bool, with observation <= now <
expiry and expiry no later than observation+300 seconds. Evidence observations
must be no later than receipt observation and no older than300 seconds.

Each evidence file uses the closed envelope
robit.omni.native-host-verification.v1 with fields schema, kind, run_id,
session_sha256, pending_record_sha256, producer, method, observed_at_unix,
passed and observations. kind is respectively downstream_settlement or
owned_cleanup; passed must be true, and observations must be a nonempty mapping.
Both bind the same run/session/raw record, actual producer and method. Cleanup
observations must additionally bind inventory_sha256 and exactly the sorted
registered owned_window_ids_absent. Other observation details are bounded audit
data; the API does not invent a generic verifier for external systems.

Before mutation: require no current run/publication failure, exact pending
record/body/mirror/session/scope/launch matches, ended original runner lifetime,
all known registered or known failed-before-spawn launches, and existing complete
inventory/_absence checks. Present reused/unknown XIDs still block. Unknown
original runner identity blocks; a different PID start proves the old lifetime
ended and grants no window authority. Publication-failed pending records remain
unsupported and blocked. Missing/false/stale/unbound evidence cannot resolve.

Success uses registry-only historical archival, retains the full previous
uncertain outcome and proof references, leaves the original mirror byte-exact,
resolves only this run, and does not implicitly admit other pending runs.
Invalid input must leave registry/index/mirrors unchanged. No bounds increase.

## Gates and bounded actual acceptance

Implement CPU RED→GREEN for the actual-shaped unresolved record; ordinary
absence stays RED, explicit fresh bound recovery is GREEN. Cover every family
in independent acceptance, including no-mutation negatives, live/unknown runner,
unregistered launch, wrong session/run/hash, unsafe files, invalid times/issuer,
stale/replayed proof, present/reused owner, incomplete inventory and publication
failure. Preserve existing075 tests; run focused and full validation with explicit
private/src:private PYTHONPATH and independently review the exact source.

Separately adapt only the already reviewed external reader5711f557 to the new
source pin/root. Before any input, freeze/review a host cleanup operator. It must
first observe the pending block, rebind exactly81790304/3036686/start4365561/UID1000,
freshly prove the complete blank buffer with actual foreground before/after,
politely close only that window, verify its absence and prior-window preservation,
and independently observe both portal activity and trunk slots idle. It issues
the source-bound private evidence in the same live operator process and calls
the supported API; no private archive/finish/flag edit. Preserve the old failure
and mirror, then obtain fresh default admission. If any proof fails, stop and
preserve the registered pending state without discard or process kill.

Only then run one fresh four-case lifecycle window against this exact source.
Its results remain separate from the failed first attempt and controlled cleanup.
Do not publish until actual evidence passes. Resume one evidence-backed controller
repair and the original whole task; do not extend cleanup into an unrelated
framework or count any of these checks as model/task competence.
