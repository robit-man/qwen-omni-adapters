# CU.RUN.CLEANUP.075 — native harness ownership admission

This isolated harness change addresses a specific failure: an unresolved prior
run must not become the next run's unrelated baseline. It provides no model,
GUI, task-completion or Egg acceptance by itself.

Live `verify_long_horizon_native` runs acquire a mandatory, nonblocking exclusive
lock under `~/.local/state/qwen-omni/native-test-runs`. The session partition binds
the actual UID, current boot ID and canonical local X11 display (`:0.0` and
`:0.1` share the `:0` lock). A trusted host may select a private alternate root
with `--run-ledger-root`; there is no ledger-off flag. Imports and prepare-only
runs do not create ledger state. Unqualified/remote display strings fail closed.

The private ledger records the actual runner PID/start/UID, a unique run ID,
case and original evidence directory. Before setup, a complete bounded managed
window inventory must reconcile every pending run. A live previous runner,
uncertain unregistered launch, still-present owned XID, changed/reused identity,
invalid record or unavailable inventory blocks setup. No window is closed by
reconciliation. Unknown ordinary baseline windows are preserved and are not
adopted as test windows. Historical ownership completeness is not claimed.

`TaskDesktop` accepts an optional host-only lifecycle publisher; the live runner
always supplies it. Direct historical callers remain compatible. Checkpoints
precede Popen, follow Popen, preserve new-window candidates and immediately
follow exact positive registration before explicit activation. A copied positive
registration checkpoint preserves the existing later executable-check and
uncertain-cleanup behavior. A namespace launcher records intent before its
existing private helper and publishes the helper's verified proof on return;
an uncertain helper failure stays pending. Raw namespace PIDs grant nothing.

The registry has at most 32 pending runs, 16 launches per run, 256 managed XIDs,
1 MiB per sealed record, 64 KiB index, 128 compact closed summaries, 256 checkpoint
events and 224 directory entries. At most 8 explicitly pinned, host-normalized
migration documents of 1 MiB each may be supplied as `--run-ledger-migration
SHA256:PATH`. A migration document binds the exact desktop session, pinned
original journal paths/hashes and complete normalized ledger records. This is
an explicit trusted host normalization step; arbitrary legacy journals, titles
and process names are never ownership evidence. Missing, malformed or unsafe
migration references block admission. Files must be private regular single-link
files owned by the actual UID; symlinks and unsafe paths are rejected.

Atomic publication uses a private exclusive temporary file, a complete bounded
write, file fsync, replace, directory fsync and byte-for-byte reopen verification.
Prepublication failures stop the next launch/input. A post-replace sync failure
is explicitly indeterminate. A failed checkpoint remains sticky through finally;
it cannot be erased by an apparent later cleanup. Closing a manager releases
only its lock. Failed/deferred or interrupted cleanup remains pending.

Existing drain, recorder shutdown and polite per-window close are retained.
There is no new automatic input, unsaved-buffer discard, shared-provider kill,
GPU or service lifecycle operation. Only a complete fresh inventory proving the
registered XIDs absent can resolve a run; a known Popen failure with no process
started is separately represented. Full run evidence is retained. Historical
reconciliation never rewrites the prior immutable run bundle: the private record
and new admission report hold a source-bound absence receipt. Bounded closed-ring
pruning removes only registry cache records after checking retained original
evidence; it never deletes full run artifacts.

CPU tests use real ledger, desktop and runner classes with explicit process,
input and transport doubles, plus one ordinary CPU crash process. They do not
prove actual blank content, safe unsaved-buffer handling or application closure.
Existing fresh setup/content/effect oracles are unchanged. Root-controlled real
normal/error/cancel/admission fixtures and separate Egg gates remain required.

## Explicit verified settlement of one uncertain run

Ordinary `reconcile` still rejects `cleanup_uncertain`, even after its registered
windows become absent. A trusted host operator may separately call
`reconcile_verified_settlement(inventory, settlement_ref={"path": ..., "sha256": ...})`.
It resolves only the pinned pending run and does not admit a new run or add
window-close, discard, process-kill, GUI or task-progress authority.

The private closed `robit.omni.native-verified-settlement.v1` receipt binds the
run/session, canonical pending body, raw sealed record and original mirror,
canonical scope/launches/current inventory, actual live verifier PID/start/UID
and actual `__main__.__file__` path/raw source hash. Its observation/expiry window
is at most300 seconds. Separate pinned closed
`robit.omni.native-host-verification.v1` envelopes attest `downstream_settlement`
and `owned_cleanup`; each is no more than300 seconds old at verification time
and no later than the receipt observation. Cleanup binds the same inventory and
the exact sorted registered XIDs proved absent. Receipt, method and each evidence
file use the existing private-file reader with a64KiB limit.

This is auditable trusted Python-host provenance, not authentication of arbitrary
JSON or a generic verifier for external observations. The host operator must
actually obtain the settlement/cleanup evidence. Missing, false, stale, unbound
or unsafe evidence cannot clear uncertainty. Existing original-runner lifetime,
registered-launch, complete inventory and ownership-absence checks still apply;
publication-failed records remain unsupported. Every validation precedes the
first write. Invalid proof leaves index, records and mirrors unchanged.

Success archives only the private registry record, retaining its entire prior
uncertain outcome, original raw identities, current inventory and proof
references. The original mirror remains byte-identical. Closed-history and
storage limits are unchanged; write failures retain existing indeterminate
publication semantics. The returned `state=resolved` is not new-run admission:
ordinary fresh admission must still reconcile every other pending run.
