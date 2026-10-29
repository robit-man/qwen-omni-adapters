# CU.RUN.CLEANUP.075 — durable ownership and fresh-run admission

Status: bounded harness repair authorized; no application close or GUI rollout.

The user identified stale applications left by earlier runs. Read-only audit
f8df1f4ca4aaae640c7c992ff29f4793c42f642f72a455ddb055d935b549df06
found15 local/5 Egg41/3 Egg31 current windows, but no positively proven still-live
test ownership in the bounded historical journals. Preserve those23; neither
claim they are all user-owned nor close them by title/process-name matching.
Latest073r2 closed its three test windows and preserved its original15 baseline.

## Failing criterion and bounded hypothesis

A fresh run must not silently adopt a positively known residual prior-test window
as unrelated state. The current runner snapshots all existing windows before
launch and writes ownership only in finally; no cross-run reconciliation exists.
A retained/deferred/failed cleanup can therefore become the next preserved
baseline. Durable registration plus a bounded per-user/display pending-run ledger
and read-only admission will prevent that known contamination. This is a test
harness repair, separate from control-discovery074 and model behavior.

## Required implementation boundary

Implement in a NEW isolated exact073r2-derived candidate. Own only the new pure
runtime ownership-ledger helper, runtime/long_horizon_desktop.py,
runtime/verify_long_horizon_native.py and dedicated lifecycle tests/documentation.
Do not edit portal/native/model/prompt/effect/oracle code or074candidate. Root
integrates only after independent gates. Existing073574-source5092PASS is the
pre-edit byte-exact baseline. Keep all MAIN changes and failed attempts intact.

Use a private host-selected state directory (default per-user native-test ledger,
segmented by desktop session/display), a bounded pending-run index and an actual
runner PID/start/UID identity. Use an exclusive local lock to prevent two test
harnesses on the same desktop claiming independent ownership concurrently. Do not
store credentials, env dumps or generated capability URLs. Reject unsafe files,
symlinks, unbounded/malformed journals and foreign-user records; never interpret
journal text as commands. Declare exact storage/read bounds before implementation.

Publish run intent and the existing launch/registration evidence atomically with
fsync/readback at the relevant launch boundaries, including immediately after
positive registration, not only at finalization. Keep exact XID/PID/start/UID,
run ID and existing before/new-window/namespace proof. Shared providers remain
shared: registration of one window never grants process-wide cleanup authority.
The runner must use this path by default, not require an optional flag the next
caller can silently forget. Keep legacy direct TaskDesktop usage compatible;
actual live runner admission/publishing is mandatory.

Before fixture launch, reconcile unresolved registered runs with a fresh managed
inventory and exact recorded ownership. A live prior runner, same-identity owned
residual, pending uncertain/unregistered launch, malformed evidence or unavailable
identity blocks admission. A proved absent prior-owned window may be reconciled
and archived; a reused/unknown identity must not be killed or guessed. Support
host-selected verified older journals for migration; do not scan or erase arbitrary
home/cache/model trees. Unknown preexisting windows remain protected and are
explicitly outside historical ownership completeness claims.

Keep existing downstream drain, recorder shutdown, owned scope and bounded polite
per-window close. No automatic unsaved-buffer discard, app kill, keyboard shortcut
or GPU/service lifecycle mutation is introduced here. If cleanup is deferred or
fails, preserve artifacts/buffers and leave a durable pending record that blocks
the next run. Only positively verified absence after existing cleanup may resolve
the run. Every normal/error/cancel path records outcome; abrupt process death must
leave enough pending evidence for next admission to refuse stale reuse.

## Required checks and actual acceptance

Independent lifecycle acceptance5196901a67d47feee75e8370a8579b9d6db6e31d7a8779178f8de92ca1721136
is frozen externally. Add meaningful CPU/integration controls for crash/retained
cleanup→next-admission blocked, active runner conflict, exact disappearance,
foreign/reused/unknown identities, namespace uncertainty, malformed/symlink/oversize
journals, atomic write interruption and immediate publication before a later
failure. Verify original normal/error/cancel/drain behavior and zero unintended
adapter input. Fake adapters are CPU evidence only, never real desktop/model pass.

Run focused and independent gates and validate.sh against exact source. Root then
uses one serialized bounded real owned-window normal/error/cancel/admission test,
preserving its buffer/artifacts and verifying no residual owned IDs plus unrelated
windows unchanged. Real fixture gates on each Egg are separate. This harness does
not itself satisfy computer-use goals, model competence or whole-task acceptance.
All actual GUI/model work remains root-controlled after source/method review.

## Revision2 — optional migration never writes referenced evidence

Before actual adoption, a CPU reproduction found that migration import reused
_publish's mirror-writing path and could change a prior ownership evidence
bundle before subsequently refusing admission. The frozen075r1 inventory is
578files14fb143ba2cfdaf3749c7f5bf93e1021dca4e09ca704fe43bcbc2ad67f87edf1;
its source and original failed reproduction remain preserved. No deployed or real
user evidence was modified by this controlled counterexample.

In a new075r2 exact copy, change only the ledger and dedicated regression:
ALL migration evidence roots are read-only. Require an existing byte-exact
matching sealed ownership mirror, validate it before index mutation, then
publish the imported record only in the private registry (mirror=False). Missing
or conflicting mirrors fail without adding/replacing any file in any referenced
root. Do not add auto-provisioning of empty migration directories. A host-selected
normalizer may separately prepare new evidence explicitly if later needed; this
optional import does not confer authority to rewrite old bundles.

Retain the authorized old mock_run helper's explicit FakeLedger compatibility
change while dedicated controls test real mandatory runner wiring. Update the
source map/patch and run the preserved RED→GREEN plus immutability/fault controls.
Root integrates only the final independent-reviewed frozen075r2 delta with final
074r3, then runs one full validation on that exact combined cohort. Intermediate
075focused checks are not claimed as a standalone full gate or live cleanup pass.
