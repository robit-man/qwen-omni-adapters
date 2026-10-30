# CU.NATIVE.COST.073 revision2 — isolate the density-test enum

Before-edit criterion: the full source gate must pass on the frozen runtime
candidate before the one actual moving-click fixture. The held clean release's
unchanged451-file gate failed with12 failures and4,350 passes. All12 fail in
later popup_fixture construction: its SimpleNamespace(**vars(STATES),HAS_POPUP=...)
receives duplicate HAS_POPUP because the new density test assigned that property
to the shared STATES object. The release receipt is
787f4b9d49ae2d38e6096a1d47ef8f04fd6029e3dc71e5ff1f5f0c8c3c73b99f.
The separate574-file073 gate remains running and immutable; preserve it terminally.

Bounded repair: in a new revision2 copy, replace only the density fixture's direct
StateType.HAS_POPUP assignment with pytest monkeypatch.setattr(...,raising=False),
so normal fixture teardown restores the shared enum. No runtime code, proof,
limit, authority, oracle or solver setting changes. Preserve frozen073 source530a
and failed clean release1d0 unmodified. Native bytes remain e9089074... .

Reproduce old test-order failure and verify the same density-then-popup sequence
passes under revision2. Freeze the two cohorts, rerun required full source gates,
and update only source paths/hashes in the unchanged actual harness. An independent
review must verify one-line test-only diff and byte-identical runtime. The actual
moving-click trial remains unrun and is authorized only after the new gates pass.
Do not add runtime optimizations, fold unrelated source, or credit this CPU repair
as native/model/whole-task success.
