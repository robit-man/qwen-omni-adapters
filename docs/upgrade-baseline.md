# Upgrade fork baseline — 2026-10-29

## Exact preserved source

- Original checkout: `/home/roko/Documents/qwen-omni-adapters-main`.
- Original starting HEAD: `24f1691767e61c94350fe0610c20036d0eec560b`.
- Preserved local snapshot: `cf0183da45b9a4992e105373e37d2d6e7bdfedbe`.
- Original remote branch: `archive/local-work-20261029`, pushed and checked
  with `git ls-remote` before resetting the original checkout.
- Snapshot Git tree: `a97594a4006438e58301ea405a37e0d8bcb08545`.
- The new clone's initial tree was verified identical to this snapshot.
- New checkout: `/home/roko/Documents/qwen-omni-upgrade`.
- Original checkout refreshed to `origin/main`, initially `7554ae0` (31 commits
  beyond the original baseline); no deployment was restarted.

This is an exact copy of the versioned source, including previously untracked
source, research and acceptance fixtures. Ignored environments, generated model
views, credentials and live runtime/PID state are not release source and were
not copied. The fork has an independent Python environment and runtime state.
The original branch and full Git history retain the checkpoint for comparison.

## Assessment: preserve, do not merge indiscriminately

The original working tree contained 20 modified tracked files (2,000 insertions,
224 deletions) and 155 untracked files (4,578,662 bytes). These became one
175-file preservation commit. The new files include real controller modules,
typed effect contracts, recovery/plan state, GPU-relocation support and test
fixtures; the work is not disposable clutter.

Eight modified tracked files already exactly matched the fetched main:
`docs/runtime.md`, `portal/start.sh`, `portal/ui_observations.py`,
`runtime/acceptance_native_support.py`, `tests/test_native_observation_recall.py`,
`tests/test_native_pointer_resolution.py`, `tests/test_native_subgoal_handoff.py`
and `tests/test_ui_intent_choice.py`. Other changes overlap with later upstream
commits. Do not blindly reapply the archive over newer upstream implementations.

The unique controller work is experimental. Its progress file says the previous
goal was paused and the whole requested outcome was not verified; several stages
have only controlled-scope passes. Retain the source and evidence, but do not
promote them as accepted whole-task improvements or merge them into original
main as part of packaging work.

## Actual pre-upgrade gate

`./scripts/validate.sh` passed compilation and Ruff. The original test run
recorded **2,472 passed, one failed** before a deliberate interrupt at 212.59 s
when the user's task changed. The full suite was not completed.

The observed failure was
`tests/test_background_agent.py::test_background_worker_replans_generic_discovery_without_camera_capture`:
the test expected a completed background task after its three-second window and
observed `pending`. This is an inherited failure, not a packaging result. Do not
weaken the assertion or alter unrelated background-task behavior to claim a green
upgrade. Fresh gate results belong in the milestone evidence.

The refreshed original main subsequently passed the full validation script:
**6,167 tests passed in 400.42 s**, followed by all remaining JavaScript, shell
and contract gates. This is validation of original `7554ae0`, not the older local
snapshot used as the upgrade fork's requested exact source baseline. The fork's
separate pre-edit run passed compilation/lint and 1,409 tests before interruption
of redundant concurrent validation; it was not a complete gate.

## First bounded engineering objective

Current `prepare_ollama_sidecar` reconstructs all selected component files on
`--overwrite`, even when the installed sidecar and existing views are unchanged;
each view reparses the sidecar inventory. Bootstrap always requests overwrite.
A missing member causes reconstruction of intact members as well.

Acceptance: preparation can use one indexed source inventory and a versioned,
source-bound receipt to reuse verified unchanged views and rebuild only misses.
Quantized tensor payloads, metadata, names, order and shapes must survive. Show
zero component reconstruction on a warm preparation, selective repair after
corruption/change and measured avoided bytes. This is the first Phase A step,
not a claim of direct filtered mmap, tensor-content deduplication, an Edge v2
weight release, lower GPU residency or achieved speech-latency targets.

## Inspected resources

`/srv/fine_tuning_suite/README.md` documents the existing single-trunk bridge and
one-tag packaging. Relevant reusable sources are
`training_suite/models/single_gguf.py`, `models/ollama_sidecar.py`,
`training/omni_encoder_bridge.py`, `omni_runtime.py` and Qwen3-TTS patches.
These resources corroborate the architectural baseline; new training has not
been run and no new neural head is claimed.

The official [Qwen3-TTS documentation](https://github.com/QwenLM/Qwen3-TTS)
describes reusable reference conditioning and distinguishes Base, CustomVoice
and VoiceDesign. Its Python API is not proof that the pinned GGUF backend already
supports all those controls. Preserve and qualify the shipped Base path first.

## Production baseline integration — 2026-10-30

The initial exact-copy fork deliberately preserved the older local tree. Before
release qualification it must also contain the refreshed production baseline
`7554ae0118a2b1527ab537a1cd11e4cd7251770c`: 31 commits add caret-verified typing,
foreground/freshness handling, planner state, GPU relocation/ingress identity,
incremental CUDA builds and quieter speech-onset handling.

Integration is tested in an isolated worktree. Overlapping controller, recovery,
media and contract fixtures use the newer production versions. The archived
local work remains reachable through `cf0183d`, its pushed archive branch and the
merge's original parent; nothing is force-pushed or erased. Nonconflicting source
and all measured P0 upgrades are retained. The CLI combines exact-bound indexed
packing with production broker probing; native grammar fixtures combine every
P0 patch with production compiler/incremental-build coverage.

The first integrated run passed 6207 tests and failed one media provenance check
because this fresh worktree had no bundled FFmpeg and readback used a PATH name.
The original production validation had its ignored bundled tools available. An
independent copy of that exact CPU tool bundle was provisioned in the upgrade
checkout for equivalent validation. No assertion or unrelated media behavior
was changed to hide the failure. The final gate result is recorded separately
in `docs/upgrades/production-integration.json`.
