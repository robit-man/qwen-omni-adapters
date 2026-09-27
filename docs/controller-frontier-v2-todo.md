# Controller frontier v2 — bounded implementation checklist

## Objective

Eliminate the observed durable-task loop in which a verified checkpoint or a
successful retrieval returns to unrestricted planning and repeatedly inspects
unchanged state instead of performing the committed mutation.

## Acceptance criterion

From a progress checkpoint whose next frontier commits `ACT`, the next external
operation must be a mutation within the declared target. Read, list, search, and
free replanning operations are rejected until that mutation succeeds or returns
an audited failure. After mutation, a fresh verifier must address the exact
changed resource.

For the preserved CareNest task, one migrated frontier-selection round may run.
After it commits `ACT`, observe at most ten controller transitions. The task must
either advance environment version beyond 12 and verify the changed resource,
or retain a concrete audited execution failure. Repeated inspection is a fail.

## Allowed files

- `src/qwen_omni_adapters/context.json`
- `portal/background_tasks.py`
- `harness/background_agent.py`
- `tests/test_background_agent.py`
- `tests/test_context_catalog.py`
- `docs/context-engineering.md`
- `docs/controller-frontier-v2-todo.md`

No other file may change during this implementation. Pre-existing untracked
workspace directories are not part of the change.

## Non-goals

- No model-weight, context-window, TTS, ASR, camera, browser, or UI changes.
- No task-, filename-, suffix-, or CareNest-specific heuristics.
- No semantic filtering of model prose.
- No tuning unrelated latency, memory, or tool-selection behavior.
- No follow-on fix discovered during the bounded live observation.

## Checklist

- [x] Freeze scope, acceptance criterion, allowed files, and stop conditions.
- [x] Add failing regressions for all five observed controller defects.
- [x] Add controller protocol v2 frontier state and v1 migration.
- [x] Make progress checkpoints atomically install the next finite frontier.
- [x] Make successful retrieval consume a declared gap and activate its durable successor.
- [x] Replace global-environment evidence freshness with resource fingerprints and scoped invalidation.
- [x] Retire equivalent controller routes across read/list/shell operation changes.
- [x] Persist manager rejections across slices/restarts and narrow exhausted decision classes.
- [x] Prove compaction, foreground preemption, and orderly restart preserve the frontier.
- [x] Update controller documentation.
- [x] Run targeted tests and `./scripts/validate.sh` once after the complete patch.
- [x] Commit and push one implementation commit.
- [ ] Pull/deploy once on the Egg without restarting unrelated services.
- [ ] Observe at most ten controller transitions and record pass/fail without another patch.

## Validation evidence

- Targeted controller/context suite: pass.
- First repository gate exposed one import-order error and two in-scope
  compatibility failures (browser element resource identity and 4K envelope
  size); no unrelated failure was patched.
- Final repository gate: 797 passed; all validation gates passed.

## Stop conditions

Stop immediately and report evidence if any of these occurs:

1. A required fix needs a production file outside the allowlist.
2. The full validation suite fails for a cause outside this change.
3. The Egg exposes a different failure after the specified loop is closed.
4. Ten post-frontier live controller transitions elapse without acceptance.
5. A second implementation commit would be required.
