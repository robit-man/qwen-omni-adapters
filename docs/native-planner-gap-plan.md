# Native planner gap — implementation plan and tracked checklist

Started 2026-10-19 on the Ornith Egg (Jetson, X11 :0). Source: the five
engineering changes in [computer-use-research-review.md](computer-use-research-review.md),
applied to failures observed in live LHN-001 runs 001–007 on this Egg.

## Observed baseline (Egg, 2026-10-19)

- Direct planner prompt ≈ 78K characters; ≈ 63% is the pasted response schema.
  One planner call ≈ 34K tokens, ≈ 72 s.
- Kev is not configured here (`not_configured` selection); role-only targets
  ("button", 8 candidates) abstain.
- Projection returned 15 of ≈ 199 nodes; the Files list (`Input`) was omitted
  until the GTK tree-row fix (step 0). Run 007 then showed `Input` but the model
  still planned focus/switch steps.
- Failed plans repeat verbatim (run 006); the 3-failure stop ends the task.
- Only `LHN-001-C09` is graded automatically; partial progress is invisible.

## Rules for every step

- One hypothesis per step: state the failing criterion, implement, run the
  relevant tests, run `./scripts/validate.sh`, commit. No stacked fixes.
- No lowered thresholds, relaxed guards, task/file-name/keyword rules, or Kev
  changes (#8 is separate). Typed roles/structure/capabilities only.
- No live desktop run until steps 1, 2 and 4 pass offline replay, and then only
  when the user says the desktop is free.
- Commits stay local until the user approves a push (auto-update deploys
  `origin/main` to the Eggs).

## Checklist

### Step 0 — Land the fixes made during runs 001–007

- [x] 0.1 Run `./scripts/validate.sh` on the current working tree.
- [x] 0.2 Fix any failure caused by the uncommitted changes (none: 6126 passed, 1 skipped).
- [x] 0.3 Commit #10 settlement: `runtime/native_run_settlement.py`,
      `runtime/native_run_ledger.py`, `tests/test_native_run_settlement.py`.
- [x] 0.4 Commit Flatpak Firefox ownership chain: CSD frame rule
      (`portal/native_ui.py`, `tests/test_native_csd_frame.py`), revalidation
      deadline, XRes client range (`portal/native_identity.py`), transient
      windows and synthetic events (`runtime/owned_flatpak_firefox.py`,
      `tests/test_owned_flatpak_transient_windows.py`,
      `tests/test_owned_flatpak_xlib.py`).
- [x] 0.5 Commit GTK tree-row projection fix (`portal/app.py`,
      `tests/test_native_projection_row_coverage.py`).
- [x] 0.6 Commit guard read-timeout tolerance (`runtime/verify_long_horizon_native.py`,
      `tests/test_long_horizon_guard_timeouts.py`).
- [x] 0.7 Commit opt-in planner trace (`portal/app.py:_planner_trace`) and docs.
- [x] 0.8 Disable the trace directory between live runs (tests write to it).

### Step 1 — Offline decision-interface replay (research change 1)

- [x] 1.1 Freeze cases: copy selected trace rows into
      `tests/live/native_decision_cases/cases/` with a manifest (source run,
      phase, sha256) and typed labels: acceptable operation family + target
      identity (node id), or `no_input` for negative controls.
      Done: `tests/live/native_decision_cases/` (manifest with sha256).
- [x] 1.2 Cases: run 006 phase 1 (pre-row-fix, Input absent → expect "not offered"),
      run 007 phase 1 (Input present), one distractor variant, one negative control.
      Done: run006 p1/p2 (Input not projected), run007 p1/p2 (projected), derived order variant and no-input control.
- [x] 1.3 `runtime/replay_planner_decisions.py`: load case, re-send exact request
      to the adapter (no desktop input), record reply, latency, prompt/schema chars.
      Done: `runtime/replay_planner_decisions.py` (loopback adapter, no desktop input).
- [x] 1.4 Scoring: (a) acceptable operation offered in projected state,
      (b) model chose it, (c) chosen target resolves uniquely; separate fields.
- [x] 1.5 `--mode direct|compact` switch that rebuilds the request through
      `ui_decision.project_direct` / `ui_decision.project` from the frozen packet.
      Frozen cases hold the direct *view* packet (`project_direct` output), not the raw packet; compact replays score the model's compact choice and record canonical-expansion errors separately. The trace now also records the raw packet for future cases.
- [x] 1.6 CPU unit tests for scoring with a fake adapter.
      Done: `tests/test_replay_planner_decisions.py`.
- [x] 1.7 Run the baseline (direct mode) on the loaded model; record results here.

### Step 2 — Compact interface without Kev (research change 2)

      Baseline direct: 0/9 correct opens of Input on projected-Input cases (3 cases × 3), 0 unique targets; prompts 81–97K chars; no-input control correct.
- [x] 2.1 Read `portal/app.py:2522` gate and `ui_decision.project/expand` to list
      every place that assumes an active provider. Finding: `ui_decision.project`
      adds `pointer_or_activation` only when the canonical schema has `ui_subgoal`
      (Kev intent-choice only). On the Egg the `Input` cell is offered only for
      `scroll`; opening it is not expressible. The canonical schema does have
      `ui_boundary`, which compact `visual_pointer/key_chord/scroll` already expand to.
- [x] 2.1b Without `ui_subgoal`, offer an exact-control activation family
      (verbs = node operations ∩ {invoke, click, double_click, right_click}) that
      expands to `ui_boundary` with the exact `node_id`; no model-chosen coordinates.
      Done: `activate` family in `ui_decision.project/_expand` (per-verb branches, exact node, never the active scope).
- [x] 2.2 Split the gate: compact projection allowed with `OMNI_NATIVE_COMPACT_PLANNER=1`
      without Kev; Kev delegation still requires an active provider.
      Done: `portal/app.py` compact gate no longer requires an active provider.
- [x] 2.3 (not needed) `portal/computer_use.py:_select`: no provider + several executable
      candidates → typed `ambiguous_requires_planner_choice` with bounded candidates
      (compact mode only); legacy behavior unchanged otherwise.
- [x] 2.4 (not needed: compact `activate` binds one exact observed node, so the `not_configured` ambiguity branch is unreachable in compact mode; direct path unchanged) Route that result into the existing slow-planner escalation
      (`ui_decision.retain_pointer_recovery`, `resolve_pointer_proposal`).
- [x] 2.5 Adjust `ui_decision.SYSTEM` text only where it claims Kev arbitration.
      Done: one sentence describing `activate` ("when offered").
- [x] 2.6 Unit tests: unique target → no model call; ambiguous → escalation;
      guards unchanged; Kev-active path byte-identical.
      Done: `tests/test_native_compact_without_kev.py`, `tests/test_native_compact_without_kev_route.py` (one exact invoke, no policy call; fails on the old gate). Legacy `effect_contracts=False` schemas are not used by the portal and are not covered.
- [x] 2.7 Replay step-1 cases in compact mode; record offered/chosen/resolved,
      latency, prompt size.

### Step 4 — Projection keeps needed facts (research change 4)

      Two-pass compact (portal-faithful subgoal → anchored decision): 6/9 correct, 9/9 unique targets, prompts 28–46K chars, 22–74 s. Deterministic per case at temperature 0; run007-p1 misses 0/3 (subgoal chooses gedit to read the brief). Derived order variant (Input last) 3/3 vs original 0/3: node order influences the choice.
- [x] 4.1 Measure per-node projected bytes on frozen states; record here.
      Run 007 projection ≈ 800–1050 bytes/node (full ancestor structure); after the row fix Input is projected.
- [~] 4.2 (deferred: no frozen case shows needed content omitted after the row fix) Rank active-window main-content items ahead of chrome by role/structure.
- [~] 4.3 (deferred: model-facing structure feeds `_target_binding` used by handoff revalidation; change needs a failing case first) Model-facing node keeps `parent_id` + container role; full structure stays
      in the executor binding (verify `_target_binding` and handoff still bind).
- [x] 4.4 Make catalogue expansion (`inspect_controls`) visible in the compact
      interface for omitted groups.
      Already true: compact schema offers `inspect_controls` with catalogue references.
- [x] 4.5 (covered by `tests/test_native_projection_row_coverage.py` GTK cases) Tests: frozen Workbench state exposes `Input`; budget unchanged;
      sensitive nodes unchanged.
- [~] 4.6 (deferred: no run reached an edit decision; needs failing evidence) #6 read-vs-edit: carry immediate objective kind into compile; read objective
      cannot compile a mutation of its source. Tests with frozen cases.
- [x] 4.7 (see 2.7) Replay step-1 cases; record.

### Step 3 — Recovery changes execution (research change 3, #7)

- [ ] 3.1 Failed-strategy fingerprint (family + `_target_binding` + reason) in
      `NativeStrategyLedger`.
- [ ] 3.2 Exclude matching candidates in `computer_use._strategy_filter`.
- [ ] 3.3 Surface excluded fingerprints and concrete cause in
      `_NativeWorkingState.planning_view`; keep `selection_receipts` cause in ledger.
- [ ] 3.4 Failure counter: repeated excluded strategy counts as failure; 3-stop kept.
- [ ] 3.5 Tests: reworded identical strategy excluded; genuine alternative allowed;
      uncertain-input locks sticky.
- [ ] 3.6 Replay run 006 sequence; record.

### Step 5 — Grade outputs and milestones (research change 5)

- [x] 5.1 Prefrozen milestone checks in `runtime/long_horizon_cases.py`
      (input opened, outputs created, archive members, source page loaded).
      Done: `progress_milestones` (inputs read back by exact full text, declared-source host URLs loaded, outputs created).
- [x] 5.2 Report milestones in `runtime/verify_long_horizon_native.py`, separate
      from pass/fail; rubric and budgets unchanged.
      Done: `report["progress_milestones"]`; never used by `independent_verdict`.
- [x] 5.3 Tests with frozen run bundles (005, 007 → zero milestones).

### Step 6 — One live check (only on user go-ahead)

      Done: `tests/test_long_horizon_milestones.py`; frozen runs 005/006/007 → zero milestones.
- [ ] 6.1 Rebind readiness pack to HEAD; confirm ledger empty.
- [ ] 6.2 Reload portal; one LHN-001 run with `--browser flatpak-firefox`.
- [ ] 6.3 Record result here; no code changes during the run.

## Results log

(append dated entries as steps complete)

- 2026-10-19 step 0: validate.sh 6126 passed / 1 skipped. Commits 83514e4 (#10
  settlement), d4e6aad (Flatpak Firefox ownership), 5a84cf3 (guard timeouts),
  d4647c8 (GTK row projection + planner trace + this plan; 0.5 and 0.7 share
  portal/app.py). Trace dir renamed to runtime-data/state/planner-trace-captured-20261019.
