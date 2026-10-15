# Recent Omnius layer audit: 2026-10-15

The reference was checked against the requested repository at
`/home/roko/Documents/Projects/Adjacent/omnius/omnius`, including its recent
committed native integration. HEAD, main, origin/main and a read-only origin
lookup agreed on `b3591ee7e6bddc44eec50f59f130419630ce43cf`, dated
2026-10-15 13:52:52 PDT. Existing working changes were preserved and excluded
from claims about committed behavior.

| Commit | Date (PDT) | Relevant change or evidence |
| --- | --- | --- |
| `7f3cf1ff` | October 14, 12:41:50 | Native desktop registrations and aliases |
| `257945bf` | October 13, 18:58:50 | Legacy desktop integration moved to the native controller |
| `6daa6a84` | October 12, 12:13:46 | Retained cross-application native tests |
| `f7e432a4` | October 12, 00:55:29 | Visible Qwen/Kev/Moondream browser fixture |

The corresponding native layer is
`packages/execution/src/rapid-computer-use/native-tool.ts`, wired through
`packages/cli/src/computer-use/runtime-native.ts`. It separates observation
from execution of an exact offered action ID, while the main agent plans.
This runtime's `native_ui`, `ComputerUseStore` and finite-action controller
implement that layer. The reference has no automatic native `run_step` API
whose omission explains the observed failures.

The retained evidence has narrower scope than full autonomous desktop use:

- The October 13–14 native reconciliation gate records 84 native nodes and
  verified X11 capture, with zero model requests and zero input actions. Live
  native Kev/Moondream acceptance remained incomplete.
- The successful Chromium → Nautilus → Gedit chains used fixed fixture/rules
  planning and zero model calls. They verified executor integration and exact
  saved bytes.
- The genuine Qwen/Kev/Moondream success was a four-action owned Chromium form:
  one Qwen plan, four Kev selections and one pixel-only Apply target. It did
  not demonstrate ordinary spoken control of the signed-in native desktop.

These findings come from the committed native reconciliation receipt,
`computer-use/evidence/cross-application-20261012/README.md`,
`computer-use/BENCHMARKS.md` and the visible-host attempt-2 manifest. No committed
model-driven native desktop success was found in the inspected recent history;
that does not rule out demonstrations outside the inspected evidence.

Two control-plane differences remain relevant. Omnius's `AgenticRunner`
maintains an objective frontier, completion ledger, typed effects, stagnation
checkpoints and independent terminal verification. This runtime already has
related background-task machinery, but the direct native-only spoken test loop
bypasses it. Omnius also maintains a current canonical context and can replace
historical transactions with durable receipts under actual context pressure.
The direct portal loop retains repeated UI maps and offers. Neither difference
proves the cause of a particular wrong model action.

Additional identical-observation encoding in the reference working tree is
uncommitted and is not treated as demonstrated behavior. Any local history
projection must preserve the newest observation, immutable receipts, raw traces,
accepted speech and execution authority. Cached action results cannot become
new observations. The [local live campaign](computer-use-live-validation-2026-10-15.md)
and [remote runs](computer-use-remote-validation-2026-10-15.md) remain the
acceptance evidence for this implementation.
