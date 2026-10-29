# 076: Control-name previews did not resolve navigation

The bounded actual-weight comparison failed its acceptance criterion. Qwen made
five new first-stage language calls on the existing leased GPU1 workers in
115.274 seconds. Three of five decisions were useful; both navigation cases
still chose the already-active file-manager window. Do not adopt this candidate
or launch its prepared whole-task run.

## Frozen change and validation

Only `portal/ui_control_catalogue.py` and its existing dedicated test changed
from the 582-file074r3/075r2 baseline. Exact accessible-name prefixes were added
to existing role groups, in source order with duplicates preserved. References,
counts, schemas, histories, selection authority and execution guards were
unchanged. The complete catalogue remained within4096 UTF8 bytes and16 entries;
per-group name arrays were bounded to256 bytes and each prefix to120 bytes.

The exact candidate inventory is
`2ce7fc015a46450052f1b32a6e5c624834c9f90698b50196b73c9dea7dad89b3`.
The full gate passed5,230 tests in308.48 seconds, preserving all582 source files.
Independent source controls and model-method review found no concrete blocker.
Full gate attestation SHA256:
`aedb6bc10a2f7db9ef05dcb734cfe0eb3e1a402b27a50ebeac4aeb74f089c594`.

## Actual model outcomes

| Frozen case | Immediate decision | Useful |
| --- | --- | --- |
| C01 | Open Input from Workbench | Yes |
| H01 | Switch from task-brief editor to Input | Yes |
| H02 | Switch from draft-3 editor to Input | Yes |
| C02 | Select active Input window c0; describe later navigation in prose | No |
| C03 | Select active Input window c0; describe later navigation in prose | No |

Each response was HTTP200, source-bound, schema-valid and without a thinking
channel. Actual prompt tokens were8588/16007/25172/26661/25694 against65536
context,4096 output and32 boundary headroom. No payload shedding was permitted.
Both navigation requests included Back/Workbench name previews. Neither chose
`inspect_controls`, so no catalogue continuation or second planner ran. Prose
about future navigation did not bind an actual Back/Workbench operation.

The comparator comprises five byte-preserved **historical** actual074 candidate
responses, with2/5 useful decisions and zero new baseline calls. The new panel's
H02 response is useful, unlike its historical counterpart. These time-separated
observations do not establish a durable causal improvement. The condition
required all three positive controls and both navigation cases to be useful;
3/5 therefore fails. Preferred target coincidence, valid JSON and discovery
metadata alone never satisfy that condition.

## Supported conclusion and disposition

Adding semantic names at this boundary was insufficient to repair these two
failures. The model correctly describes which window is active yet selects that
same window while leaving the intended navigation in prose. The mismatch
between the stated next operation and its bound immediate action remains the
specific unresolved execution-contract failure. Coarse pointer/activation
semantics and retained history remain hypotheses, not proven root causes.

Freeze this negative result. Do not tune labels, histories, thresholds, schema
order or task budgets in the same condition. The prepared frontend and whole
task remain HELD; no helpers were installed and no service was restarted. The
eight GPU worker lifetimes, lease, GPU1 residency and healthy broker state were
verified before and after the panel. The inference queue was idle afterward.

There were zero GUI inputs, physical effects, whole-task passes or new Ornith/Egg
tests. The separately passing075 stale-run cleanup fixture remains a local
candidate milestone, not production adoption. Unknown old windows remain
preserved. Original long-horizon and cross-device acceptance is still owed.

## Durable evidence

All paths below are under
`/home/roko/Documents/qwen-omni-research/2026-10-17/`:

- `control-name-preview076-model-preparation-20261017/actual-first-20261017/report.json`
  SHA256 `10fe316ce59d65b4878952c423312c34189f3551b8a9e870f4a38dd0e848326d`.
- `control-name-preview076-model-preparation-20261017/actual-independent-review/transport-source-audit.json`
  SHA256 `bf2ba6a608175b515eeec7a36bce1b992130cc50cd6a81cf0c3be63adcef6ddb`.
- `control-name-preview076-model-preparation-20261017/actual-independent-review/final-grade.json`
  SHA256 `5f5e313f8a715a90a2c8979ae5378097df496367bde2db59dddfdd30b784ee2b`.
- `control-name-preview076-root-integration-20261017/model-panel-config.json`
  SHA256 `6ad14e19f5e1c32214829e5b913e8089aac003739e65be7981e34174b4abd5a2`.
- `control-name-preview076-root-integration-20261017/epoch8-after-model-panel.json`
  SHA256 `e1eb27ba807ecdfedb10b81a2a43f0fe949aabb10a3f3f3e9deb14958a0da2a1`.
- `control-name-preview076-whole-task-disposition-20261017/held-disposition.json`
  SHA256 `72ce0c6a98a86d2b222f306e0560a180fa7e98a10d044390f3d6ddc55a1cbbde`.
