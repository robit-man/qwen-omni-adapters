# CU.NATIVE.POINTER.072 — measured pre-button read exhaustion

The actual controlled primitive still fails. Exact filename writing and a separate
fresh, blind readback pass; the following moving Cancel operation reaches the
correct button position but does not attempt a button command.

The frozen actual report is
`runtime-data/gpu1-acceptance/operator-audit/execution-sop-20261017/native-pointer072/runs/controlled-first-20261017T193515Z/report.json`,
SHA256 `d08871ee3824c244fd04016c82a9f5f9d6a238559fdabb0bb5f4abd527b09129`.
The added diagnostic records phase `button_activation`, cause
`native_budget_exceeded`, read **12,001/12,000**, with **2,965 ms remaining**.
Movement was attempted and reached Cancel; button, down and release were not
attempted. This identifies the current072 failure, without retroactively assigning
an exact dispatch phase or counter to071's less-instrumented failure.

The same owned dialog remains. Positive editor return and post-return UI buffer
verification were not reached. Separate cleanup restores all15 original windows;
all12 protected process lifetimes, disk bytes/stat/tree and574 source hashes are
preserved. The explicitly preregistered072 setup moved the pointer inside the new
owned editor before opening Save As; that movement ensures Cancel starts elsewhere.
The original solver limits and semantic effect oracles remain unchanged.

Source-only validation:382 focused controls,143 independent controls,110 method
controls and the full5,088-test gate pass. These do not count as model or whole-task
acceptance. Diagnostic observability is verified; the full native primitive fails.
No model call, CUDA lifecycle change, adoption, commit or push occurred.

All574 source files, raw actual evidence, harness, method, gates and independent
review are archived and read back outside runtime-data under
`/home/roko/Documents/qwen-omni-research/2026-10-17/native-pointer072-independent-actual-review-20261017`:
665 files,20,174,299 bytes; review
`c4eceefd04bf1800bd9405f7025d8bfb6645b8d0470cf694beafcfde467339e8`,
manifest `32af7a4b40a93e01f8943c3904073a0337f01d988b9d6eee9254a59400d874f8`,
readback `32bd2cba877267c5541f03c9278c1efff79249d7296d6cb51550826151ce6962`.

The separate073 hypothesis removes only immediate duplicate bus-property reads
from fresh identity checks. All proof coverage and original bounds remain. Its
actual sufficiency is unproven; see the before-edit073 plan. Original local
Qwen/Ornith and Egg whole-task acceptance remains due.
