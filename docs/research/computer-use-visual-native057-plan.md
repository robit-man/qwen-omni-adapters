# CU.VISUAL.NATIVE.057 — exact backend grammar compatibility

The 056 sampler failure is now isolated. The actual backend printed a grammar
containing `char{1,2000}` for the evidence field. Its pinned parser rejects a
repetition whose generated-rule cost reaches 2,000. The exact full printed
grammar and a separate schema-converter/parser reproduction fail with the same
error. Changing only that repetition to 1,999 parses successfully. Nullable
labels are not the cause. Bounds above 2,000 are unsuitable because this parser
silently removes that upper bound.

The isolated `/tmp/qwen-native057-20261017` candidate changes only the native
assessment schema's evidence limit from 2,000 to 1,999 characters, plus an
explanatory comment. This output subset satisfies the unchanged host limit of
2,000. All action guards, fields, enums, prompts, images, target/point queries,
native profile and original twelve semantic oracles remain unchanged. No worker
reload or public protocol change is involved.

Both schemas exported from the actual new module pass the pinned native
converter/parser. The original full server grammar remains RED; its single-bound
variant is GREEN. The 559-file candidate is frozen at inventory
`33c99b0e80f7d290da8667745c30650f355338d7980cf61e13a99a2ef36b17e6`.
Its full required gate passed: 5,524 tests, with all 559 source files unchanged.
The independent review also passed. The original twelve cases completed once
against the resident GPU1 worker. All twelve replies were structured, but only
eight passed semantic checks and three incorrectly admitted a semantic gate.
The native parser compatibility criterion passes; desktop admission remains
blocked by the separate semantic failures. See
`computer-use-visual-native057-results.md`.

After that gate and independent review pass, repeat the original twelve cells
once through the same resident GPU1 worker with fresh broker admission checks.
Capture actual request/response bodies and independently grade structure and
semantics. Any unsupported or wrong verdict remains failed; no prose salvage,
guard relaxation or reuse of old failed-run budgets is permitted.

Only a passing model panel permits the one bounded physical desktop diagnostic.
The original Qwen/Ornith long tasks, complete visual operations, breadth matrix,
main publication and both Eggs remain outstanding. Whole-task passes remain zero.

Evidence:

- Exact backend grammar/log:
  `/home/roko/Documents/qwen-omni-research/2026-10-17/visual-assessment056-independent-controls-20261017/native-grammar-error-evidence/`.
- Source-isolation controls and actual full-grammar RED/GREEN:
  `/home/roko/Documents/qwen-omni-research/2026-10-17/visual-assessment056-grammar-diagnosis-20261017/`.
- Actual new module schema compatibility:
  `/home/roko/Documents/qwen-omni-research/2026-10-17/native057-compatibility-review-20261017/receipt.json`
  (SHA256 `8ed49986b0420bc97b0e95c8ad505a6feb6d79d0dfd974f402d4cc72e20436f5`).
- Independent pre-edit native compatibility acceptance:
  `/home/roko/Documents/qwen-omni-research/2026-10-17/native-schema-compatibility-independent-acceptance-20261017/acceptance.json`
  (SHA256 `8032e385e57810807586ead08b926bdb07198ff410450b9a44dc25c15ac7841c`).
