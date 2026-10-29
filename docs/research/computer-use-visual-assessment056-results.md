# CU.VISUAL.ASSESSMENT.056 — native grammar initialization failed

The frozen 559-file candidate passed the full required gate: 5,524 tests in
303.66 seconds, with source hashes unchanged. Independent source review and
29 CPU boundary controls passed. All existing 055 action/executor guards stayed
unchanged.

The first actual native request was rejected with HTTP 400:
`Failed to initialize samplers: failed to parse grammar`.
The exact request and error response are retained. The run stopped without
retry after one request attempt and no completed model assessment; the other
eleven cases were unattempted. No physical input or service change occurred.

The complete attempt lasted 1.857 seconds. Eight residency samples were
recorded, with one positive GPU1 sample wholly inside the 0.306-second failed
HTTP request. The worker cohort, active lease, all 559 source files and local
logical manifest pin were preserved; postflight remained healthy. Residency
does not make this a completed inference or semantic result.

The sent schema is valid under the general JSON Schema validator. That does
not establish compatibility with this server's JSON-schema-to-GBNF conversion
and grammar parser. The next local gate must reproduce the exact native
conversion and parser failure, identify the offending construct, and validate
the smallest compatible representation before another inference request.
The follow-on CPU and actual-server-log diagnosis now identifies the evidence
field's 2,000-character repetition bound as the cause. Exact RED/GREEN evidence
and the single-bound repair are in `computer-use-visual-native057-plan.md`.
The original failed request and its zero-completion result remain unchanged.

The original 055 failure remains unchanged. No 056 frontend adoption, main
publication, Ornith acceptance or whole-task success occurred. The original
Qwen/Ornith tasks, missing visual operations, breadth matrix and both Eggs
remain required.

Evidence: `/home/roko/Documents/qwen-omni-research/2026-10-17/visual-assessment056-independent-controls-20261017/actual-first-20261017/`.
The failed HTTP response SHA256 is
`0aa5995f7f5ccbf7cf6610c9a053c015fca85199ba8f37736a1c1c584513dd47`.
The independent zero-completion grade SHA256 is
`d5b6c52ded6f1b60774198113a0afbce2f1e6effd88550418d475d3266c9b42f`.
