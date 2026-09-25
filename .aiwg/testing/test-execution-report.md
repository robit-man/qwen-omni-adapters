# Virtual-context test execution report

- Date: 2026-09-25
- Resident budget: 16,384 tokens
- Matrix sizes: 16K, 32K, 64K, 128K, 256K, 512K, 1M
- Matrix cases: 560 preparation cases
- Result: adversarial preparation gate and one-sample 32K RULER breadth gate passed

Structured replay and the reference-location oracle each passed 70/70 scenario
instances with 1.000 evidence recall, replay recall, and exact-provenance
accuracy. Immutable reconstruction was 63/63 exact spans. The structured path
replayed zero declared near-duplicate distractors and never exceeded its 14,000
input-token ceiling.

The weaker baselines remained visible: FIFO passed 15.7%, dense-only 90%, and
bounded recurrent text 80%. These failures are evidence that the gate is not a
hidden-answer or always-green fixture. See `virtual-context-v1-evidence.md` for
the per-length resource measurements and the bounded live RULER result.

The live Jetson breadth sweep used one official 32K sample from each of RULER
v1's 13 tasks. Hybrid and dependency-aware oracle-assisted conditions scored
100% mean; the exact-token FIFO control scored 39.23%. Hybrid used 73,026 prompt
tokens across the sweep versus FIFO's 182,151, with 1.38-second versus
21.49-second median inference latency. The transformer remained capped at
16,384 tokens and all endpoint runs used the active tokenizer and native
no-thinking branch. Detailed per-task hashes and metrics are retained in
`evidence/ruler-v1-32k-jetson.json`.

Meaningful multi-sample 64K-256K RULER runs and broad live downstream task
evaluation remain open. Therefore this report supports the training-free V1
path and first task-breadth milestone, not native-256K equivalence or release of
learned compression branches.
