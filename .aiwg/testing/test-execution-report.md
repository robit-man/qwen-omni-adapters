# Virtual-context test execution report

- Date: 2026-09-25
- Resident budget: 16,384 tokens
- Matrix sizes: 16K, 32K, 64K, 128K, 256K, 512K, 1M
- Matrix cases: 560 preparation cases
- Result: adversarial preparation/oracle gate passed

Structured replay and the reference-location oracle each passed 70/70 scenario
instances with 1.000 evidence recall, replay recall, and exact-provenance
accuracy. Immutable reconstruction was 63/63 exact spans. The structured path
replayed zero declared near-duplicate distractors and never exceeded its 14,000
input-token ceiling.

The weaker baselines remained visible: FIFO passed 15.7%, dense-only 90%, and
bounded recurrent text 80%. These failures are evidence that the gate is not a
hidden-answer or always-green fixture. See `virtual-context-v1-evidence.md` for
the per-length resource measurements and the bounded live RULER result.

The official 13-task RULER matrix and broad live downstream task evaluation
remain open. Therefore this report supports the training-free V1 preparation
layer, not native-256K equivalence or release of learned compression branches.
