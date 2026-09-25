# Virtual-context master test plan

## Objective

Demonstrate that a replaceable model with an approximately 16K physical window
can operate over lossless 256K-to-1M source history by paging in sufficient,
provenance-bearing evidence without treating derived summaries as source truth.

## Gates

1. Unit/contract gate: append-only storage, offsets, indexes, supersession,
   controller actions, evidence replay, context ceiling, and session isolation.
2. Adversarial preparation gate: ten scenario families at 16K through 1M for
   FIFO, dense, lexical, hybrid, recurrent, recursive replay, structured replay,
   and oracle baselines.
3. Oracle ceiling gate: recursive support plus exact high-information source
   locations must produce a complete bounded evidence pack. Derived answer
   labels are not source locators. An oracle failure is a packer/base-task issue.
4. Production retrieval gate: structured replay must match the oracle evidence
   ceiling, reconstruct all provenance, stay within the input ceiling, and
   exclude scenario-declared near-duplicate distractors.
5. Official RULER gate: all 13 RULER v1 tasks at meaningful sample counts and
   source lengths, scored by the upstream evaluator for FIFO/hybrid/oracle.
6. Live task gate: conversation, documents, and cross-file code work against the
   target Jetson model, including tool loops and persistent constraints.

## Required measures

- exact evidence and replay recall;
- multi-hop, chronology, constraint, numerical-evidence, and symbol coverage;
- provenance and reconstruction fidelity;
- forbidden distractor/conflict handling;
- resident/source tokens and compression ratio;
- retrieval rounds, preparation/inference latency, RAM/VRAM, and index bytes;
- official downstream answer/task score and oracle gap.

Learned latent memories, neural gates, reversible virtual tokens, and KV-level
optimizations are independent experimental gates. They cannot replace immutable
raw evidence until reconstruction and downstream ablations pass.
