# Test quality gate: virtual-context V1

Status: **CONDITIONAL PASS**

Passing evidence:

- immutable raw evidence and exact reconstruction;
- deterministic context cap with output headroom;
- 70/70 structured replay and 70/70 oracle preparation cases through 1M;
- zero declared near-duplicate replay on the production baseline;
- meaningful FIFO, dense-only, and recurrent baseline failures;
- one official 256K RULER variable-tracking model run at 100% for hybrid/oracle
  and 0% for FIFO.

Conditions before a full release claim:

- complete the official 13-task RULER FIFO/hybrid/oracle matrix;
- record downstream model answer accuracy, p50/p95 latency, and oracle gap;
- add realistic live long-conversation/document/code-agent distributions;
- keep learned latent/KV branches non-authoritative until their independent
  reconstruction and ablation gates pass.
