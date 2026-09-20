# Deferred component candidates

These are research notes, not installed dependencies or active runtime design.
Evaluate them only after the existing comprehension, tool, TTS, and restoration
gates remain stable. A candidate must not become another authority that can
silently refuse, finalize, or redirect user work.

## Laya decision model

Candidate: [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)
(Apache-2.0).

Possible role: a fast, non-generative System-1 scorer for explicit typed
decisions such as selecting a known route, estimating whether a foreground
turn should remain synchronous, or judging whether observed task state merits
another primary-model step. It should advise scheduling; the primary model and
verified tool evidence must retain ownership of task execution and completion.

Reasons to evaluate:

- one encoder forward pass returns typed choices/scores with probabilities;
- the published English checkpoint is 421M parameters with a 512-token
  context, while the multilingual checkpoint is 322M with a 1024-token
  default context;
- the model card reports roughly 33–40 ms GPU latency for one question and
  supports batching multiple questions in one call;
- it generates no prose, which makes a narrow structured judge interface
  easier to validate than another autoregressive agent.

Constraints and required gates:

- Do not preload it on the constrained unified-memory host until measured
  residency proves coexistence; the model card lists approximately 647–808 MB
  of checkpoint weights, before framework/runtime overhead.
- Benchmark CPU residency and latency first so the fast judge cannot force
  comprehension/TTS eviction or create another reload loop.
- Fit and evaluate the exact local decision schemas. The model card warns that
  base checkpoints are near chance on its typed-decisions benchmark without
  task-specific fine-tuning and that shipped probabilities are over-confident
  before domain temperature calibration.
- Treat its score as advisory. It must never convert uncertainty, memory
  pressure, low confidence, or a predicted refusal category into task
  completion or a reason to avoid available tools.
- Compare end-to-end latency, decision accuracy, calibration, false terminal
  decisions, memory headroom, and interruption behavior against the current
  primary-model-only baseline.
- Preserve exact decision inputs/outputs in bounded diagnostics and expose the
  judge state beside the resulting primary-model/tool action.

No Laya code, weights, service, prompt, or dependency is wired into the runtime
at this checkpoint.
