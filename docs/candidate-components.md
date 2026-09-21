# Component evaluation notes

These notes retain evaluation constraints for learned or external components.
Laya is installed as a shadow decision plane; other candidates must be
evaluated only after the existing comprehension, tool, TTS, and restoration
gates remain stable. No learned component may become an authority that can
silently refuse, finalize, or redirect user work.

## Laya decision model (integrated in shadow mode)

Candidate: [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)
(Apache-2.0).

Implemented role: a non-generative System-1 scorer for explicit typed
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

The centralized design, lifecycle, decision inventory, replay harness, and
current host measurements are documented in
[`decision-plane.md`](decision-plane.md). The remaining activation gates are:

- Retain the measured multilingual/int8 residency profile on the constrained
  unified-memory host; loading the larger English fp32 path does not preserve
  the generic memory floor.
- Treat the current CPU measurements as shadow telemetry, not a low-latency
  fast path. Re-measure with a supported GPU PyTorch build before activation.
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

Laya remains shadow-only because the first real replay did not meet the
accuracy, calibration, or latency gates. That is an evidence-based deployment
state, not an optional or isolated integration.
