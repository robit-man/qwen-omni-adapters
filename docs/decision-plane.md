# Laya System-1 decision plane

The runtime has a first-class `DecisionPlane` between normalized state and the
deliberative model. Application modules consume typed `DecisionResult` values;
only the isolated loopback worker imports Laya. Prediction, authorization, and
execution remain separate layers.

## Before and after

```text
BEFORE
input -> LLM -> tool -> LLM judges result -> tool/finish
                    ^                         |
                    +-------------------------+

AFTER
input -> deterministic normalization/provenance
      -> one batched Laya decision wave (shadow until calibrated)
           | high-confidence, calibrated, reversible route
           +-----------------------------------------------> direct hint
           | uncertain / open-ended
           +-----------------------------------------------> deliberative LLM
      -> deterministic policy authorization
      -> typed tool executor
      -> normalized observation
      -> one batched post-action wave -> continue/escalate/finish signal
```

Exact validation, duplicate hashes, resource arithmetic, authority, and policy
remain deterministic. Laya is used only for bounded judgments. The LLM retains
open-ended interpretation, planning, synthesis, novel arguments, debugging,
and final wording. Human or policy gates cannot be bypassed by confidence.

The complete A/B/C/D/E classification and lifecycle inventory is in
[`decision-topology.md`](decision-topology.md).

## Decision waves

Questions are versioned in `src/qwen_omni_adapters/decision-plane.yaml`; raw
question text does not live in call sites.

| Wave | One state snapshot, one forward pass |
|---|---|
| `input_routing` | request kind, work shape, retrieval disposition, hierarchical tool family, risk signal |
| `pre_action` | goal consistency and requested scope |
| `post_action` | observed outcome, retry/continue disposition, completion state |
| `context_relevance` | reversible relevance and duplicate filtering signals |

The foreground portal, adapter after media comprehension, and durable worker
submit these waves asynchronously in shadow mode. The worker rejects concurrent
overflow within the configured `max_wait_ms`; callers immediately retain the
deliberative fallback instead of accumulating a model-inference queue.

## Lifecycle and residency

`runtime/laya_server.py` owns checkpoint download, device selection, preload,
warmup, inference serialization, input bounds, health, and metrics. The daemon
starts it before the adapter and portal, waits for a representative warm pass,
then publishes `OMNI_DECISION_PLANE_ENABLED=1` to children. Missing packages,
memory pressure, invalid output, timeout, and worker failure all escalate to
the existing path and do not stop the agent.

The default edge profile keeps only the 322M multilingual checkpoint resident
on CPU; this also prevents an optional optimizer from bypassing the host CUDA
lease broker.
It covers English and non-English input, has a 1024-token checkpoint context,
and uses dynamic int8 to retain the generic host memory floor. Operators can
change device, checkpoint, preload,
quantization, timeout, and residency through the YAML file or documented
environment overrides. Selecting CUDA is valid only after the deployment has
acquired that exact device through the host broker; `auto` is intentionally not
the portable default.

Health is available in the normal portal health response and directly on the
loopback worker:

```bash
curl -fsS http://127.0.0.1:8930/health
```

It reports readiness, checkpoint, package/model version, device, quantization,
warmup time, inference/failure/busy counts, last latency, uptime, and current
memory headroom. Decision JSONL traces contain hashes and typed outputs, not
raw user state.

## Caching, fallback, and authorization

The cache key includes normalized immutable state, question wording/version,
checkpoint/package version, preload set, policy version, and shadow state.
Changing any of these invalidates cached output. Cache hits pass through the
same metrics and trace path as fresh decisions.

The application may use a result only when all of the following hold:

1. the decision family has real workload calibration;
2. its configured per-family threshold is met;
3. shadow mode is off for that family;
4. deterministic validation succeeds;
5. policy permits the operation; and
6. any required human confirmation is present.

Otherwise the result escalates. A prediction never authorizes or executes a
tool itself.

## Replay and calibration

The replay harness compares historical or labelled traces with the current
path and produces machine-readable JSON plus a Markdown report:

```bash
OMNI_DECISION_PLANE_URL=http://127.0.0.1:8930 \
  .venv/bin/python scripts/replay_decision_plane.py trace.jsonl \
  --output-json runtime-data/laya-replay.json \
  --output-report runtime-data/laya-replay.md
```

It calculates accuracy, per-label precision/recall, confusion matrices, Brier
score, ECE, threshold coverage/error, false fast paths, LLM calls avoided,
task-success estimate, and p50/p95/p99 latency. Fixture-only data can exercise
the machinery but cannot activate a fast path.

### Current constrained-host result

The 2026-09-20 real-checkpoint shadow replay used 28 ASR-style traces spanning
file creation/editing, code/build/test work, browser navigation/forms/search,
web/document/memory/system/camera routing, consequential actions, and
post-action completion/retry states.

| Metric | Result |
|---|---:|
| Model / device | Laya 0.3.4 multilingual / CPU dynamic int8 |
| Warmup | 841 ms |
| Decision wave p50 / p95 / p99 | 2645 / 2808 / 3116 ms |
| Fast-path coverage | 0% (shadow) |
| Escalation | 100% |
| LLM calls before / after | 61 / 61 |
| Task success before / estimated after | 100% / 100% |
| End-to-end latency change | 0% because shadow waves are asynchronous |
| Best fixture accuracy | 60% for tool family and work shape |

These results prove integration and graceful coexistence, but they do **not**
justify an active fast path. This host has CPU-only PyTorch, so the measured
wave is not millisecond-scale, and generic-checkpoint accuracy is insufficient.
The next activation step is to collect real shadow traces, then evaluate a
GPU-capable build or a specialist checkpoint against per-family false-fast-path
budgets. Until then Laya remains a visible, replayable optimizer that cannot
change production behavior or add critical-path latency.
