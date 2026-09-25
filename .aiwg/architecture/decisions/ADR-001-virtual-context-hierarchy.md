# ADR-001: Lossless virtual context above a bounded transformer window

- Status: Accepted for V1 foundation
- Date: 2026-09-25
- Owners: qwen-omni-adapters runtime

## Context

Compact Jetson deployments share one memory pool among the model, KV cache,
vision, pointing, TTS, the desktop, and ordinary host work. Raising the model's
advertised context to 256K does not guarantee useful 256K reasoning and causes
the KV allocation to compete with capabilities that must remain resident.

The existing memories are unsuitable as the source of truth:

- portal memory is a small session-scoped key/value cache;
- passive semantic memory merges and eventually decays entries;
- background-task compaction intentionally discards old message detail;
- document retrieval is in-memory, fixed-chunk, and TTL-bounded.

Research reviewed in the linked findings supports bounded recurrent state,
active retrieval, query-time construction, exact evidence replay, and separate
physical-KV optimization. It also documents overwrite and primacy failures when
recurrent memory is the only representation.

## Decision

Implement virtual context as a model-agnostic layer above the inference endpoint.
The default physical working window for compact audio-bridge profiles is 16,384
tokens. A larger native positional range is not the deployment default.

The hierarchy is:

| Level | Role | Authority | Lifetime |
|---|---|---|---|
| L0 | active attention and KV | exact resident prompt | one inference/stream |
| L1 | pinned system, constraints, current plan, recent state | exact or verified | task/turn |
| L2 | recurrent and structured memories, optional latent blocks | derived | versioned |
| L3 | query-specific retrieved evidence cache | exact source spans | one retrieval run |
| L4 | indexed immutable raw history | source of truth | append-only |
| L5 | optional cold archive | source of truth | policy controlled |

Movement is explicit and observable: PAGE_IN, PAGE_OUT, PIN, UNPIN, EVICT,
EXPAND, MERGE, SUPERSEDE, and RECONSTRUCT.

V1 uses SQLite WAL storage and FTS5 with no new service dependency. Document and
chunk rows have database triggers that reject update/delete. Derived memories may
be superseded, but previous versions and provenance remain queryable.

Every source chunk records document/message/source identity, timestamp/version,
token/character/byte offsets, parent structure, neighbor IDs, content hash, and
verbatim text. Every derived memory requires at least one source pointer.

Retrieval unions lexical BM25, deterministic hashed dense vectors, exact strings,
code symbols, metadata, recency, entities, and controlled graph expansion. A learned
semantic embedder can replace the weight-free dense fallback through the same
interface. Candidate selection uses channel fusion, optional reranking, source caps,
and MMR-style diversity.

The controller exposes PRETHINK, RETRIEVE, WRITE, ANSWER, and STOP. Retrieval may
recurse through discovered dependencies. The final pack replays exact evidence
immediately before the query and marks summaries as derived.

The 16K budget is allocated by role. Active constraints/current plan are pinned;
recent conversation is evicted before verified evidence. If active pinned state
cannot fit, generation fails with a scope error rather than silently forgetting it.
Production uses the resident llama.cpp `/tokenize` endpoint and reserves the live
tool/control envelope plus chat-template and output headroom on every initial and
tool-follow-up pack.

## Consequences

Positive:

- Source information survives compression and recurrent rewrites.
- Retrieval failures can be separated from base-model failures with oracle packs.
- Evidence, citations, chronology, and old superseded values remain recoverable.
- The model trunk and inference server remain replaceable.
- Jetson capacity is reserved for co-resident perception and speech instead of a
  nominal 256K KV cache.

Costs:

- Ingestion, indexing, retrieval, and reranking add RAM, disk, and latency.
- The weight-free dense fallback improves fuzzy matching but is not equivalent to a
  learned semantic encoder; that remains a pluggable, separately measured upgrade.
- A deterministic controller is weaker than the trained InfMem policy.
- Current code must be integrated gradually with portal/session retention rules;
  immutable within a corpus does not override an explicit user Trash policy for
  deleting the entire session corpus.

## Rejected alternatives

### Make 256K the default with RoPE scaling

Rejected as the core design. It increases resident KV and accepted length without
proving usable reasoning or protecting Jetson headroom.

### Replace raw history with a rolling summary

Rejected. It creates recursive degradation and makes exact recovery impossible.

### Use vector RAG alone

Rejected. It is weak for exact identifiers, error strings, chronology, code symbols,
conflicts, and graph dependencies.

### Ship latent compression as the sole memory

Rejected until reconstruction and downstream fidelity are independently established.

## Rollout

1. Land the isolated store/retriever/controller/packer and deterministic tests.
2. Add oracle and hybrid benchmark harnesses under a hard 16K cap.
3. Integrate read-only context preparation behind a feature flag.
4. Dual-write source turns/documents without changing current prompt behavior.
5. Compare current versus virtual packs in shadow telemetry.
6. Enable exact replay for selected sessions, then recursive retrieval.
7. Add trained controller/latent/KV experiments only after V1 fidelity stabilizes.

## Verification

- Database immutability and exact offset reconstruction.
- Persistent constraints after 100K to 1M distractor tokens.
- Exact values, chronology, conflicts, supersession, and code topology.
- Oracle versus production evidence packs.
- RULER with a pinned prompt/scoring revision.
- Context total never exceeds 16,384 including the live request envelope and reserved
  output headroom, counted with the active model tokenizer.
- Full project validation and Jetson memory/residency checks before deployment.

## References

See `.aiwg/research/reports/long-context-virtualization-findings.md` and
`.aiwg/research/source-manifest.json`.
