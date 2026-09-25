# Virtual-context V1 impact analysis

## Change boundary

V1 introduces an isolated `qwen_omni_adapters.virtual_memory` package and a
benchmark script. It does not replace the portal's privacy-scoped cache or passive
memory in this change. Compact Jetson bridge profiles now default to a 16K physical
context ceiling.

## Components added

- `models.py`: evidence, memory, retrieval, and working-context records.
- `chunking.py`: Python/Markdown/conversation/JSON/log structural chunking with
  a 512–2K-class overlapping fallback configurable by token budget.
- `store.py`: immutable evidence, FTS5, dense vector rows, symbols, entities,
  relationships, superseding memories, and reconstruction.
- `retrieval.py`: query planning, hybrid union, reranking hook, diversity/source caps.
- `controller.py`: explicit recursive controller and sufficiency loop.
- `extractor.py`: conservative user-authored constraint/decision/fact promotion with
  exact provenance and supersession.
- `embedding.py`: weight-free deterministic dense fallback and replaceable interface.
- `tokenization.py`: exact active-model token counts through llama.cpp `/tokenize`.
- `recurrent.py`: bounded recurrent memory with periodic raw-evidence regeneration.
- `packer.py`: hard context/envelope accounting, pinning, content-aware eviction,
  and exact replay.
- `engine.py`: inference-backend-neutral façade.
- `telemetry.py`: bounded memory-operation trace.
- `scripts/benchmark_virtual_context.py`: oracle/FIFO/hybrid sparse benchmark.
- `scripts/run_virtual_context_ruler.py`: scorer-compatible RULER v1 preparation and
  OpenAI/Ollama endpoint execution for FIFO, hybrid, and oracle-location baselines.

## Existing components affected

`model_catalog.py`, `daemon.py`, and `deploy.sh` change only the compact bridge
default/advertised physical context ceiling from 262,144 to 16,384. An explicit
environment override remains possible for controlled experiments.

The portal now has off/shadow/active modes. Shadow is the guided-deployment default;
active replaces history with the bounded pack and repacks after every tool result.
No public adapter request/response schema, model weights, or Ollama manifests change.
No CUDA workload is started by this development change.

## Persistence and privacy

Evidence is immutable inside one corpus database. Corpus lifecycle is a higher-level
policy: a browser-session corpus may still be destroyed as a unit by Trash/TTL, while
a user-authorized durable history corpus may persist. Individual source rows must not
be silently rewritten or summarized away.

SQLite WAL permits concurrent readers and one writer without another resident service.
Index growth is measured in benchmark output. Backup must copy the database plus WAL
consistently or use SQLite backup APIs.

## Failure modes

- FTS5 missing: store initialization fails; do not silently downgrade to vector-only.
- Learned embedder unavailable: the local hashed dense fallback remains available
  alongside lexical/exact/symbol/entity/graph paths.
- Exact tokenizer unavailable: shadow mode records a bounded error and leaves the
  live prompt unchanged; active mode fails closed.
- Retrieval insufficient: controller marks answer unresolved after its bounded budget.
- Pinned state overflow: packer raises; it does not evict an active invariant.
- Bad reranker cardinality: retrieval fails closed with a clear error.
- Broken provenance: derived memory write is rejected.
- Latent artifact mismatch: future backend must reject differing model/compiler hashes.

## Migration strategy

Use a strangler rollout. First dual-write raw turns and attachments, then shadow-build
working contexts and compare them with current prompts. Only after fidelity and privacy
tests pass should virtual packs become the generation input. Existing session tools stay
available during migration.

## Test impact

The focused suite covers structural offsets, database immutability, hybrid channels,
graph traversal, supersession, recursive retrieval, exact replay adjacency, hard pinning,
and a constraint surviving more than 100K intervening tokens.

The suite also covers Markdown/JSON/log/conversation paths, conflicting memories,
class-specific TTL defaults, recursive code/relationship topology, raw regeneration,
exact-tokenizer behavior, feature-flagged portal integration, and bounded tool-loop
repacks. Official model-answer RULER runs and concurrency/soak measurements remain
release gates, not completed claims.

## Operational impact

The 16K default reduces worst-case resident KV pressure on unified-memory Jetsons and
prevents the launcher from returning to a previously unsafe 32K/256K tier simply because
the same stack was temporarily unloaded. It does not itself prove enough headroom for
every workload; the existing live admission and GPU-residency checks remain mandatory.

## Rollback

The new package is isolated and can be disabled without migrating existing portal state.
The physical context default can be overridden with
`OMNI_COMPREHENSION_CONTEXT_TOKENS` for a controlled rollback/benchmark. Do not delete
virtual-memory databases until source retention requirements have been confirmed.
