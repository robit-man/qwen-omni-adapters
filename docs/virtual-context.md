# Virtual context

Compact bridge deployments use a 16,384-token resident transformer window as a
working set over a larger lossless corpus. This is external memory, not RoPE scaling:
raw history remains in SQLite and selected exact evidence is paged into the prompt.

## Current V1 foundation

The `qwen_omni_adapters.virtual_memory` package provides:

- immutable document/chunk storage with exact token, character, and byte offsets;
- structure-aware Python, Markdown, conversation, JSON, and log chunking;
- BM25, always-available hashed dense, exact string, symbol, metadata, recency,
  entity, and graph retrieval over the same chunks; the deterministic dense encoder
  is a fuzzy lexical fallback, and a learned semantic embedder can replace it through
  the same interface;
- provenance-bearing facts, constraints, decisions, entities, relationships,
  episodes, open questions, current plans, and optional latent-memory records;
- supersession without silent overwrite;
- recursive PRETHINK/RETRIEVE/WRITE/ANSWER/STOP control;
- deterministic corpus-wide word-frequency aggregation with full immutable
  provenance for questions that cannot be answered by sparse top-k retrieval;
- exact evidence replay adjacent to the current query;
- a hard context allocator that pins active constraints and reserves output headroom;
- PAGE_IN/PAGE_OUT/PIN/UNPIN/EVICT/EXPAND/MERGE/SUPERSEDE/RECONSTRUCT telemetry.

This foundation is model-agnostic. Guided Jetson deployment enables `shadow` mode:
the portal dual-writes raw conversation/extracted document text and builds observable
working sets, but does not change live prompts. Set `OMNI_VIRTUAL_CONTEXT_MODE=active`
only for controlled end-to-end validation. Portal Trash removes the entire isolated
session corpus together with the existing session caches.

## Minimal use

```python
from pathlib import Path

from qwen_omni_adapters.virtual_memory import (
    ContextBudget,
    HashingEmbedder,
    HybridRetriever,
    ImmutableEvidenceStore,
    RecursiveMemoryController,
    VirtualContextEngine,
    WorkingContextPacker,
)

embedder = HashingEmbedder()
store = ImmutableEvidenceStore(
    Path("runtime-data/virtual-context.sqlite3"), embedder=embedder
)
retriever = HybridRetriever(store, query_embedder=embedder)
controller = RecursiveMemoryController(retriever)
packer = WorkingContextPacker(budget=ContextBudget(max_tokens=16_384))
engine = VirtualContextEngine(store, controller, packer)

engine.ingest("lossless source text", source="conversation:message-1")
turn = engine.prepare_turn(
    "What did the source say?",
    system_contract="Answer from replayed evidence.",
)
```

Production injects the active llama.cpp model's `/tokenize` endpoint through
`OMNI_VIRTUAL_CONTEXT_TOKENIZE_URL`; guided Jetson deployment configures the local
comprehension worker automatically. The built-in conservative counter remains an
offline/test fallback and cannot guarantee byte-identical counts for every tokenizer.

## Benchmark

Run the deterministic adversarial matrix first:

```bash
.venv/bin/python scripts/benchmark_virtual_context_matrix.py \
  --output /tmp/virtual-context-matrix.json
```

It holds the resident transformer budget at 16,384 while testing 16K, 32K,
64K, 128K, 256K, 512K, and 1M source-token fixtures. Ten scenario families
cover sparse and multiple needles, supersession/chronology, numerical evidence,
multi-hop entity traversal, cross-file code topology, a buried constraint,
user decisions, exact log strings/citations, and adversarial near-duplicate
entities. The report compares FIFO, dense-only RAG, lexical-only RAG, hybrid
RAG, bounded recurrent text, recursive exact replay, structured replay, and a
labelled oracle. Expected terms locate oracle chunks and score completed packs;
they are never added to production retrieval queries.

The matrix is a memory-preparation gate. It measures evidence recall, replay
recall, exact provenance, forbidden distractor replay, controller sufficiency,
resident tokens, retrieval rounds/latency, index/RAM growth, compression ratio,
and reconstruction fidelity. It does not award numerical reasoning or model
answer accuracy merely because operands reached the prompt. Use official RULER
and task-level model evaluation for those downstream claims.

The older one-fact sparse ladder remains useful as a fast diagnostic:

```bash
.venv/bin/python scripts/benchmark_virtual_context.py
.venv/bin/python scripts/benchmark_virtual_context.py --lengths 512000,1000000
```

It compares a final-16K FIFO baseline, hybrid retrieval, and an oracle evidence
pack. It measures memory-subsystem fidelity only and does not substitute for the
adversarial matrix, model-answer evaluation, or RULER.

For model-answer evaluation, generate official RULER v1 JSONL with the upstream
tooling, then prepare or execute predictions without loading the full source into the
transformer:

```bash
.venv/bin/python scripts/run_virtual_context_ruler.py /path/to/ruler/jsonl \
  --output-dir /tmp/ruler-hybrid --baseline hybrid

.venv/bin/python scripts/run_virtual_context_ruler.py /path/to/ruler/jsonl \
  --output-dir /tmp/ruler-oracle --baseline oracle \
  --endpoint http://127.0.0.1:8000/v1/chat/completions --model MODEL
```

The output JSONL preserves the official fields and adds `pred`, so scoring is
performed by NVIDIA RULER's official evaluator. The harness records the pinned RULER
v1 revision and reattaches the generator's separated `answer_prefix`. Its
oracle-assisted condition combines recursive dependency retrieval with exact
locations for high-information references; short derived labels such as yes/no
are never treated as source locators. Endpoint runs default to native
no-thinking mode, derive the active llama.cpp `/tokenize` route, and report the
published all-match or QA partial-match score, prompt/completion tokens, finish
reason, and p50/p95 inference latency in `virtual-context-run.json`. The JSONL
remains suitable for independent upstream evaluation. A preparation-only run
does not produce model predictions and must not be reported as a RULER score.

## Authority rules

- Raw source is authoritative and immutable inside its corpus.
- Exact replay outranks derived memory.
- Unresolvable derived content is marked unverified.
- Compression generation reduces selection authority.
- Conflicts remain separate until provenance or supersession resolves them.
- Active constraints/current plans are deterministic pins, not similarity retrieval.
- Pinned overflow stops generation and asks for narrower scope.

## Experimental branches

Learned Lychee-style gates, R³Mem-style reconstruction, latent buffer compilation,
KIVI, PyramidKV, Quest, and StreamingLLM support belong behind optional backend
interfaces. None may become the sole source representation without reconstruction,
oracle, and downstream fidelity evidence.

See the [architecture ADR](../.aiwg/architecture/decisions/ADR-001-virtual-context-hierarchy.md)
and [research synthesis](../.aiwg/research/reports/long-context-virtualization-findings.md).
