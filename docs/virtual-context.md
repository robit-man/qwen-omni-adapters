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
- coverage-pinned exact identifiers/entities plus overlap-aware MMR, so a
  duplicate page cannot evict another explicitly requested key or dependency;
- deterministic corpus-wide word-frequency aggregation with full immutable
  provenance for questions that cannot be answered by sparse top-k retrieval;
- query-time exact-value and assignment-graph compilation: complete relations
  become small verified memories with reconstructable source offsets, while
  partial or conflicting relations fall back to raw evidence;
- query-focused exact-span replay adjacent to the current query;
- a hard context allocator that pins deterministically relevant constraints,
  reserves verified structured memory before evidence expansion, and reserves
  output headroom. Constraint scope comes from explicit global metadata, an
  active subject/entity, or exact task terms; unrelated historical constraints
  are never pinned merely because they exist;
- PAGE_IN/PAGE_OUT/PIN/UNPIN/EVICT/EXPAND/MERGE/SUPERSEDE/RECONSTRUCT telemetry.

This foundation is model-agnostic. Guided Jetson deployment enables `active` mode
after the cache-isolated 16K/256K RULER gate: the portal stores raw conversation and
document text losslessly, then replaces old textual history with a bounded working
set on every initial and tool-follow-up turn. Set `OMNI_VIRTUAL_CONTEXT_MODE=shadow`
for side-by-side diagnostics without prompt replacement, or `off` to disable the
subsystem. Portal Trash removes the entire isolated session corpus together with the
existing session caches.

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
On Jetson, the packer also rereads the comprehension launcher's selected-context
state on every turn. The environment value is a ceiling, not an assumption about
the currently resident KV allocation.

At reduced physical windows, whole-corpus deterministic operations are paged in as
verified structured views with pointers to every contributing immutable chunk.
Arbitrary local source pages are not replayed as if they prove a global aggregate;
the raw corpus remains lossless and selectively expandable.
The same rule applies to multi-key lookups and assignment chains only when every
requested relation resolves from retrieved evidence. The compiler never reads
benchmark references or expected answers.

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

# Causal ablation: lexical retrieval only, one controller pass, and no
# deterministic whole-corpus aggregation or exact-relation compiler.
.venv/bin/python scripts/run_virtual_context_ruler.py /path/to/ruler/jsonl \
  --output-dir /tmp/ruler-bm25-one-pass --baseline hybrid \
  --retrieval-profile bm25-only --controller-rounds 1 \
  --disable-aggregation --disable-compilation \
  --endpoint http://127.0.0.1:8000/v1/chat/completions --model MODEL
```

RULER does not materially exercise the runtime's explicit entity graph, Python
call/import topology, pinned constraints, or versioned decisions. Run the separate
domain answer gate against those paths. A preparation-only run proves retrieval,
provenance, sufficiency, and physical packing; adding an endpoint measures whether the
deployed model can actually use the prepared evidence:

```bash
.venv/bin/python scripts/run_virtual_context_domain_eval.py \
  --output /tmp/domain-preparation.json \
  --source-length 256000 --physical-context 4096

.venv/bin/python scripts/run_virtual_context_domain_eval.py \
  --output /tmp/domain-hybrid.json \
  --source-length 256000 --physical-context 16384 \
  --physical-context-state-file runtime-data/state/comprehension-context-tokens \
  --endpoint http://127.0.0.1:8901/v1/chat/completions \
  --model local-audio-bridge
```

The ten answer tasks cover sparse/multi-source recall, supersession chronology,
numerical operands, a three-hop entity path, cross-file definitions/calls/imports/tests,
a buried negative constraint, a durable user decision, an exact log/request join, and
near-duplicate conflicting entities. Production preparation never reads required or
forbidden evaluator terms. The labelled oracle uses fixture terms only to locate raw
source pages. Reports retain controller/index traces, exact-provenance state, live
physical context, endpoint token usage, latency, compression, and per-term failures.
Use `--retrieval-profile hybrid-no-graph` to causally test the entity/code topology
channels; use `--baseline fifo` and `--baseline oracle` for the resident-tail floor and
evidence-pack ceiling.

The output JSONL preserves the official fields and adds `pred`, so scoring is
performed by NVIDIA RULER's official evaluator. The harness records the pinned RULER
v1 revision and reattaches the generator's separated `answer_prefix`. Its
oracle-assisted condition combines recursive dependency retrieval with exact
locations for high-information references; short derived labels such as yes/no
are never treated as source locators. Endpoint runs default to native
no-thinking mode with `cache_prompt=false`, derive the active llama.cpp
`/tokenize` route, and report the
published all-match or QA partial-match score, prompt/completion tokens, finish
reason, and p50/p95 inference latency in `virtual-context-run.json`. The JSONL
remains suitable for independent upstream evaluation. A preparation-only run
does not produce model predictions and must not be reported as a RULER score.
The `bm25-only`, `dense-only`, `hybrid-no-graph`, and `hybrid` retrieval
profiles plus the controller/aggregation/compiler switches support isolated
ablation without changing or regenerating the sealed benchmark fixture. Every
record and `virtual-context-run.json` retain the selected settings.

### Jetson milestone evidence

The first live stretch run used the Ornith 1.5 audio-bridge model on the 32 GB
Jetson. Although the configured ceiling was 16,384 tokens, the resident launcher
selected 8,192 tokens; every run reread that state and reported
`physical_context_source=resident_state`.

With one official RULER v1 sample from each of its 13 task classes:

| source length | condition | mean task score | prompt tokens | inference p50 / p95 |
|---:|---|---:|---:|---:|
| 512K | hybrid virtual context | 100% | 29,381 | 4.76s / 7.90s |
| 1M | final-window FIFO | 7.69% | 75,658 | 9.74s / 12.35s |
| 1M | hybrid virtual context | 100% | 22,699 | 3.49s / 6.80s |
| 1M | oracle-assisted locator | 100% | 28,345 | 4.86s / 8.23s |

The 1M hybrid run matched the oracle ceiling on this fixture set and used
341x-3,449x source-to-resident compression. Query-time relation compilation was
trained on neither these fixtures nor their answers: it activates only when a
retrieved multi-key lookup or assignment graph resolves completely, retains exact
source offsets, and falls back to replayed raw evidence on ambiguity. The benchmark
references never enter the production retrieval query. The corresponding report
hashes and per-task scores are recorded in
`.aiwg/testing/evidence/ruler-v1-512k-1m-jetson.json`.

This is milestone evidence, not a claim of general 1M-context equivalence. It is one
generated sample per task class. Multi-seed runs, code/document/conversation suites,
and subsystem ablations remain required for statistical and domain coverage.

A second sealed run used three freshly generated samples from each task class at
256K source length while the complete TTS and pointing stack remained resident. The
live allocator selected a 4,096-token physical window. The final hybrid run answered
all 39 samples and NVIDIA's pinned evaluator scored every task at 100% with no null
predictions. It used 41,872 prompt tokens across the run, 3.03s p50 / 4.13s p95
inference, and 153x-905x source-to-resident compression.

The first pass over that sealed fixture scored 91.03%, which exposed rather than hid
three gaps: exhaustive values for one key, a natural-language bridge entity, and
source-exact/ambiguous answer rendering. Oracle reruns were limited to those failures
to classify them. The resulting fixes operate only on the query and immutable source:
bounded exact-key corpus scans fail closed on saturation, natural bridge expansion is
limited to two candidates and one round, and ambiguous answers retain all qualifying
source-exact spans. The unchanged fixture then passed in full. Checksums, initial
failures, oracle diagnostics, the evaluator compatibility shim, and final per-task
scores are recorded in
`.aiwg/testing/evidence/ruler-v1-256k-seed9137-jetson.json`.

This adds held-out positions and more examples but is still one seed with only three
samples per class. More independent seeds, length sweeps, domain suites, and ablations
remain necessary.

The same sealed fixture now supports explicit causal ablations. A diagnostic run used
the first sample from every task, kept the physical window at 4,096 tokens, and left
evidence-insufficient cases unanswered:

| condition | mean task score | answered / sufficient | prompt tokens |
|---|---:|---:|---:|
| full V1 | 100% | 13 / 13 | 13,848 |
| full V1, no graph channels | 100% | 13 / 13 | 13,849 |
| full V1, one controller round | 95.38% | 13 / 13 | 13,712 |
| no frequency aggregation | 84.62% | 12 / 12 | 14,432 |
| no exact-relation compilation | 87.69% | 13 / 13 | 17,568 |
| recursive hybrid raw replay | 72.31% | 12 / 12 | 18,163 |
| one-pass hybrid raw replay | 69.23% | 11 / 11 | 16,507 |
| one-pass BM25 raw replay | 61.54% | 10 / 10 | 15,308 |
| one-pass hashing-dense raw replay | 23.08% | 8 / 8 | 9,668 |

On this slice, recursion contributes 4.62 points to the otherwise complete system,
aggregation owns the two frequency tasks, and exact-relation compilation owns most of
the multi-value/query and variable-chain gap. Removing graph channels has no effect;
their value must be measured on code and entity-topology suites rather than inferred
from RULER. The current hashing embedder is an auxiliary recall channel, not a viable
standalone dense retriever. Exact condition manifests, non-perfect task scores, and
report hashes are recorded in
`.aiwg/testing/evidence/ruler-v1-256k-ablation-seed9137-jetson.json`.

The topology-rich domain suite was then executed against a separate deterministic
256K corpus at the same live 4,096-token physical window:

| condition | tasks passed | answered / sufficient | prompt tokens | inference p50 / p95 |
|---|---:|---:|---:|---:|
| hybrid virtual context | 10 / 10 | 10 / 10 | 4,736 | 4.19s / 22.42s |
| labelled oracle pages | 10 / 10 | 10 / 10 | 3,943 | 4.36s / 13.08s |
| final-window FIFO | 0 / 10 | 10 / 10 | 17,230 | 7.23s / 11.07s |

Hybrid matched the oracle ceiling at 307x-1,185x source-to-resident compression.
Every hybrid/oracle answer retained exact provenance and passed a second sufficiency
check over the pages that actually survived packing. This final check is also in the
production portal engine: a promising retrieval candidate that is later evicted can
no longer authorize generation.

The entity-graph task has an isolated causal control. With graph channels, one
retrieval round paged the three-hop terminal fact and passed. Without graph channels,
the one-pass condition was evidence-insufficient and made no inference call. Recursive
no-graph retrieval recovered through an exact bridge lookup in two rounds and passed,
but took 10.69s versus 5.11s for graph-backed one-pass execution. Full reports,
predictions, traces, source hash, and report hashes are recorded in
`.aiwg/testing/evidence/virtual-context-domain-256k-jetson.json`.

This suite is a development fixture, not held-out population evidence. It establishes
that the implemented topology, constraint, chronology, conflict, and code-symbol paths
work end to end on the deployed model; additional randomized domain seeds remain
required for confidence intervals.

The production portal was separately exercised after enabling the fully co-resident
TTS and pointing stack. Its launcher selected a stricter 4,096-token physical window.
An authenticated 1,748,926-character conversation with three exact values buried among
40,000 events was reduced to a 3,164-token working allocation; the live Ornith model
returned all three identifiers and values exactly in 7.17 seconds. The synthetic
session corpus was then deleted through the normal Trash path. This gate also verifies
that policy remains in the system role while retrieved memory, exact evidence, and the
real current query are presented together in the user working set. Evidence is recorded
in `.aiwg/testing/evidence/virtual-context-portal-4k-jetson.json`.

## Authority rules

- Raw source is authoritative and immutable inside its corpus.
- Exact replay outranks derived memory.
- Unresolvable derived content is marked unverified.
- Compression generation reduces selection authority.
- Conflicts remain separate until provenance or supersession resolves them.
- Task-relevant constraints/current plans are deterministic pins, not similarity retrieval.
- Answer authority is recomputed after packing against the final resident evidence set.
- Pinned overflow stops generation and asks for narrower scope.

## Experimental branches

Learned Lychee-style gates, R³Mem-style reconstruction, latent buffer compilation,
KIVI, PyramidKV, Quest, and StreamingLLM support belong behind optional backend
interfaces. None may become the sole source representation without reconstruction,
oracle, and downstream fidelity evidence.

See the [architecture ADR](../.aiwg/architecture/decisions/ADR-001-virtual-context-hierarchy.md)
and [research synthesis](../.aiwg/research/reports/long-context-virtualization-findings.md).
