# Long-context virtualization research findings

Status: implementation-driving synthesis

Date: 2026-09-25
Scope: a lossless virtual-memory harness for a transformer capped near 16K tokens

## Research question

Can a fixed-window model treat its attention/KV window as a high-speed working set
over 256K to 1M+ source tokens without making a lossy summary the only remaining
copy of history?

The reviewed work supports a qualified yes for targeted retrieval and bounded
reasoning workloads. No reviewed paper establishes that 16K attention is generally
equivalent to native 256K attention. The practical path is a hierarchy that keeps raw
evidence externally, retrieves recursively, replays exact spans, and treats learned
compression as an optional acceleration layer.

## Corpus and provenance

The local corpus contains the full PDFs and extracted text for REF-007 through
REF-020. Checksums are recorded in `source-manifest.json`. Findings below are based
on the full papers, not abstracts alone.

Evidence-quality shorthand follows GRADE-like reasoning:

- Moderate: peer-reviewed study with relevant experiments, downgraded when tasks
  are mostly synthetic or implementation portability is uncertain.
- Low: preprint with direct experiments but no completed peer review, often further
  limited by narrow backbones or unpublished code.
- Very low: speculative transfer to this runtime or a result that depends on a
  materially different model architecture.

## REF-007 — MemAgent

Source: [MemAgent](https://arxiv.org/abs/2507.02259)
Quality: Moderate for the bounded recurrent-memory pattern; low for broad real-world
generalization.

MemAgent processes `query + previous memory + next chunk` and rewrites a fixed-size
textual memory after every chunk. Its representative 8K layout gives roughly 1K
tokens to the query, 5K to the current chunk, 1K to memory, and 1K to generation.
The controller is trained with multi-conversation reinforcement learning, sharing the
final answer reward across the memory-update trajectory.

The important reproducible idea is not a larger position encoding. It is recurrent
segment processing whose resident state stays bounded while source length grows.
The paper reports extrapolation far beyond the training length and studies memory
budgets between 256 and 4,096 tokens.

Its failure analysis is more important to this design than its headline score:

- relevant evidence can be appended and later overwritten or truncated;
- an early bridge fact can be discarded before its downstream dependency appears;
- an early wrong interpretation can create primacy bias that survives later evidence.

These failures rule out recurrent text as the sole representation. This project uses
it only as L2 derived memory; every source chunk remains recoverable at L4.

Reproduction decision: benchmark memory budgets 512/1K/2K/4K and chunk sizes
2K/4K/8K, but compare every recurrent run with exact retrieval and reconstruction.

Related sources: REF-008 adds active retrieval and early stop; REF-009 avoids an
expensive generation update for every chunk; REF-011 defers construction to query
time; REF-020 exposes claimed-versus-effective context.

## REF-008 — InfMem

Source: [InfMem](https://arxiv.org/abs/2602.02704)
Quality: Low because it is a recent preprint; the explicit controller decomposition
is nevertheless directly testable.

InfMem augments recurrent scanning with globally indexed fine-grained retrieval. Its
inference loop is PRETHINK, RETRIEVE, WRITE, then ANSWER, with STOP available when
memory is sufficient. PRETHINK compares the question against current memory,
identifies missing facts, chooses a focused retrieval query, and controls top-k.
WRITE jointly compresses the previous memory, current sequential chunk, and retrieved
evidence into the bounded state.

The paper trains trajectories first with a larger teacher and then applies RL. Rewards
cover final correctness, protocol validity, non-truncated memory, and an exponentially
decayed preference for the earliest sufficient stopping point. The reported gains are
largest on multi-hop tasks, where global retrieval recovers bridge evidence missed by
linear scanning. Reported decoding work and latency are much lower than MemAgent.

Implementation decision: make controller actions first-class trace events. A final
answer is not authorized merely because the context looks coherent; it requires an
evidence-sufficiency decision. Early STOP freezes working memory, preventing redundant
later chunks from overwriting an already sufficient state.

The deterministic V1 provides the state machine and replaceable hooks for learned
planning/sufficiency. It does not pretend that heuristic token overlap reproduces the
paper's trained controller.

Related sources: REF-007 supplies recurrent memory; REF-012 supplies exact evidence
replay; REF-011 supports late query-specific construction; REF-009 learns a
state-dependent relevance gate.

## REF-009 — LycheeMemory

Source: [LycheeMemory](https://aclanthology.org/2026.acl-long.365/)
Quality: Moderate for its peer-reviewed architecture and reported ablations; lower
for transfer to different trunks and on-device runtimes.

LycheeMemory divides the system into a Compressor, Gate, and Reasoner. Four-thousand-
token source chunks become compact KV-style memory blocks. At query time, a Gate sees
the query, candidate block, and evolving plaintext working memory. Only relevant
blocks invoke the Reasoner and update working memory.

Compression uses interleaved learned memory tokens. Training includes reconstruction,
question answering, and creative generation before end-to-end RL. The Gate is trained
separately as a classifier, while the Compressor and Reasoner are jointly optimized
for downstream reward. State-dependent gating matters for multi-hop work because an
entity learned in one reasoning step can make a later block newly relevant.

The paper reports extrapolation from roughly 7K training contexts to 1.75M and lower
peak memory and latency than MemAgent. These are architecture-specific results and
must not be treated as proof for arbitrary GGUF/Ollama models.

Implementation decision: define an optional latent-block interface with a Gate that
conditions on evolving memory, but keep a textual/raw fallback. Gate recall is more
important than precision because a false negative can erase the only route to a fact.

Related sources: REF-007 is the recurrent baseline; REF-010 adds reversible hierarchy;
REF-013 creates portable buffer tokens; REF-019 performs query-aware KV paging.

## REF-010 — R³Mem

Source: [R³Mem](https://aclanthology.org/2025.findings-acl.235/)
Quality: Moderate for the peer-reviewed reversible-memory result; lower for exact
reconstruction outside its evaluated domains.

R³Mem represents history as virtual memory tokens and trains forward compression plus
backward reconstruction. It builds hierarchical document, paragraph/entity, and
sentence relationships rather than collapsing an entire history into one summary.
Cycle-style objectives encourage the compressed form to recover source content.

The key contribution for this project is a separate reconstruction metric. A compressed
artifact is not trusted merely because one QA benchmark remains high. Its relevant raw
span must be selectively recoverable, with reconstruction fidelity reported separately
from downstream answer accuracy.

Implementation decision: every derived memory carries source chunk IDs and local
offsets. EXPAND recovers exact source now. Learned RECONSTRUCT is a later backend and
must be labeled derived/unverified when it cannot resolve to immutable evidence.

Related sources: REF-013 uses portable latent buffers; REF-009 stores KV-style blocks;
REF-012 replays source text; REF-011 argues against destructive ingestion-time choices.

## REF-011 — LazyMem

Source: [LazyMem](https://arxiv.org/abs/2607.22690)
Quality: Low because it is a future-dated preprint in the current corpus; its ablation
question is clear and reproducible.

LazyMem preserves raw interactions and delays memory construction until a query exists.
It retrieves broadly, processes overlapping candidate windows, and constructs a small
answer-specific memory. The paper reports strong LongMemEval results with only a few
hundred answer-context tokens, but those numbers require reproduction on this stack.

The architectural lesson is robust: future importance cannot be known reliably at
write time. Ingestion should perform lossless storage and cheap indexing, not irreversible
summarization. Query-time compression can use the actual question and can be regenerated
from raw evidence if a first attempt is wrong.

Implementation decision: the V1 ingestion path never summarizes. Retrieval targets
50–200 union candidates, then reranks and selects exact spans for the current query.
Write-time facts and episodes are accelerators with provenance, not replacements.

Related sources: REF-008 supplies iterative retrieval; REF-012 supplies replay;
REF-007 shows overwrite failures; REF-013 explores higher-density query-independent
latent storage.

## REF-012 — ReContext

Source: [ReContext](https://arxiv.org/abs/2607.02509)
Quality: Low because it is a recent preprint; moderate confidence in the general replay
principle because the operation is simple and independently testable.

ReContext uses model-internal question-to-context relevance signals to select candidate
tokens, materializes their containing source spans, and recursively replays an ordered
evidence pool immediately before final generation. Its full-context setting preserves
the original prompt, while replay changes the state from which later relevance is
computed. Two rounds are generally better than one, but ideal depth and token budget
vary by task.

The method requires internal relevance signals and assumes the original long context
is already resident. This project cannot reproduce that assumption under a hard 16K
cap. It reproduces the safer semantic operation: recursively select evidence from the
external corpus and replay exact source spans adjacent to the query.

Implementation decision: summaries and exact replay use distinct tags and authority.
Evidence selection always points to original chunk offsets, never to text copied from a
prior summary. Replay must be the final context region before the current query.

Related sources: REF-008 controls recursive search; REF-011 defers construction;
REF-020 measures retrieval and multi-hop failures; REF-019 uses a related query-aware
selection idea at the KV-page level.

## REF-013 — Latent Context Compilation

Source: [Latent Context Compilation](https://arxiv.org/abs/2602.21221)
Quality: Low because it is a preprint with one principal backbone; very low for claiming
drop-in compatibility with the current quantized runtime.

Latent Context Compilation optimizes a disposable LoRA as a compiler, distilling a
specific context into portable buffer-token KV states consumed by the frozen base model.
The LoRA is discarded. Training combines context reconstruction with KL alignment on
context-agnostic instructions so the buffers remain on the model's instruction-following
manifold. The paper reports useful fidelity at 16x and sometimes 32x compression.

The 16x result is directly relevant to the 256K-to-16K target, but it is not a license
to discard text. Buffer tokens are model/version specific, compilation costs include
gradient work, and exact numeric/code detail needs independent testing.

Implementation decision: define latent artifacts as optional L2 records with model hash,
compiler version, compression ratio, reconstruction score, and source provenance. Never
make them the only copy. Require exact strings, chronology, code symbols, and multi-hop
ablation before enabling them by default.

Related sources: REF-010 emphasizes reversibility; REF-009 learns reusable block
compression; REF-011 favors query-time construction; REF-017 quantizes physical KV
rather than semantic content.

## REF-014 — Infini-attention

Source: [Infini-attention](https://arxiv.org/abs/2404.07143)
Quality: Low-to-moderate for the published experiments, very low as an immediate
integration because it changes attention and requires continued training.

Infini-attention combines exact local masked attention with a bounded compressive
linear-attention memory. A learned gate mixes local attention output and long-term
memory output. Memory updates occur segment by segment, keeping state bounded.

Implementation decision: retain as a V2 model-level research branch. It cannot be
implemented faithfully by prompt orchestration around stock Ollama and must not delay
the external-memory harness.

Related sources: REF-015 is another neural-memory intervention; REF-016 stabilizes a
rolling cache without long-term recall; REF-007 is deployable above the model.

## REF-015 — Titans

Source: [Titans](https://arxiv.org/abs/2501.00663)
Quality: Low as a preprint for this use; very low for transfer to an unchanged Qwen/Ornith
runtime.

Titans treats attention as short-term memory and adds a neural long-term memory updated
at inference. Its surprise-based update, forgetting, and architectural variants—memory
as context, gate, or layer—offer a path to learned persistent state beyond KV.

Implementation decision: isolate Titans-style updates behind a model-training branch.
External raw evidence and provenance remain mandatory even if a neural memory is later
added, because neural weights do not provide exact citations or simple rollback.

Related sources: REF-014 uses compressive attention memory; REF-009 uses latent blocks
plus textual working memory; REF-010 explicitly reconstructs source.

## REF-016 — StreamingLLM

Source: [StreamingLLM](https://arxiv.org/abs/2309.17453)
Quality: Moderate: peer-reviewed, reproducible, and widely integrated; not evidence of
historical recall.

StreamingLLM keeps a few initial attention-sink tokens plus a rolling recent KV window.
This prevents the perplexity collapse caused by evicting the initial tokens and enables
stable generation over very long streams. The paper explicitly states that it does not
extend long-term memory; discarded history remains unavailable.

Implementation decision: use attention sinks only if the serving backend exposes a safe
rolling-KV mode. Historical recall still pages source evidence through the virtual-memory
system.

Related sources: REF-017, REF-018, and REF-019 optimize physical KV; REF-007 and REF-008
provide semantic memory outside it.

## REF-017 — KIVI

Source: [KIVI](https://arxiv.org/abs/2402.02750)
Quality: Moderate-to-high for tested model families and system measurements; portability
to Qwen3.8/Jetson kernels must be measured.

KIVI finds persistent outlier channels in keys but not values. It therefore quantizes
keys per channel and values per token, keeping a short full-precision residual before
grouping. The paper reports 2-bit KV with little quality loss on evaluated models, lower
peak memory, larger batches, and higher throughput.

Implementation decision: treat KIVI as a physical-memory experiment with its own quality
gate. It can enlarge L0 or free VRAM but cannot replace retrieval. Test exact values,
long-range code references, and speech/tool workloads before enabling 2-bit cache.

Related sources: REF-018 varies retention by layer; REF-019 pages by query; REF-013
creates semantic buffer tokens rather than quantizing ordinary KV.

## REF-018 — PyramidKV

Source: [PyramidKV](https://arxiv.org/abs/2406.02069)
Quality: Low-to-moderate; reported LongBench results are relevant, but kernel/backend
compatibility and task sensitivity need local validation.

PyramidKV allocates more retained KV to lower layers and progressively less to higher
layers, based on observed layerwise attention concentration. The paper reports preserving
quality with a small fraction of the full cache on tested models.

Implementation decision: expose per-layer cache retention only through a backend
capability interface. Compare against uniform eviction and full KV while holding the
semantic retrieval system fixed.

Related sources: REF-017 reduces bytes per KV element; REF-019 selects query-relevant
pages; REF-016 retains sinks and recent tokens.

## REF-019 — Quest

Source: [Quest](https://arxiv.org/abs/2406.10774) and
[official repository](https://github.com/mit-han-lab/Quest)
Quality: Moderate for its peer-reviewed kernel result; model/backend transfer remains
an engineering question.

Quest divides the KV cache into pages, stores per-channel key minima/maxima, estimates
an upper bound on query-page attention, and loads only top-ranked pages for exact
attention. The paper distinguishes self-attention kernel speedup from smaller end-to-end
inference speedup.

Implementation decision: use Quest as the design reference for L0 query-aware paging,
not as a semantic evidence retriever. Page choices and cache hit rates belong in the
same debug trace as L1–L4 movement, but their correctness ablation is separate.

Related sources: REF-018 is layer-aware eviction; REF-017 is KV quantization; REF-012
performs query-aware evidence replay in text space.

## REF-020 — RULER

Source: [RULER](https://arxiv.org/abs/2404.06654) and
[official repository](https://github.com/NVIDIA/RULER)
Quality: Moderate-to-high as a peer-reviewed behavioral benchmark; synthetic tasks do
not replace realistic repository, conversation, or document evaluations.

RULER contains 13 tasks across retrieval, multi-hop variable tracking, aggregation, and
question answering. It varies context length and hard distractors, then distinguishes
claimed length from effective length. The study shows that simple needle tests can stay
perfect while broader long-context behavior collapses.

Important protocol detail: the original evaluation appends an answer prefix and uses
recall-style scoring. Later reproductions that omit the prefix are not directly comparable.
This project must pin a benchmark revision and record its prompt/scoring protocol.

Implementation decision: RULER is the first external benchmark, but not the only one.
Add exact strings, chronology, conflicting versions, code symbol tracing, persistent
constraints, citations, and reconstruction. Always report the oracle evidence-pack ceiling.

Related sources: REF-007 and REF-008 report RULER-derived results; REF-012 evaluates
evidence use at 128K; REF-016 demonstrates why infinite streaming is not recall.

## Cross-paper conclusions

1. Storage and working context are different systems. No paper justifies destroying the
   only source copy after compression.
2. Recurrent memory provides bounded processing, but blind overwriting is unsafe.
3. Retrieval needs explicit planning, dependency discovery, and evidence sufficiency.
4. Query-time construction is safer than irreversible ingestion-time selection.
5. Exact replay close to generation is a low-risk, testable improvement.
6. Latent memory is promising at the required 16x ratio but remains an accelerator.
7. KV quantization, retention, and paging optimize L0; they do not solve semantic recall.
8. Effective context must be measured against oracle retrieval and realistic failure cases.

## V1 reproduction mapping

- MemAgent: bounded recurrent-memory slot and configurable future chunk/memory sweeps.
- InfMem: explicit PRETHINK, RETRIEVE, WRITE, ANSWER, STOP trace events.
- LazyMem: append raw evidence first; construct answer context only at query time.
- ReContext: replay exact source spans immediately before the current query.
- R³Mem: EXPAND/RECONSTRUCT operations and separate reconstruction fidelity.
- Lychee/LCC: optional latent artifact interfaces, not a default source of truth.
- KIVI/PyramidKV/Quest/StreamingLLM: separate backend experiments after V1 fidelity.
- RULER: pinned external benchmark plus local adversarial suites and oracle ceilings.

## Current evidence and non-claims

The implemented deterministic harness has passed local 16K–1M synthetic storage and
retrieval runs: oracle and hybrid retrieval replayed the exact value, followed a
controlled two-hop dependency, recovered chronology, and pinned the early constraint,
while a final-window FIFO baseline missed the old value beyond its resident window.

The RULER adapter pins v1 revision
`e8bbff677ca2c239640dc90f93310dcf32408c93`, reattaches its separated answer prefix,
keeps references out of production retrieval, emits official-scorer-compatible JSONL,
and labels reference-location oracle runs.

On the live 32 GB Jetson, the model launcher selected an 8,192-token physical window
under a configured 16,384-token ceiling. One official RULER v1 sample from each of all
13 task classes produced these model-answer results:

- 512K hybrid: 100%, 29,381 prompt tokens, 4.76s p50 and 7.90s p95 inference;
- 1M final-window FIFO: 7.69%, 75,658 prompt tokens, 9.74s p50 and 12.35s p95;
- 1M hybrid: 100%, 22,699 prompt tokens, 3.49s p50 and 6.80s p95;
- 1M oracle-assisted locator: 100%, 28,345 prompt tokens, 4.86s p50 and 8.23s p95.

The final 1M hybrid run matched the oracle ceiling on this fixture set at
341x–3,449x source-to-resident compression. Both initially scored 90.38% before a
generic query-time exact-relation compiler was added; identical hybrid and oracle
failures isolated the problem to answer-path reasoning rather than retrieval recall.
The compiler uses only retrieved text and exact provenance, never references or
expected answers, and declines partial or conflicting compilations. Report hashes and
per-task results are preserved in
`.aiwg/testing/evidence/ruler-v1-512k-1m-jetson.json`.

These are single-sample-per-task milestone results. They establish a working 1M
virtual-memory path on the deployed model, not native-long-context equivalence,
population-level confidence, learned compression fidelity, or broad real-world task
coverage.

A subsequent sealed seed generated three new samples per task at 256K. With TTS and
pointing co-resident, the allocator selected a stricter 4,096-token physical window.
The first hybrid pass scored 91.03%; targeted oracle diagnostics separated a
natural-language dependency miss from exhaustive multi-value compilation and exact
answer-rendering faults. Generic, reference-blind fixes were then applied: bounded
exact-key scans over immutable evidence, a two-candidate/one-round proper-name bridge
expansion, and source-exact ambiguity preservation. On the unchanged 39-sample
fixture, hybrid scored 100% on all 13 tasks using NVIDIA's pinned metric code, with
3.03s p50 / 4.13s p95 inference and 153x-905x compression. The initial failures,
oracle diagnostics, schema compatibility shim, hashes, and final scores are preserved
in `.aiwg/testing/evidence/ruler-v1-256k-seed9137-jetson.json`.

This result provides better position and example coverage than the first milestone,
but three samples per task from one seed still do not establish population-level
confidence or broad domain generalization.

A one-sample-per-task causal matrix over that sealed fixture then isolated the V1
components. Full V1 scored 100%; removing graph channels also scored 100%; one-pass
full V1 scored 95.38%; removing aggregation scored 84.62%; removing exact-relation
compilation scored 87.69%; recursive raw hybrid replay scored 72.31%; one-pass raw
hybrid scored 69.23%; BM25-only scored 61.54%; and hashing-dense-only scored 23.08%.
Insufficient conditions were left unanswered. On this slice, recursive control adds a
smaller but real variable-tracking gain, while query-time aggregation and relation
compilation explain the large gap between ordinary hybrid RAG and the full system.
Graph retrieval needs a topology-rich benchmark, and the current hashing embedder is
not suitable as a standalone semantic retriever. Exact settings and report hashes are
preserved in
`.aiwg/testing/evidence/ruler-v1-256k-ablation-seed9137-jetson.json`.

## Required next experiments

- Repeat the 16K–1M model-answer curve with additional independent sealed seeds and
  randomized evidence positions; the completed 256K run covers only one seed with
  three samples per RULER task class.
- Extend the completed BM25/dense/graph/recursion/aggregation/compiler ablations to
  all three sealed samples per task, additional seeds and lengths, no-replay and
  fixed-budget conditions, plus topology-rich code/entity suites.
- Continue using oracle-context answers to distinguish model faults from retrieval faults.
- Add answer-quality curves to the completed 512/1K/2K/4K recurrent-memory by
  2K/4K/8K chunk-budget integrity matrix.
- Replace the weak hashing-dense lane with a production-quality embedding model and
  rerun dense/hybrid ablations without changing the immutable corpus.
- Extend the implemented code topology, chronology, supersession, contradictions, and
  100K+ constraint fixtures into model-answer and adversarial held-out evaluations.
- Measure latency, inference tokens, RAM/VRAM, SQLite/index size, and evidence tokens.
- Prototype latent compilation only after the training-free V1 has a stable fidelity curve.
