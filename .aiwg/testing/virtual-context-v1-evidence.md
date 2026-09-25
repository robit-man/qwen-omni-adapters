# Virtual-context V1 verification evidence

- Date: 2026-09-25
- Physical context budget: 16,384 tokens
- Scope: deterministic storage, retrieval, controller, packing behavior, and
  a bounded live-model sparse-recall smoke
- Non-claim: this is not a complete RULER model-answer score or proof of native
  long-context equivalence

## Repository validation

Command: `./scripts/validate.sh`

Result: 634 tests passed in 30.41 seconds. Source, contract, VAD, call-queue,
browser-cache, and unit validation gates passed.

## Synthetic source-length ladder

Command:

```bash
.venv/bin/python scripts/benchmark_virtual_context.py \
  --lengths 16000,32000,64000,128000,256000,512000,1000000
```

| Source tokens | Chunks | Resident evidence/input tokens | Ratio | Ingest s | Retrieval s | Index bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 16,000 | 19 | 3,565 | 4.507x | 0.084 | 0.045 | 1,101,824 |
| 32,000 | 38 | 3,627 | 8.859x | 0.166 | 0.092 | 2,056,192 |
| 64,000 | 73 | 3,888 | 16.525x | 0.358 | 0.109 | 3,932,160 |
| 128,000 | 144 | 3,247 | 39.604x | 0.681 | 0.128 | 8,093,696 |
| 256,000 | 288 | 3,627 | 70.875x | 1.458 | 0.181 | 15,777,792 |
| 512,000 | 573 | 3,888 | 132.197x | 3.015 | 0.250 | 31,682,560 |
| 1,000,000 | 1,118 | 3,334 | 301.296x | 5.985 | 0.344 | 62,005,248 |

At every length in this deterministic fixture:

- the dense index covered every chunk;
- hybrid and oracle evidence recall were 1.0;
- controlled two-hop dependency recall was 1.0;
- old/new chronology recall was 1.0;
- the early hard constraint remained pinned;
- the exact value was replayed adjacent to the query.

The final-16K FIFO baseline found the value only at the 16K source length. The
hashed dense channel did not put the target in its selected set at 256K and 1M;
exact/BM25 retrieval recovered it. This is retained as an ablation signal, not
hidden or converted into a dense-retrieval success claim.

## Adversarial 16K-to-1M matrix

Command:

```bash
.venv/bin/python scripts/benchmark_virtual_context_matrix.py \
  --lengths 16000,32000,64000,128000,256000,512000,1000000 \
  --output /tmp/virtual-context-adversarial-matrix.json
```

The matrix contains ten scenario families at each length and eight baselines,
for 560 preparation cases. It scores the evidence made resident; it does not
use expected answers in production queries and does not claim downstream model
answer correctness.

| Baseline | Pass rate | Evidence recall | Replay recall | Exact provenance |
|---|---:|---:|---:|---:|
| final-window FIFO | 15.7% | 0.000 | 0.281 | 0.000 |
| dense-only RAG | 90.0% | 0.980 | 0.980 | 0.980 |
| lexical-only RAG | 100% | 1.000 | 1.000 | 1.000 |
| hybrid RAG | 100% | 1.000 | 1.000 | 1.000 |
| bounded recurrent text | 80.0% | 0.900 | 0.880 | 0.000 |
| recursive exact replay | 100% | 1.000 | 1.000 | 1.000 |
| structured exact replay | 100% | 1.000 | 1.000 | 1.000 |
| reference-location oracle | 100% | 1.000 | 1.000 | 1.000 |

| Source tokens | Ingest s | Index bytes | Peak process RSS MiB | Maximum production resident input | Minimum production ratio |
|---:|---:|---:|---:|---:|---:|
| 16,000 | 0.339 | 626,688 | 28.7 | 713 | 22.44x |
| 32,000 | 0.362 | 827,392 | 30.7 | 713 | 44.88x |
| 64,000 | 0.415 | 1,294,336 | 34.0 | 713 | 89.76x |
| 128,000 | 0.582 | 2,146,304 | 40.2 | 716 | 178.77x |
| 256,000 | 0.915 | 4,173,824 | 48.8 | 716 | 357.54x |
| 512,000 | 1.732 | 8,089,600 | 66.3 | 716 | 715.08x |
| 1,000,000 | 3.206 | 15,605,760 | 101.9 | 719 | 1,390.82x |

All 63 provenance spans reconstructed byte-for-byte from immutable chunks.
The structured replay path excluded both forbidden near-duplicate controller
values at every length. Peak RSS is the process high-water mark from one
sequential run, and the very high ratios reflect sparse evidence needs; they
must not be generalized to dense-evidence tasks.

The first matrix run exposed two non-benchmark-specific defects. Controller
sufficiency treated conversational filler as evidence anchors and rejected
complete retrievals. Separately, every active fact/decision was being injected
when no explicit subject scope was supplied. The current implementation weights
query-derived exact identifiers, retains terminal dependency facts, keeps
constraints/current-plan records deterministic, and subject-scopes other
derived memories. The strengthened adversarial gate now fails on forbidden
near-duplicate replay.

## Official RULER 256K preparation gate

NVIDIA RULER's `variable_tracking.py` generated one four-hop sample at a
reported 255,984-token length using its `cl100k_base` accounting. The question
requires reconstructing five assignment edges and is preceded by a separate
few-shot assignment graph that acts as a hard distractor.

| Condition | Required edges replayed | Resident tokens | Compression | Retrieval queries | Sufficient |
|---|---:|---:|---:|---:|---:|
| FIFO | 0/5 | 14,000 | 18.28x | 1 | baseline-only |
| Hybrid | 5/5 | 7,238 | 35.37x | 6 | yes |
| Reference-location oracle | 5/5 | 7,238 | 35.37x | 6 | yes |

This gate initially found two real failures: the controller stopped after only
2/5 dependency edges, and the packer later evicted one recovered edge. The
regressions now require anchored relationship closure before STOP, fail closed
when the dependency budget expires, ignore the unrelated few-shot graph, and
use dependency-aware exact line replay when a full chunk does not fit.

This is preparation/replay evidence, not an answer-accuracy score. The complete
13-task FIFO/hybrid/oracle model run remains a release gate.

## Official RULER 13-task 32K preparation gate

One upstream-generated sample for each RULER v1 task was prepared with
`cl100k_base` accounting: eight NIAH variants, variable tracking, common-word
extraction, frequent-word extraction, SQuAD QA, and HotpotQA. Hybrid preparation
reported sufficient evidence for 13/13 tasks under the same 16,384 resident cap.

This sweep exposed three failures before it passed:

- one of four NIAH values shared a large source chunk with another value and was
  lost by single-span replay; a chunk can now replay multiple disjoint exact
  spans, each with its own offsets;
- QA documents were fixed-token chunks spanning unrelated records; `Document N:`
  sections are now independent structural chunks and exact entity titles receive
  a query-derived reranking signal;
- frequent/common-word extraction requires corpus-wide aggregation, not sparse
  retrieval; a deterministic frequency memory now retains full provenance to all
  immutable input chunks and keeps its long source list out of the resident tag.

All NIAH, variable-tracking, and word-frequency reference terms were resident.
The QA packs contained the exact support documents. HotpotQA's expected `yes`
is intentionally not required to occur in source evidence; it is a conclusion
the model must draw from the two replayed nationality statements. These are
preparation results, not the pending live 13-task model scores.

## Live official RULER model-answer gate

The same official four-hop sample was then evaluated against the resident
Ornith worker on the Jetson, still hard-capped at 16,384 physical tokens. The
score below applies NVIDIA RULER's `variable_tracking` `string_match_all`
metric: the fraction of the five reference variables present in the answer.

| Condition | Output allowance | Resident tokens | Compression | Score |
|---|---:|---:|---:|---:|
| FIFO | 256 | 14,000 | 18.28x | 0 |
| Hybrid before dependency-focus eviction | 256 | 7,487 | 34.19x | 0 |
| Hybrid before dependency-focus eviction | 512 | 7,487 | 34.19x | 20 |
| Hybrid after dependency-focus eviction | 256 | 7,264 | 35.24x | 100 |
| Reference-location oracle | 256 | 7,264 | 35.24x | 100 |

The failed hybrid conditions are retained because they exposed that a
high-scoring but disconnected few-shot assignment graph was still being paged
into residual evidence capacity. Query-time compression now focuses replay on
chunks containing the dependency identifiers discovered by the controller and
logs other recoverable candidates as `dependency_focus` evictions. The focused
hybrid answer and oracle answer were identical:
`IWSHA, YCSMT, RQMUC, FRHPM, and NLTIS.`

This is one official RULER sample, not a statistically meaningful task score.
It establishes the first milestone path end to end; it does not replace the
complete task/length matrix.

## Live Jetson sparse-recall smoke

The Jetson AGX Orin service was deployed with a hard 16,384-token comprehension
ceiling and shadow virtual memory. A live request against its resident Ornith
worker used 254,822 source tokens with the target fact in the middle:

- FIFO retained 14,000 tokens, replayed no evidence, and returned no answer.
- Hybrid retained 5,386 tokens (47.31x), replayed two exact evidence chunks,
  and returned `Q7M-441-PLUTO` exactly in 7.35 seconds with a 256-token output
  allowance.
- A diagnostic 64-token allowance had returned only `Q7M`; this was not counted
  as a pass. Direct exact-copy probes confirmed the model could reproduce
  hyphenated, underscored, and numeric values, isolating that result to output
  headroom rather than retrieval corruption.
- Comprehension, TTS, and pointing workers simultaneously held Tegra GPU device
  handles after the run; no comprehension-weight eviction was used.

## Covered invariants

- append-only source rows enforced by SQLite triggers;
- exact character/byte offsets and provenance reconstruction;
- structure-aware code, Markdown, JSON, log, and conversation chunking;
- BM25, hashed dense, exact, symbol, metadata, recency, entity, relationship,
  and code-topology indexes;
- class-specific importance/TTL defaults and explicit supersession;
- user-authored constraint promotion without promoting casual `I must` speech;
- PRETHINK/RETRIEVE/WRITE/STOP/ANSWER sufficiency traces;
- exact evidence replay and content-aware eviction;
- recurrent 512/1K/2K/4K memory by 2K/4K/8K chunk integrity matrix;
- periodic recurrent regeneration from immutable raw evidence;
- live tool-loop repacking and tool/control-envelope reservation;
- exact llama.cpp token-counter adapter and shadow-mode safe fallback;
- RULER input/reference isolation, answer-prefix restoration, oracle labelling,
  and official-scorer-compatible output records.

## Remaining release evidence

1. Run the 13 official RULER v1 tasks with the target model for FIFO, hybrid,
   and reference-location oracle conditions.
2. Report answer accuracy, oracle gap, inference tokens, rounds, p50/p95 latency,
   peak RAM/VRAM, and context-envelope use.
3. Run realistic long conversation, document, and code-agent workloads; the
   deterministic adversarial fixtures above are necessary regression evidence,
   not a substitute for those live task distributions.
4. Keep learned gates, latent compilation, reversible memories, and KV
   compression/paging behind independent reconstruction and ablation gates.
