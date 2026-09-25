# Virtual-context V1 verification evidence

- Date: 2026-09-25
- Physical context budget: 16,384 tokens
- Scope: deterministic storage, retrieval, controller, and packing behavior
- Non-claim: this is not a model-answer score or proof of native long-context
  equivalence

## Repository validation

Command: `./scripts/validate.sh`

Result: 621 tests passed in 23.09 seconds. Source, contract, VAD, call-queue,
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
3. Run realistic long conversation, document, and code-agent workloads with
   adversarial revisions, distractors, and conflicting entities.
4. Keep learned gates, latent compilation, reversible memories, and KV
   compression/paging behind independent reconstruction and ablation gates.
