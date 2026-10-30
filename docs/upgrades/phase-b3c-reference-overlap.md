# B3c — prepare saved reference conditioning during language inference

Acceptance criterion: a known saved full-reference voice should not defer all
checkpoint validation and reference preparation until the language response has
finished. Previously the language branch submitted no preparation request, and
the first TTS block performed that work serially. The bounded hypothesis was
that earlier preparation could reduce first-audio time without retaining TTS
generation weights or speaking uncommitted text.

`OMNI_TTS_PREPARE_REFERENCE=1` opts into one bounded background preparation per
adapter process. The internal loopback route accepts saved voice fields only;
the adapter sends no assistant content or reasoning. Preparation takes a private
reference snapshot, validates content and exact checkpoint identities, then
fills the existing one-entry, 128 KiB reference-conditioning RAM cache. It
releases the encoder and removes temporary files. Inline reference audio stays
request-local and does not enter this preparation path. No neural checkpoint or
public adapter field/event changed.

Preparation uses the existing TTS request lock without queuing behind active
speech. It skips an already resident TTS worker. A matching speech request joins
the work; another voice, embedding-only request or residency change cancels the
owned encoder before waiting. The encoder checks cancellation every 100 ms.
Tegra's 3,072 MiB admission and 2,048 MiB active encoding floors remain intact;
the 6,144 MiB idle TTS retention floor is unchanged. Optional preparation failure
falls back to ordinary synthesis. Unresolved tools still suppress speech in both
synchronous and streaming adapter paths.

## Real synthesis route checks

The A100 and Orin route gates both passed saved A/B/A, conditioning reuse after
weight shedding, inline/saved equality, complete-WAV/stream equality, batch and
cancellation cleanup. Every matching recording was byte-identical to the
preparation-disabled B3a receipt. Preparation took 1.807 seconds on A100 and
3.187 seconds on Orin, with no TTS worker loaded. Neither timing alone establishes
an end-to-end gain. The scoped A100 lease was released; existing production
services remained running.

## Real primary-language to PCM comparison on Orin

Three matched cold pairs alternated preparation-disabled and enabled execution.
Each used the actual adapter streaming function, a private loopback TTS HTTP
server and the already resident Ornith primary at its native chat endpoint.
`cache_prompt:false`, `think:false`, deterministic language sampling, the same
full-reference voice, exact sidecar, adaptive 4K context and automatic four-frame
packets were preserved. Each language response committed the same sentence:
"The blue bicycle is parked beside the garden gate."

| Measurement | Disabled median | Prepared median | Reduction |
| --- | ---: | ---: | ---: |
| Request to first PCM | 8.162 s | 6.797 s | 16.7% |
| Request to complete response | 12.760 s | 11.387 s | 10.8% |

All six PCM recordings were byte-identical and passed raw continuity. Per-pair
first-PCM reductions were 28.1%, 12.7% and 17.4%. Available memory never fell
below 3,311.7 MiB; this is headroom evidence, not a full-system peak-memory
comparison. Every first-PCM producer held the integrated GPU device handles.

For this short response, language completed in about 1.4–1.6 seconds. Checkpoint
identity validation overlapped language, but the native reference encoder itself
began afterward. The receipt therefore proves earlier preparation and lower
observed pipeline latency; it does not prove concurrent acoustic encoding for
this fixture, committed-first-phrase synthesis, physical speaker latency,+p95/p99 behavior, or the proposed 20% median first-audio target. The feature
remains opt-in pending broad concurrency and voice/language qualification.

The pipeline receipt records the private pilot's SHA-256. The maintained
`runtime/qualify_reference_overlap.py` reproducer follows that path with bounded
pair count and fresh output directories; the recorded runs used its private
precursor. `runtime/qualify_tts_reference_http.py --automatic-packets
--prepare-reference` reproduces the route gate. Both require the exact matching
native libraries. On discrete CUDA the route gate requires its own scoped
12,000 MiB lease; the pipeline comparison targets the resident Ornith on Tegra.

Full model parity, both-backbone Orin qualification, trained-head quality and
Hugging Face/Ollama publication remain separate, unfinished release milestones.
