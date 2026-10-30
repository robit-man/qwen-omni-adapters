# Phase A3 — native filtered reads without reconstructed TTS files

The v2 reader selects a component's metadata and tensor inventory before its
architecture-specific loader runs. Tensor offsets remain anchored to the same
read-only sidecar. It does not route a heterogeneous tensor set into an
unmodified llama.cpp graph or replace the stock Ollama model layers.

[Native verification](phase-a3-native-views.json) compared all 689 tensors and
56 metadata keys against the two actual component GGUFs. Shapes, quantization
types, tokenizer arrays, source offsets and 1,476,331,204 tensor bytes matched.
Invalid selectors left the source context intact. No GPU was initialized by
this structural verifier.

An initial full-file mmap prefetch implementation was rejected: it touched the
sibling codec region and raised worker peak RSS by approximately 438 MiB.
The accepted reader disables full-file MAP_POPULATE for indexed model views,
so only selected tensor pages are faulted. Other models retain their existing
prefetch policy.

[Final scoped CUDA comparison](phase-a3-tts.json) used the same native binary,
checkpoint, seed and enabled conditioning cache for materialized and direct
inputs. All 12 PCM results matched byte-for-byte, all continuity checks passed,
and both references returned identical A after B.

| Reference | Materialized peak worker RSS | Direct peak worker RSS | Reduction |
| --- | ---: | ---: | ---: |
| Female | 1,268,268 KiB | 1,091,376 KiB | 13.95% |
| Alternate/default | 1,272,272 KiB | 1,089,728 KiB | 14.35% |

First packets after the initial utterance remained around 0.07 seconds. One
process per layout/reference does not establish p95/p99 latency. This is worker
RSS on an A100, not a claim about total system memory or reduced GPU weights.

[Preparation receipt](phase-a3-preparation.json): two reconstructed files totaling
1,482,388,192 bytes become two source aliases using 8,192 allocated bytes, with
zero reconstructed tensor payload. Median verified warm preparation fell from
5.776 to 4.742 seconds over three sequential runs per mode. Existing benchmark
components were retained; these are new-cache costs, not reclaimed disk space.

The [route smoke](phase-a3-routes.json) exercised real Flask handlers and native
CUDA workers using those aliases: complete 24 kHz mono PCM16 WAV, ordered streamed
PCM, request-local reference cloning, cancellation, and a fresh B after cancelled
A. Outputs matched materialized references. It used an in-process client and the
same loader ABI before the final prefetch-only fix; the final native comparison
above covers that updated allocation policy. It does not qualify a public
deployment or all voice languages.

Both builders apply the incremental patch; apply-once/repeat-build controls keep
the native pin. Full `scripts/validate.sh`: **5,035 passed**, followed by all
remaining gates. Windows source integration was updated, but no Windows native
build or performance qualification is claimed. The task-owned CUDA leases were
released and peer workloads were retained.
