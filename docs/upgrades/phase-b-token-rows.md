# Phase B2 — bounded reads from the existing quantized token table

## Change

The Base checkpoint's `token_embd.weight` has shape `[2048, 155008]`. Its stored
Q4_K data occupies 178,569,216 bytes; the previous helper expanded the entire
table to 1,269,825,536 float bytes before speaking. This allocation was unrelated
to the number of tokens in the actual utterance.

The incremental native patch adds a capacity-checked, single-row checkpoint
read using the same GGML dequantization operations as the whole-table API. TTS
requests only the rows it needs. A model-bound FIFO retains at most 512 rows by
default, with an independent 8 MiB byte cap. `--tts-conditioning-cache-rows`
accepts 0–4096; zero disables retention. No quantization, checkpoint, language
trunk, speaker computation, generation sampler or codec state changes.

The original whole-table API remains available to other callers. Cache-disabled
helpers also use bounded row reads; disabling speaker reuse no longer forces
the full float expansion. The metadata/reference cache and decoded-state resets
from B1 remain in place. Both build scripts recognize composed incremental
patches, preserving repeated-build behavior after overlapping successors.

## Correctly bound native comparison

[Measurement receipt](phase-b-token-rows.json). The control was the verified B1
executable with its matching shared libraries and conditioning cache enabled.
The candidate was the new token-row executable and its own libraries. The test
records `/proc/<pid>/maps` bindings and refuses a library from the other build.
An initial comparison was excluded because ELF RUNPATH selected newer libraries
for the copied control; its excluded receipt is retained only in local evidence.

Both modes loaded the same TTS and codec GGUFs under a scoped 12,000 MiB A100
lease and generated A → B → A for both shipped references, seed 42 and two-frame
streaming. All 12 raw PCM continuity checks passed. Every candidate PCM matched
its control SHA-256 exactly, including each repeated A after B.

| Reference | B1 peak worker RSS | Candidate peak worker RSS | Reduction | B1 process start → first PCM | Candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Female | 2,054,828 KiB | 1,269,816 KiB | 38.20% | 2.781 s | 2.173 s |
| Alternate/default | 2,057,408 KiB | 1,273,196 KiB | 38.12% | 2.782 s | 2.142 s |

These process-start timings combine measured graph loading and the first
utterance's first packet, on an already warm host filesystem. They improved
21.87% and 23.01% in these two reference cases. There was one newly loaded process
per mode/reference, followed by three utterances; this does not establish p95,
p99, whole-conversation latency, a cold-boot result or an Orin SLO.

The candidate performed zero whole-table loads. Across each three-utterance
sequence it read 30 token rows, hit cached rows 87 times and retained 245,760
token-row bytes, plus the separately bounded speaker conditioning. It encoded
the speaker once and reused that conditioning twice. Warm first packets remained
about 0.07 seconds. The removed float expansion improves CPU residency; the
measurement does not claim lower discrete GPU weight allocation.

## Validation and remaining gates

- Real native build and repeated builder invocation passed at the pinned commit.
- Full `./scripts/validate.sh`: **5,019 passed**, followed by all remaining gates.
- Each independent native patch has an apply-once/repeat-build control; no test
  or runtime gate was disabled.
- The task-owned GPU lease was released and existing owners were retained.
- Full reference-transcript/acoustic-prompt support, adaptive windows, Orin
  concurrency and trained cognitive heads remain separate milestones.

The source validation and implementation hashes are recorded in
`phase-b-token-rows-validation.json`. Reference bytes and generated speech remain
in bounded, ignored local benchmark directories; no credentials are retained.
