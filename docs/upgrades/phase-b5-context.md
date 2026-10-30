# B5 — bounded speech context without changing generated PCM

The native speech worker otherwise inherits a fixed 32768-position context.
Short text and its bounded acoustic generation require far fewer positions.
`OMNI_TTS_ADAPTIVE_CONTEXT=1` now selects 4K/8K/16K/32K using the complete UTF-8
byte count, generation-frame budget and 256 control positions. This conservative
bound follows the pinned Qwen3-TTS helper's byte BPE and one position per frame.
It does not estimate from characters, truncate input, or reduce generation limits.
The option remains off by default pending complete profile qualification.

Workers bind their selected context into their reuse profile. A batch selects
its largest member's context once, including an ephemeral inline-reference batch.
The primary reasoning trunk's KV policy and loaded state are untouched.

## Actual native comparisons

Both references were compared on an A100 and the production Orin, each using its
own matching native libraries. Each reference has A/B/A, short speech in all ten
supported languages, full 4096-character English input and full 4096-character
Chinese input. Long-input capacity probes use a 16-frame generation cap; they
are not long-utterance consistency tests. Short probes retain their 96-frame cap.

All **28 baseline/candidate waveforms per device** matched byte-for-byte. No
cross-device PCM equality is asserted. Tegra device-handle residency was required
for every worker; its 2048 MiB active-run guard was never crossed. Production
primary and harness stayed active throughout the isolated benchmark units.

| A100 resident GPU samples | Fixed 32K | Adaptive |
| --- | ---: | ---: |
| Short speech | 6266 MiB | 3106 MiB (4K) |
| Long English input | up to 6276 MiB | 3568 MiB (8K) |
| Long Chinese input | up to 6276 MiB | 4472 MiB (16K) |

These are per-worker resident samples after allocation/generation, not continuous
full-system peaks. A separate short A/B/A probe sampled the owned worker during
generation and observed 6266 versus 3106 MiB as well. No timing improvement is
inferred from these memory probes.

Orin's 100ms system-headroom minimum was 3179/3284 MiB for the two fixed-context
reference runs and 3419/3332 MiB for their adaptive runs. CPU worker peak RSS was
approximately 1.9 GiB in both policies after the long-input cases. This mixed
workload does **not** establish a 10% reduction in Orin full-system peak memory.
The primary deployment and its other activity were not reset to improve a score.

## Preserve and expose baseline limitations

The first broad run stopped on a raw-PCM continuity diagnostic in baseline
Russian speech. The revised comparison records the complete diagnostic for
every baseline and candidate rather than concealing the inherited flags or
changing the independent verifier's thresholds. The A100 female reference flags
Russian and Portuguese; Orin flags German and long English for that reference,
and French for the alternate. Corresponding candidates have exactly the same
waveform and diagnostic. The English A/B/A continuity gates remain green.

These unchanged flags remain part of broader voice qualification. Neither
`complete:true` in the comparison receipt nor source tests mean a clean
perceptual voice-quality or whole-release gate. `release_qualified:false` is
explicit. No existing voice capability is removed to obtain memory savings.

## Reproduction and source gate

Run `runtime/benchmark_tts_context.py --help`. Discrete CUDA requires the scoped
broker wrapper, one reserved UUID and a fresh readiness file. Tegra requires a
positive available-memory floor and uses private component files. Supply both
reference WAVs and matching native-library directories. Each worker is closed
before the next layout; no public service is created.

The focused route tests verify that long UTF-8 text is unchanged, every frame
budget survives, ordinary prompts reuse one context and mixed-length batches
reuse the largest tier. `./scripts/validate.sh` passed **6213 tests** and all
remaining source/JavaScript/shell/contract gates. Exact hashes and native input
identities are recorded in `phase-b5-validation.json`.
