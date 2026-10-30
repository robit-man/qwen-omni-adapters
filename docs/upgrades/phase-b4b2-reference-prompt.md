# B4b2 — native full-reference conditioning and immutable reuse

The pinned native Qwen3-TTS Base backend now accepts the reference transcript
and all sixteen acoustic-code groups, retaining its speaker embedding. This
closes the native prompt-construction gap after B4b1. The public request,
saved-profile and runtime reference-encoder integration remain pending; this
milestone does not make the current portal use full-reference conditioning.

## Implementation and boundaries

`llama.cpp-qwen3tts-reference-prompt.patch` applies after the existing ten
patches to llama.cpp `458681e1d5d4a29a1463c4732e03226cf384b997`.
It follows Qwen's `generate_icl_prompt(non_streaming_mode=True)` from
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py):
reference text, target text and text EOS paired with codec PAD; then codec BOS
and summed reference-code embeddings paired with text PAD. Acoustic generation
still streams. The embedding graph reads all codebooks from the existing
projector; it adds no trained parameters or independent language model.

Before target decoding, the causal waveform decoder consumes only the reference
codes in batches of at most 72 frames. Its reference PCM is discarded. Target
output starts from the last real reference frame. A fresh helper, language KV,
semantic sampler and acoustic RNG are retained for every request. Reference
priming cannot advance the target acoustic RNG.

One model/projector-bound cache entry holds reference codes, transcript,
embeddings and **reference-only** decoder state. Exact reference content,
resolved language and speaker-encoding epoch must match. Target decoder state,
target codes, generated PCM, language KV and samplers never enter this cache.
The entry is limited to 16 MiB; replacement frees the previous allocation and
oversize entries are declined. Wrapper-level identity/normalization keys and
request-local/session handling remain work for the public integration.

The additive CLI accepts `--tts-reference-prompt-file FILE`, a private JSON file
of at most 128 KiB, with schema `robit.qwen3tts.reference-prompt.v1`,
`reference_text` (1–8192 UTF-8 bytes), and `codes` (1–375 frame-major arrays of
16 integers in 0–2047). A speaker reference is mandatory. Bad schema, dimensions,
codes or context capacity fail explicitly. Existing invocations without this
flag retain the original speaker-embedding path. The public voice field is not
yet introduced.

## Measured qualification

Both devices used the same installed three-component sidecar:
`7838f27ad60ed5181c60e9b6e7263df944933f07099602aa70a5e96c8640f388`.
The Base model and codec remain the exact B4b1 checkpoints. Reference codes are
the B4b1 frozen-oracle fixtures, already proven integer-exact on both devices.
The 19-frame reference uses the supplied short transcript; the 163-frame case
uses an explicit machine transcript, not a human-verified/default profile.
The long case exercises three reference-decoder batches.

| Device/reference | Warm first PCM, cache off | Cache on | Reduction |
| --- | ---: | ---: | ---: |
| A100, 19 frames | 91.84 ms | 70.05 ms | 23.73% |
| Orin, 19 frames | 508.41 ms | 241.38 ms | 52.52% |
| A100, 163 frames | 198.00 ms | 88.72 ms | 55.19% |
| Orin, 163 frames | 1038.54 ms | 289.44 ms | 72.13% |

These are medians of the two warm requests in each bounded A/B/A case, measured
from an already loaded native worker. They exclude reference encoding and model
loading and are not end-to-end or population latency claims. Cached entry sizes
are 5,375,839 and 6,564,915 bytes. Each case records one embedding computation,
one reference decoder prime and two cache hits.

All **24 outputs** (two references × two cache policies × three prompts × two
devices) were byte-identical between cache policies on their respective device,
and both A requests matched. Raw PCM continuity passed. A separate comparison
against the preserved B2 control returned **12 exact legacy PCM outputs** through
the new installed indexed sidecar. The first attempted legacy comparison
incorrectly gave indexed aliases to the older non-indexed baseline; it failed
before readiness. The corrected comparison uses the hash-verified original
materialized components for that control.

The already resident Orin reasoner transcribed all eight distinct generated
device/reference/target cases exactly. This verifies fixture intelligibility,
not speaker similarity, accent preservation, expressive quality, long target
consistency, all-language quality or public session isolation. All these remain
release gates. On Orin, minimum sampled available memory stayed above 5340 MiB
while the production reasoner and harness remained active; the private child
guard's 2048 MiB floor was not crossed. No production service was restarted.

## Reproduction

Build with `scripts/build_llama_cpp.sh` (or the PowerShell counterpart). Prepare
a private JSON using the validated reference codes and their corresponding
transcript. No model download is performed by this qualifier:

```bash
python runtime/qualify_tts_reference_prompt.py \
  --binary BUILD/bin/llama-tts \
  --model INSTALLED_SIDECAR --projector INSTALLED_SIDECAR \
  --reference VOICE.wav --reference-prompt PRIVATE_REFERENCE.json \
  --work-dir NEW_DIRECTORY/work --ready-file NEW_DIRECTORY/ready \
  --out REPORT.json
```

On discrete CUDA, reserve the exact GPU and a positive VRAM budget through
`docker gpu run`, using the fresh readiness file as its ready check. The
qualifier writes that file only after native readiness and actual PID residency,
then waits for broker admission before generation. On Tegra, supply
`--minimum-available-mib 2048` without a discrete scope. All children close on
failure or completion. Native library mappings, checkpoint/source receipts,
counter observations and individual PCM hashes accompany this report.
