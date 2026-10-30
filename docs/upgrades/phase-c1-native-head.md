# C1 — qualified English CTC head and native GPU graph

Acceptance criterion: an attached-head candidate must pass the fixed 15% mean
speech WER / 5% empty-audio false-transcript pilot limits, then reproduce its
qualified outputs in native CUDA execution. C0's best greedy candidate reached
17.11% validation WER and remained disabled. The trained head uses the shared
frozen 1280-wide audio features and 2,496,000 learned parameters; it executes no
language-model reasoning stack.

## Training and decoder evidence

A training-only recovery-domain rebalance repeated its 1,273 training records
eight times. All 315 held-out records stayed identical, with no new unique
audio. The twentieth-epoch run selected 18.47% validation WER and was rejected;
test predictions were not evaluated for that candidate. It worsened every
validation speech stratum, so additional repetitions were not adopted.

The original best head's validation predictions contained 341 words outside a
training-derived lexicon, compared with 151 such words in the references. A
CTC prefix beam constrained to that training lexicon reduced broken word
fragments. It uses 16 beams, 32 frame candidates and 35,670 training words.
The 411,815-byte lexicon contains no held-out transcripts and loads no external
language model. Eight independent exhaustive small-lattice comparisons verify
the CTC probability accumulation, including repeated tokens and blank states.

| Fixed split | Speech / empty clips | Mean speech WER | Empty false transcripts |
| --- | ---: | ---: | ---: |
| Validation | 154 / 6 | 13.613% | 0 / 6 |
| Test, first evaluated after validation passed | 147 / 8 | 9.068% | 0 / 8 |

Validation includes 47 synthetic vocabulary-recovery clips (30.28% WER).
The test set contains procedural speech and LibriSpeech, with no recovery clips.
These are English pilot results; multilingual, unseen-word and broad acoustic
quality remain unqualified. The decoder returns no complete path on four
validation clips and one test clip; the pilot explicitly falls back to greedy
CTC for those cases. A complete lexicon path does not establish confidence about
an unknown word. The existing full audio bridge remains authoritative and no
fast path is enabled.

## Native execution

`runtime/export_audio_ctc_head.py` checks the joint qualification receipt, exact
head/lexicon/tokenizer identities and tensor layout before exporting eight F32
tensors plus the tokenizer, lexicon and provenance metadata. The component is
10,518,848 bytes, architecture `robit_audio_ctc`, SHA-256
`e939e15685138903f1c49701ccba830dcac058c780d116b5319532bd892bca15`.
Its single-tag attachment is separately qualified in
[C1b](phase-c1b-indexed-head.md); neither complete release is qualified.

`runtime/audio_ctc_probe.cpp` executes LayerNorm, affine/GELU, a causal
three-frame convolution and the CTC output projection entirely on one visible
CUDA backend. There is no CPU inference fallback. It loads no reasoning weights.
The pinned aligned F32 matrix fast path uses TF32 instructions, causing an
initial 0.0218 logit discrepancy. Padding each matrix product with one zero
input dimension preserves the trained operation and avoids that path.
`NVIDIA_TF32_OVERRIDE=0` is also required for qualification. The shared inference
backend and existing model graphs are unchanged.

All 315 held-out native feature matrices were compared with the qualified Torch
head on A100. Maximum absolute logit difference was 0.00006104, with zero
different greedy frames among 37,167 frames. Every lexicon-decoded transcript
matched the qualified reference. The same checked-in decoder was used.

On Tegra, the maintained `runtime/qualify_audio_ctc_native.py` compared all 315
matrices with the A100 logits and qualified transcripts. Maximum absolute logit
difference was 0.00008392 and every decoded transcript matched. The CUDA worker
held the required integrated-GPU handles. Available memory stayed above
6,949.8 MiB while the deployed Ornith service remained active. Median head graph
execution was 3.318 ms, excluding audio encoding, file reads and CPU decoding.

These probes consume **cached native encoder features**. They establish native
head execution on both devices; they do not establish an end-to-end Orin ASR
route, streaming encoder state, shared live encoder residency, complete-model
latency or memory savings. No deployed service restarted.

## Reproduction and remaining integration

The exporter requires build-only Torch/safetensors. Compile the probe against
the pinned GGML headers and matching CUDA libraries, for example:

```bash
g++ -std=c++17 -O2 runtime/audio_ctc_probe.cpp \
  -I vendor/llama.cpp/ggml/include -L /path/to/matching/ggml/lib \
  -Wl,-rpath,/path/to/matching/ggml/lib \
  -lggml -lggml-base -lggml-cuda -o /path/to/native-ctc-probe
```

The maintained qualifier takes `--binary`, `--head`, `--manifest`, `--work`,
`--ready` and `--out`. A manifest binds the head, feature files and reference
logits by SHA-256, and supplies training-lexicon token pieces plus fixed
validation/test references. Use fresh paths. Discrete CUDA requires an owned
4096 MiB scoped foreground lease; Tegra uses its own residency proof and
unchanged 3072/2048 MiB admission/active headroom floors. The A100/Torch parity
receipt records the private numerical-reference pilot; the Tegra receipt used
the maintained qualifier.

Next work is native execution on newly encoded audio, shared encoder
integration, streaming and independently broadened quality
tests. Decision/retrieval heads and recurrent sensory state remain separate
training milestones. Neither complete backbone release is qualified or published.
