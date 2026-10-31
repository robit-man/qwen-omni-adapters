# Current-encoder English CTC audit

The existing 2,496,000-parameter CTC head still passes its fixed English text
limits on A100 after native patches 13/14 changed encoder framing. Actual raw
WAV inputs were encoded again with all 16 native patches. No head, encoder,
projection, primary or TTS weights were trained or changed in this audit.
The [compact receipt](current-encoder-ctc-20261031.json) binds the reports,
manifests, producer binaries and weights by SHA-256.

| Device | Fixed split | Speech / empty | Mean speech WER | Empty false transcripts |
| --- | --- | ---: | ---: | ---: |
| A100 | Validation | 154 / 6 | 13.579% | 0 / 6 |
| A100 | Historical test | 147 / 8 | 8.843% | 0 / 8 |
| Orin | Validation diagnostic | 154 / 6 | 13.751% | 0 / 6 |
| Orin | Historical test diagnostic | 147 / 8 | 9.005% | 0 / 8 |

The unchanged text limits are mean speech WER at most 15%, empty false
transcripts at most 5%, and each speech cohort within one percentage point
of its fixed reference. All pass. The test recordings were previously used
to evaluate this same head; these results are a compatibility regression,
not a fresh blinded quality estimate. The 47 synthetic recovery recordings
remain difficult: A100 validation WER is 30.36% in that cohort.

The head uses the existing training-derived lexicon, 16-beam decoder and
32 frame candidates with its original greedy fallback. No external language
model, new vocabulary or transcript-based postprocessing was introduced.
Each encoder and head worker proved CUDA residency. On Orin, the existing
core and harness services stayed running; available memory during the head
diagnostic remained above 8,802.7 MiB.

## Orin numerical gate remains open

All 315 Orin frame counts match A100. Their feature values fail the separately
preregistered cross-device gate: minimum cosine 0.985593 and maximum relative
RMSE 16.962%, against required cosine 0.999 and RMSE 1%. The initial qualifier
stopped before head scoring. A separate downstream diagnostic then reused
those exact features and unchanged head to obtain the Orin text results
above. It explicitly records component compatibility as false.

Two fixed silence/natural-speech stage traces show matching mel input and
divergence amplified through the transformer. Disabling TF32 by environment
alone has no effect. Explicit FP32 matrix accumulation reduces maximum final
feature RMSE to 1.761%, still above the unchanged limit. Disabling Flash
Attention with that setting worsens it to 3.084%. Both precision variants are
private diagnostic binaries; neither changes production code or weights.
Further precision permutations were not launched.

The original [C1 audit](phase-c1-native-head.md) compared cached features.
This audit adds current raw-audio encoder evidence, with encoder and head
executing separately. Shared live encoder/head residency, partial streaming,
multilingual and unseen-word quality, complete-model latency and resource
savings remain unqualified. The head stays disabled. Environmental sound
parity is an independent release blocker; neither complete bundle is
qualified or published by this result.
