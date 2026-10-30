# Environmental audio bridge repair — 2026-10-30

**Not qualified. Publication remains gated.** The user explicitly chose repair
before publication, with no experimental-preview exception.

The first repair attempt adds 2,840 CC0 FSD50K audio clips covering 180 leaf
classes, while retaining all 3,636 original replay records. New environmental
splits are uploader-disjoint: 2,200 train / 240 validation / 400 test. Every
ESC-50 source ID is excluded, speech-like ontology classes are excluded, and
source hashes do not cross splits. The FSD50K dataset entity is CC-BY-4.0;
individual added audio files are CC0. Audio is not redistributed with weights.

The official metadata archives pass their published MD5 checks. Selected audio
ZIP members were retrieved by HTTP ranges and individually CRC32 checked;
source SHA256 values are recorded. Whole audio archive MD5 verification is
**not** claimed. The normalized combined manifest SHA256 is
`ce07f73c91464aa2ac8e6969e43a3460db11ceb8200ebf0a76dcef077452ca61`.
See [the preparation receipt](environmental-repair-data-20261030.json).

Only Ornith's final audio projection was trained: 5,246,976 parameters, 400
steps, gradient accumulation 4, learning rate 3e-5, seed 42, frozen original
language and audio encoder. The initial projection was exported from the exact
installed GGUF, rather than assuming a similarly named training checkpoint.
Fine-tuning source revision: `e0741b40497746ba107b9d0fa25b6bcd9d61b434`.
See [the complete training receipt](environmental-repair-ornith-attempt1-20261030.json).

Validation loss falls from 0.679111 to 0.240158 in 1,419.56 seconds. The exported
GGUF SHA256 is
`46b789a7e129f9536d38e3d43133cd81e70c0488934fa04ddd6de37c3f2829d1`.
Only `mm.a.mlp.2.weight` and `.bias` change; all 858 frozen audio/vision tensors
remain byte-identical. Sampled peak Torch reserved memory is 27,574 MiB under
a 32,000 MiB scoped lease. This is training memory, not a runtime-memory result.

Native sound identification still fails on the unchanged evaluation probes.
Lower teacher-forced loss does not qualify semantic recognition. Diagnostic
reference generation also misidentifies dog barking as bird calls and church
bells as music. Removing audio boundary tokens alone does not repair these
failures. Further training and paired held-out/ASR evaluation are required.

The training-to-native audit found a separate, measurable encoder defect.
See [the native frame/floor repair](audio-encoder-alignment-20261030.md). The
main reasoning checkpoint, native vision weights and Qwen3-TTS Base remain
unchanged. No additional reasoning model is added to either bundle.

Original compact Ornith and legacy full-Omni captions for the same 240 new
validation clips are retained in run-local evaluation receipts. Their HTTP
completion status is not treated as an accuracy score. The new 400-clip test
split remains unused for candidate selection.
