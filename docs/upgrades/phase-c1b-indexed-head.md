# C1b — attach the trained head under the existing weight umbrella

Acceptance criterion: the qualified head must belong to each backbone's single
model tag and load directly from its indexed sidecar, while preserving every
existing TTS component and output. C1 exported a qualified standalone candidate;
the sidecar inventory and loader did not yet support its namespace.

The existing v2 envelope now accepts the optional eight-tensor
`audio_ctc_head` view. Its architecture and encoder ABI are inspected before
packing or resolution. The head carries its tokenizer, training-derived lexicon,
checkpoint and qualification identities. Older two/three-component v2 payloads
retain their behavior. The expanded inventory requires this matching runtime.

`pack-tts --audio-ctc-head FILE` includes the view and explicitly records it as
disabled. Default preparation still selects TTS/model, codec and reference
encoder only. An explicit head view can be materialized, or aliased on POSIX.
The native head loader uses the existing filtered GGUF selector and original
tensor offsets. It loads only eight head tensors from the same file containing
the sibling speech graphs. It executes no additional language model.

## Installed candidates and exact identities

Both new local candidates share sidecar
`sha256:d995ba1b768e5e3cc86a65e818601e8cae94ad06c6091330e54ba3db3be5d7d3`:

- `robit/ornith-1.5-omni:edge-v2-ctc-candidate-20261030`
- `robit/qwen3.8-27b-e03-omni:edge-v2-ctc-candidate-20261030`

Each tag retains its exact previous language/projector layers and receives its
own binding descriptor. The shared payload is 1,682,999,584 bytes, an increase
of 10,519,936 bytes over the reference-only sidecar. This adds a trained
capability candidate; it is not a storage or peak-memory reduction. Previous
tags and blobs remain available.

All four extracted components are byte-identical to their sources:

| Component | Tensors | Original GGUF SHA-256 |
| --- | ---: | --- |
| Qwen3-TTS Base | 311 | `8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129` |
| TTS codec | 378 | `6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2` |
| Reference acoustic encoder | 129 | `587b2b59ebfa815c5490fbc4bddb5dde3454cfbca4eef108d8688b668628a0e5` |
| Trained English CTC head | 8 | `e939e15685138903f1c49701ccba830dcac058c780d116b5319532bd892bca15` |

## Actual graph and voice regression

The head executed directly from the expanded source on A100 and Tegra using all
315 frozen native-feature fixtures. Every decoded transcript matched C1. A100
logits were byte-identical to its standalone reference; Tegra's maximum absolute
difference remained 0.00008392. The explicit installed-tag alias materialized
zero payload bytes. The probes load no sibling TTS or reasoning weights.

Both devices also exercised the actual speech routes with reference preparation,
automatic four-frame packets and continuous full-reference batches together.
Saved A/B/A, inline/saved equality, WAV/stream equality, reference reuse after
weight shedding, repeated continuation, cancellation and temporary-file cleanup
passed. Eight matching recordings on each device were byte-identical to B3b.
The batch receipt's `inline_batch_exact:false` records the intended difference
between a continuous utterance and concatenated independent blocks; repeated
continuous batches matched exactly.

These checks qualify packaging and component loading. Newly encoded end-to-end
ASR, live encoder sharing and streaming remain unfinished. The English head
stays disabled and the full audio bridge remains authoritative. Decision,
retrieval, recurrent state, complete two-backbone parity and Orin concurrency
qualification remain separate milestones. Hugging Face and Ollama publication
remain authorized after full release qualification, with no upload claimed here.
