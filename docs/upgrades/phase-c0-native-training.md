# C0 — native feature extraction and guarded head training

The attached-head milestone needs actual training on the deployed representation,
with independently held-out data and no additional language-model runner. The
repository previously had no standalone path for extracting that representation
or training and rejecting a CTC candidate against it.

`runtime/audio_feature_probe.cpp` loads only vocabulary metadata from the supplied
language GGUF. It never creates a language context, loads language tensors or
executes a language forward pass. Its existing multimodal encoder callback copies
the 1,280-wide activation after the final projector's first GELU, before the
backbone-specific final projection. It gathers all native eight-second chunks.
The encoder and all reasoning weights remain frozen.

## Verified representation and executable path

All 524 frozen audio tensors have the same names, shapes, quantization and bytes
across the original source and both backbone projectors. Their canonical tensor
digest is `8ee28b14130f8ad1dd5bdb38139090772c5217675e811ea85a8d309be9607210`.
The audio metadata also matches. The reasoning weights and final projections
remain distinct; nothing interchanges those backbones.

The maintained extraction entrypoint ran four real audio fixtures through each
of the original projector and both installed candidate projectors. All twelve
native feature outputs were byte-identical. Cases included silence and a
recording spanning multiple native chunks. The scoped worker's actual GPU
residency was checked before input admission.

PyTorch reconstruction was insufficient as an assumed substitute: the tested
silence activation had mean row cosine about 0.633 against native execution,
and speech differed particularly near padded boundaries. Native extraction
therefore supplies the training cache. No existing encoder math was changed.

`runtime/prepare_audio_features.py` requires a fresh output/readiness path and
an owned, exact-GPU reservation of at least 4096 MiB. It binds reusable features
to the producer binary, projector, source-audio and feature hashes. Keep source
files and manifests immutable during extraction. No temporary model component
or full-precision language GGUF is reconstructed.

Build the probe against the already patched native libraries:

```bash
g++ -std=c++17 -O2 runtime/audio_feature_probe.cpp \
  -I vendor/llama.cpp/include -I vendor/llama.cpp/ggml/include \
  -I vendor/llama.cpp/common -I vendor/llama.cpp/tools/mtmd \
  -L vendor/llama.cpp/build/bin \
  -Wl,-rpath,"$PWD/vendor/llama.cpp/build/bin" \
  -lllama-common -lllama -lmtmd -lggml -lggml-base \
  -o runtime-data/audio-feature-probe
```

Run `prepare_audio_features.py --help` for explicit model, projector, manifest,
binary, work and readiness arguments. This training helper currently targets
Linux with discrete CUDA and the broker. It is not a new deployment service.

## Real training and retained failed results

`runtime/train_audio_ctc_head.py` uses cached native features and build-only
Torch/safetensors/tokenizers dependencies. It requires one exact reserved GPU,
checks source/feature hashes and dataset binding, rejects cross-split content
aliases and impossible CTC alignments, and requires speech plus empty-reference
examples in both evaluation splits. It snapshots input identities, saves only
head weights and selects checkpoints using validation data. Test prediction is
disabled until the validation gate passes. A rejected candidate exits nonzero
and explicitly remains unqualified for release or runtime activation.

The pilot gate was fixed at validation mean speech WER <= 0.15 and empty-audio
false-transcript rate <= 0.05. This is an engineering pilot target, not a claim
that a head matching it can replace the full bridge or pass streaming parity.

| Candidate | Train features/data | Best validation mean WER |
| --- | --- | ---: |
| Frame-independent, 921,088 parameters | PyTorch, original corpus | 32.38% |
| Causal three-frame head, 1,708,032 parameters | PyTorch, original corpus | 22.91% |
| Same causal head | Native, original corpus | 23.13% |
| Same head plus 10,000 training clips | Native | 18.49% |
| Nine-frame causal context | Same expanded native data | 18.41% |
| Three-frame head, 2,048-token vocabulary | Same expanded native data | 18.05% |
| Same 2,496,000-parameter head, full 100-hour split | Native | 17.11% |

All selected candidates produced zero false transcripts on the six validation
noise clips. This small sample does not establish broad non-speech safety.
Eight-prefix sequence decoding changed the fixed 512-token model's WER only
from 18.49% to 18.44%, so it was not adopted. Common-layer initial parameters
were byte-checked for the causal-context ablation. All failures are retained.

The full-data candidate achieved 11.32% WER on the 29 LibriSpeech validation
clips, 7.43% on 78 procedural speech clips, and 36.74% on 47 synthetic
vocabulary-recovery clips. The remaining error concentrates in that last
domain. It still fails the combined pilot gate. No candidate test predictions
were generated, no head was attached to a tag, and final transcription continues
through the existing full bridge. Existing bridge test reports and this pilot's
validation results are different fixtures and are not a matched comparison.

The last maintained training run took 446.5 seconds with a 635.6 MiB peak in
Torch's allocator. That is not complete GPU residency or a whole-system peak.
Its feature cache covers 30,941 unique recordings. Head training reuses those
native activations rather than repeating the encoder every epoch.

## Data provenance and release status

The original corpus retains 3,636 records, with 160 validation and 155 test rows.
The expansion adds all 28,539 training clips from 251 speakers, totaling about
100.59 hours, with no speaker overlap with the original corpus. Validation/test
rows remain byte-equivalent under canonical JSON hashing. Data comes from
[LibriSpeech train-clean-100, CC BY 4.0](https://www.openslr.org/12/).
The 6,387,309,499-byte archive matched the
[published upstream checksum](https://www.openslr.org/resources/12/md5sum.txt);
its recorded SHA-256 is
`d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2`.
The verified archive, feature caches and unqualified checkpoints are retained
under the task's explicit `/srv/fine_tuning_suite/training_suite/outputs/`
directory. Weights are not committed.

C0 establishes the shared native training path. C1 quality, actual native head
execution, partial-streaming qualification, the other trained heads, whole-model
qualification and publication remain open. Hugging Face and Ollama publication
remain authorized after those release gates pass; none occurred in this step.
