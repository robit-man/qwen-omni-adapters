# Conditional primary-trunk adapter investigation

Publication remains gated on semantic parity. No adapter described here is a
qualified release, and no new runtime path is enabled.

Ten fixed projection checkpoints and two independent temporal acoustic heads
failed their mandatory gates. A paired encoder-tap diagnostic also failed its
preregistered improvement test. These negative results are preserved in the
[trial review](class-guided-trial-review-20261031.json).

The next bounded hypothesis is that a small conditional delta in the existing
primary trunk can interpret its frozen audio representation more effectively.
This adds no language runner. The exact base checkpoint, encoder, rich850 audio
projection and Qwen3-TTS cloning weights remain frozen.

The Ornith experiment uses rank-eight, alpha-eight LoRA on gate/up/down FFNs in
layers 24–31: 3,145,728 trainable parameters. The pinned converter exports all
48 tensors, whose F32 values match the source adapter exactly. An untrained
zero/nonzero compatibility test proves actual native activation and explicit
off/on/off cache switching. Across two ordinary-text prompts, disabled and
zero adapters reproduce the vanilla top-ten token log probabilities exactly;
the nonzero fixture changes shared-token probabilities. This is a 4K-context
compatibility check, not a media or model-quality result. See the
[native compatibility receipt](audio-lora-compatibility-20261031.json).

A preregistered 128-step fit canary uses the same 64 previously selected event
training recordings, with 16 separate training-split speech controls receiving
no optimizer updates. Event-name CE falls from 5.23237 to 0.16225, remaining-token
CE from 1.84857 to 0.31561, and speech-control CE from 0.25547 to 0.25310. All
fit gates pass. Base parameter versions and gradients remain unchanged, and
disabling the adapter restores exact base logits. Peak Torch-reserved memory
is 22,214 MiB; the measured fit/evaluation phase takes 151.20 seconds. Fit on
seen examples does not establish generalization.

The actual native 16K audio probe remains unqualified. It identifies clapping,
gunfire and a passing siren, retains the exact sentence fixture and recognizes
a bell/chime within a mixed description. It still mistakes dog/alarm/printer
audio and hallucinates a sound for silence. No inference prompt or label-based
routing is introduced. The adapter stays private and disabled in production.

One full-corpus trial therefore starts from fresh zero-B adapter parameters,
retaining the canary architecture, AdamW learning rate, clipping and audited
fourfold event-token weighting. Its immutable manifest contains 12,458 training
records: 7,137 event recordings, 4,859 speech replay and 462 nonspeech replay.
It processes 3,200 serial-four updates with fixed native checkpoints at
1,000/2,000/3,200. Unknown FSD transcripts remain unscored. No held-out test
recording is used. Loss cannot promote a checkpoint; exact transcription,
acoustic identity and silence gates, full ASR/environmental comparisons and
final Orin qualification remain required. Qwen needs its own exact-checkpoint
compatibility, training and A100 qualification.

The source suite passes all 6,303 tests before and after this investigation.
The old Orin services remain active and their portal returns HTTP 200.
