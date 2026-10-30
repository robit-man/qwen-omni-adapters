# Environmental projection repair: richer acoustic targets

The second Ornith training run completed 1,600 steps, starting from the prior
400-step projection. Validation loss fell from 0.20773 to 0.13692, with the best
checkpoint at step 1,450. Its exported projector changes only the final audio
projection weight and bias; all 858 other vision/audio tensors remain identical.
The primary reasoning checkpoint and TTS weights remain unchanged.

Actual eight-fixture diagnostics recover the held-out dog bark and clock alarm
and preserve the clean spoken sentence. Music is still called rain and church
bells a Hammond organ; steady noise is underdescribed. This is a failed semantic
release gate, regardless of the lower training loss. See
[the diagnostic outputs](ornith-training2-diagnostics-20261030.json).

The training reference also fails to recognize several training-set barks.
Native/reference feature alignment and exact prompt framing are already checked;
the observed failure therefore is not solely GGUF conversion. A precision assay
found 98.96% of projection weights changed during training, so a proposed stuck
BF16 optimizer explanation was rejected without changing the optimizer.

The bounded next hypothesis is that richer acoustic descriptions provide more
useful semantic supervision than short ontology labels. The existing full-Omni
baseline supplied offline pseudo-captions for 2,440 CC0 training/validation clips;
it receives no held-out test clips and is never added to the deployment bundle.
The new manifest keeps 2,319 richer targets and all 3,636 original replay records.
It rejects 45 unverified teacher speech transcripts, 66 prohibited camera claims
and 10 malformed/extra-markup responses. All 555 test records are byte-identical
to the preceding manifest. See [dataset provenance and exclusions](rich-acoustic-targets-20261030.json).

These pseudo-captions are not independently verified human ground truth. Format
and policy filtering does not guarantee their acoustic accuracy. A new Ornith
projection-only run uses this manifest and the completed best checkpoint, with
all other training settings unchanged apart from the recorded seed. It remains
an experiment until actual sound identification, ASR replay and independent
held-out evaluation pass. Qwen's label-only run was intentionally stopped after immutable checkpoint 950
(of 1,600 planned steps), following the failed step-900 sound diagnostics. Its
CUDA process exited and its lease was released with broker verification; a
separate richer-target run starts from that preserved checkpoint. See
[Qwen diagnostics](qwen-training1-diagnostics-20261030.json).
Publication on either registry remains gated by the user's explicit choice to
repair parity first.
