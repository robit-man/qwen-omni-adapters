# Class-guided projection repair: data accepted, training active

Both repair jobs started on reserved A100s at approximately 03:34 UTC on
2026-10-31. **Neither model is qualified for publication.** The source corpus
and supervision upgrade is complete; semantic qualification remains open.
The [machine-readable checkpoint](class-guided-training-20261031.json) retains
the dataset identity, predeclared selection policy and observed CUDA jobs.

The immutable manifest has SHA-256
`f391b0c7dc903a78c7c3a2eef8eb9b261c1e5a521a929b361dc6a14aa1474b68`:

| Split | Records |
| --- | ---: |
| Training | 12,458 |
| Validation | 385 |
| Test | 555 |

All prior rich-target validation lines and every original test line remain
byte-identical. Training includes 7,137 FSD recordings, original replay and
2,000 additional natural utterances from 250 disjoint speakers. All FSD speech
targets use the [audited partial-label collator](partial-audio-supervision.md).
No teacher transcript is accepted as verified speech.

Target triage retains 3,733 descriptive pseudo captions and uses human event
labels without teacher descriptions for the remaining recordings. It does not
drop uncertain recordings. Sixty `Camera` annotations remain in source metadata
but their target category is normalized by exact ontology ID `/m/0dv5r` to
mechanical/electronic device sounds, respecting the fixed perception policy.
Generated training targets contain no camera claims. The actual preparation
source and its digest are retained with the immutable manifest.

Ornith starts from the previous 850-step checkpoint under a 32,000 MiB GPU0
lease; Qwen3.8 starts from its 750-step checkpoint under a 60,000 MiB GPU1 lease.
Both loaded with no missing, unexpected or mismatched language weights and
proved CUDA residency on their reserved UUIDs. Only the existing final audio
projection trains. The reasoner, audio encoder, native vision and Qwen3-TTS
weights remain frozen. Both jobs perform up to 3,200 serial four-example
accumulation steps at learning rate `3e-5` and seed 42.

Explicit human event labels change the training target's wording distribution.
The unchanged validation captions can therefore penalize wording independently
of recognition quality. Checkpoints at steps 1,000, 2,000 and 3,200 were
preselected before training for native evaluation: the fixed 13 repair
diagnostics, the same 160 ASR/control inputs, and full 240-clip environmental
validation. NLL alone cannot select a release. No labels or titles enter native
inference requests; unused test data must not select a checkpoint. Final
qualification remains Qwen3.8 on A100 and Ornith on Orin.

A separate data-only repeat of the earlier pooled acoustic head remains
**rejected**. Its unchanged architecture/settings, expanded CC0 data and true
silence negatives improved common-class macro AP to 0.47724, versus 0.45839,
and repaired the bell/silence diagnostics. Top-one annotated accuracy was
36.67%, below the predeclared 42.5% gate, and dog/alarm diagnostics still failed.
No threshold was relaxed, no native integration was attempted and its lease was
released. Full-sequence projection training continues independently.

The existing Orin service and call harness remain active; their health check
returned HTTP 200 after SSH reconnection. Nothing was published to HF/Ollama,
and no existing deployment was changed.

The versioned partial-label integration passes all 6,303 source tests and every
script harness, including eight focused regressions. Its executable AST matches
the real-tokenizer-audited helper in the active jobs; neither job was modified
or restarted to publish this build support.

## First checkpoint: rejected

The immutable Ornith 1,000-step projection has SHA-256
`ad39f8cb8c04c360b454eece0b1a1075186a1280dcce0c435492d1a889fcb776`.
Only the two intended tensors changed; all 858 frozen audio/vision tensors
remain exact. Its native 16K/F16 A100 development diagnostic completed all 13
inputs with verified GPU residency. This is not final Orin qualification.

Clapping is now identified, and exact speech, silence, procedural alarm and dog
checks remain sound. Recorded clock alarm, bells, group voices, printer,
gunfire and siren still fail. The same checkpoint's full-precision model
reproduces the main errors: voices become birds, the printer becomes a vehicle,
and gunfire becomes an explosion. This evidence does not justify a runtime or
quantization patch. Both temporary diagnostic scopes were released.

The [semantic review](class-guided-step1000-rejection-20261031.json) rejects
publication. Larger ASR/environmental suites and unused test data are deferred
because a mandatory semantic gate already failed. Both existing training jobs
continue toward the remaining preselected checkpoints; no threshold changed
and no extra model variant was started at that checkpoint.

## Completed Ornith trial and bounded loss experiment: rejected

All three preselected Ornith checkpoints (1,000/2,000/3,200) fail mandatory
native semantic diagnostics. Qwen's 1,000/2,000 checkpoints also fail; its
original run continues unchanged to the preselected final checkpoint. The
[seven-checkpoint review](class-guided-trial-review-20261031.json) retains exact
projector identities, output digests and the actual observations. Every export
changes only the final projection's two tensors; 858 frozen audio/vision tensors
remain byte-identical. All native checks completed at 16K/F16 with verified
A100 residency. Ornith checks here are development diagnostics, not final Orin
qualification. No held-out test data selected a checkpoint.

A read-only audit of 64 seen Ornith training recordings measured mean event-name
CE 2.0889 versus 0.2238 for the remaining tokens. Teacher-forced and same-prefix
generation agreed on the first event-token argmax in all 64 cases. This
motivated one bounded 800-step continuation emphasizing event-name tokens by
four, using the same corpus, final projection, learning rate and serial-four
accumulation. Unknown speech stayed masked; transcription replay stayed intact.
The primary trunk, audio tower, vision and cloning weights did not change.

Before training, both real tokenizers aligned all 7,137 event-training records.
Actual-model loss/gradient checks passed on event, speech and silence recordings:
all-ones logit derivatives matched and the derivative of the loss difference
through the projection was zero. Separate identical BF16 backward passes varied
by 1.75%; the shared-graph derivative check controls that measurement issue
without relaxing tolerances. The weighted loss matched independent manual NLL.
The preflight made no optimizer updates.

Both fixed weighted checkpoints (400/800) are rejected. The final checkpoint
recognizes the recorded alarm, bark, crowd, applause and siren, but still calls
bells piano and a printer a siren, and changes "beside" to "behind" in the
sentence fixture. The loss adjustment is not a qualified repair and remains a
private build experiment. No new inference prompt, routing workaround or
production model was introduced. Full ASR/environmental suites remain mandatory
for a viable candidate; these failed early checkpoints were not advanced.

The complete source and script gates passed again: 6,303 tests before and after
the private experiment. Passing source tests does not clear model parity.
Publication and promotion remain blocked, and the old Orin deployment remains
unchanged.

Repeating the read-only audit on the same 64 training recordings after the
weighted trial lowers mean event-name CE from 2.0889 to 1.9238. Remaining-token
CE rises from 0.2238 to 0.2513; overall CE is 0.4351 versus 0.4303. The seen
printer's event CE falls from 6.2572 to 3.1040 but it is still called an engine.
This is partial training-fit movement, not a semantic pass. The 800-step trial
processes only 3,200 of 12,458 training rows. One further bounded pass now covers
the whole unchanged corpus: 3,200 serial-four updates, fixed native checkpoints
1,600/3,200, the same audited loss, architecture and learning rate, initialized
from the completed weighted checkpoint. Its launch makes no quality claim.
