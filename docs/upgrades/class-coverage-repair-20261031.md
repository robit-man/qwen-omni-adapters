# Auditory class coverage and partial transcript supervision

Publication remains gated. This checkpoint verifies a data and supervision
repair, not improved model quality. See the [machine-readable receipt](class-coverage-repair-20261031.json).

The original environmental corpus builder excluded `Human voice`, `Speech`,
`Singing` and their descendants before sampling. Its training split therefore
had no `Human group actions` examples even before caption filtering. Treating
the gap solely as a caption-filter problem was incorrect.

All six development archive volumes now pass their pinned whole-file MD5
checks. Four thousand additional CC0 environmental recordings and 937 CC0
human-activity recordings have been extracted with source CRC/SHA verification
and deterministic 16 kHz mono normalization. Only training data was added.
Protected uploaders, protected source hashes and 1,524 ESC source IDs remain
excluded. The human recordings include laughter, shouting, singing, whispering,
crowds and child speech; ontology parents are reduced to the most specific
annotated labels in the training targets.

Weak event labels do not establish a transcript. The new training collator
masks unknown FSD transcript targets instead of teaching an empty or invented
transcript. Its actual-tokenizer audit passes for both backbones: replay and
encoder tensors stay identical, the token spanning the speech/observation
boundary is masked, and acoustic/stop targets remain identical. Supplied
unverified words are rejected. The production media prompt is unchanged.

Caption collection and target triage remain in progress. The queued trial
retains human event labels, uses conservative teacher descriptions as explicitly
unverified pseudo labels, restores all 7,137 FSD training examples, and adds
2,000 speaker-balanced natural utterances rather than uniformly adding the
entire 28,539-utterance expansion. All prior validation/test lines must remain
byte-identical before training can start. Each family is scheduled for 3,200
serial four-example accumulation steps from its existing best projection.
Only that final projection is trainable; the reasoning trunk, audio encoder,
vision and cloning weights remain frozen.

An offline text grader was rejected after scoring only 21/24 on its predeclared
calibration. It incorrectly treated a generic mechanical hum as a printer, an
unspecified electronic tone as an alarm, and unspecified plucked strings as a
guitar. It will not score release parity, and its threshold was not relaxed.

The complete source validation passes before and after this work: 6,295 tests
plus the contract, VAD, call queue, browser cache, installer and script gates.
No production source or deployed service changed. Native auditory parity and
the Orin/A100 qualification remain open; no HF or Ollama publication occurred.
