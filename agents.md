# Qwen Omni Upgrade — durable user ground truth

Accepted 2026-10-29. This is the user's engineering mandate for this repository.
Read this file together with `AGENTS.md` before making changes. Preserve these
requirements across context renewal. Targets and proposed designs below are not
claims of implemented capability, compatible weights, or measured improvement.

## Native Ollama audio requirement — 2026-11-01

The user additionally requires the completed weights to work through native
Ollama ASR and TTS. Permission to use substantial implementation changes does
not waive parity, cloning, isolation, GPU leases or artifact limits. Qualify
actual `/v1/audio/transcriptions` and `/v1/audio/speech` calls against each exact
tag, including streamed and complete 24 kHz PCM and arbitrary reference clones.
Route presence, model capability labels and an adapter proxy are not native
runner compatibility. Preserve the existing adapter and one identity per trunk.

Official Ollama v0.40.13 (04088e7c9d41cba5d33bafbf788d87a465c79d5b)
adds LFM3 speech on Linux/Windows; installed v0.40.11 already registers both audio
routes. Neither fact proves support for our Qwen3-TTS or trained audio bridge.
Inspect model-specific loaders, prompt rendering, scheduler and native worker
interfaces before integration. Do not replace Base with LFM3/Phonon or rename
incompatible tensors. Record whether qualification uses stock or a pinned,
patched Ollama build. Develop and test isolated instances before any change to
the shared host Ollama/broker. Do not disrupt other owners or use GPU1.
See `docs/upgrades/native-ollama-audio-20261101.json` for pinned source evidence.

Real patched-native Ornith AND Qwen A100 mechanism gates now PASS, each2ASR calls and5
streamed speech calls, including both default voices, A→B→A and arbitrary inline
WAV plus transcript/acoustic conditioning. All5PCM outputs match the existing
native runtime byte-for-byte; one primary PID and exactGPU0 residency are proven.
The test server exited and its lease released normally. No public weights,
new audio files or shared-service changes. This is pinned patched0.40.13 support,
not stock compatibility or full model acceptance. Orin, WAV/cancellation,
languages, saved profiles, sampling/error behavior, performance and full parity
remain release gates. See `native/ollama/README.md` and the exact receipt in
`docs/upgrades/native-ollama-ornith-mechanism-20261101.json` and
`docs/upgrades/native-ollama-qwen-mechanism-20261101.json`.

Native build 0.40.13-robit-omni.2 now preserves the existing numeric TTS controls:
explicit top_k 0/top_p 0, temperature 0, seed −1 and maximum 2147483647 are accepted;
invalid values return HTTP 400 before scheduling/loading. Actual Ornith/A100
default, zero-filter and maximum-seed PCM match the existing runtime exactly;
random seed produces valid PCM. CPU Go checks cover12 parser-derived cases,
nonfinite values and pre-scheduling rejection; all6394 repository checks pass.
The private server and CUDA workers exited and the lease released normally.
See `docs/upgrades/native-ollama-sampling-parity-20261101.json`. This closes only
the numeric-control criterion, not whole native/voice/model acceptance.


## Active convergence goal and GPU assignment — 2026-11-01

The user explicitly requires training until audio parity/convergence and then
publishing the validated weights. Keep that full goal active across turns.
Use physical GPU0 (`GPU-3538758b-6ba7-fa1b-a80c-e315a606c3f0`) and GPU2
(`GPU-fc9d264d-8a20-bf8c-eb72-72a6dcbdaf2a`) only; leave GPU1 open for others.
Preserve all other owners and broker admission. This supersedes earlier
experiment GPU choices. Do not infer a process exited from its supervisor exit;
verify the actual CUDA worker has exited before releasing its lease.

Qwen's original GPU1 run was stopped on this instruction, exactly resumed from
5,000 on GPU0, and completed6,400. All48 exported tensors match; primary/encoder/
bridge/TTS remain frozen and disabled logits match exactly. Native semantics
still fail bells/printer/gunfire/siren/noise/tones/clock-alarm, so6,400 is REJECTED.
Full160 replay yields215/3,336 word errors(6.445%), zero false transcripts on six
nonspeech controls, but natural speech36/594 versus legacy9/594 remains open.

Qwen DATA-only trial `qwen-full-diversity-pass-1` completed6,400→7,400 on GPU0,
fixed1,000 updates, same62,314-row verified corpus as Ornith. Preserve Qwen's
ORIGINAL class-uniform-event plus once-per-row replay algorithm; do NOT claim
it uses Ornith's2/1/1 sampler. Original12,458 rows/targets are an exact prefix.
All48 tensors, AdamW/RNG and source position342 are verified; a new sampling
schedule is declared for the expanded corpus. Source parameters remain frozen.
The four-backward maximum-input preflight peaks55,684MiB after reset under60GiB
Torch/61.75GiB lease, zero optimizer updates. Whole-process training allocation
peaked60,902MiB and remained within the resource limits. The first1,000 updates
forecast3,007 novel IDs; this is not a full-corpus pass or convergence claim.
Native13/full160 and source42 assessments are now complete and rejected. See
`docs/upgrades/qwen6400-and-temporal-baseline-20261101.json`.

Latest Ornith DATA-only trial completed12,162 and is REJECTED. Noise/tonal
categories improve and alarm/printer recover, but raw silence, group voices,
clapping and gunfire remain wrong; continuous-alarm and melody wording also
misdescribe the intermittent alarm and steady three-tone chord fixtures.
Full160 replay is232/3,336errors(6.955%); natural speech41/594 versus legacy9/594.
All48 exported tensors remain exact and frozen checkpoint/disabled logits match.

Matched temporal control ran16 original validation clips twice in one native
PID, explicit adapter scale0/1, identical audio/prompt/frozen primary/projector.
BOTH repaired850 bridge and12,162 adapter miss ordered/mixed-source combinations
that full Omni handles. A read-only unquantized BF16 source diagnostic then ran
the same29 natural and16 temporal WAVs: source36/594word errors versus native41
and legacy9; temporal structure also fails in source. No labels/titles/answers
enter source generation, the prompt matches exact SHA9f7386..., and all weights
remain frozen. One source response has malformed tags (41tokens, endedEOS),
retained as a failure. These results do not establish quantization as the sole
cause and do not justify another arbitrary precision/prompt sweep.

Qwen7,400 is REJECTED: clapping improves but dog→cat, bells→piano and other
environmental errors remain. Full160 replay217/3,336 errors versus prior215;
natural speech37/594 versus legacy9; all six nonspeech transcripts empty.
A matched BF16 source diagnostic completed42clips: natural speech41/594 source
versus37native and9legacy. Source also misidentifies dog/cat, bells/piano and
printer/cupboard, so precision alone cannot explain the learned errors. Source
and native differ on additional clips; retain both gates and generation-backend
comparison limits. The zero-update audit of the exact16 previously seen TRAIN
examples completed:
eventCE1.4799 at3,200→0.5303 at7,400, remainingCE0.1280, zero first-event
argmax changes with/without future targets. Bark/siren/printer/group examples
recover; clapping/details still fail. Seen fit does not establish generalization
or isolate the effect of the latest data expansion. Retain frozen projection
and assess data-only virtual scenes next; do not stack an architecture change. See
`docs/upgrades/qwen7400-source-native-review-20261101.json`.
Previous Ornith12,162 training/evaluation workers exited. Its verified dead
training reservation was reclaimed; all peer scopes/budgets/contexts remained
unchanged. See `docs/upgrades/ornith12162-source-native-review-20261101.json`.
The exact12,162 sampler/order/RNG and17,389-record seen set are reconstructable;
a longer deterministic schedule preserves the completed4,000-draw prefix and
can avoid resetting the remaining speech pool. This is bookkeeping, not proof
that unchanged classification training will restore temporal comprehension.

Virtual TRAIN scene preparation is now COMPLETE:2,048 metadata-only recipes
(512event sequences,256event overlaps,512DSP sequences,256DSP overlaps,
256different-voice overlaps,256speech/background mixtures). The original62,314
records/targets remain unchanged; total TRAIN64,362. All parents are retained,
hashed TRAIN-only sources; no validation/test audio is used. All wave hashes,
whole-parent spans, grounded construction captions and unknown-transcript masks
pass. Six independent scalar PCM references match vectorized rendering; old
sampler/RNG and collator tensors remain exact. Metadata adds4,439,921bytes and
no audio/feature files. Weak source event labels and peak-aligned mixtures still
require perceptual evaluation; construction checks are not semantic parity.

Ornith's scene-data-only12,162→13,162 trial completed and is REJECTED.
Native13 still misidentifies alarm, group voices, clapping and printer. Fixed16
still misses ordered/overlapping sources and reverses noise→tones. Full160:
237/3,336word errors versus232previous; natural44/594 versus legacy9/594. All six
nonspeech transcripts remain empty. Raw-zero neural hallucination is retained;
the separate exact-zero runtime DSP is not relabeled as a neural pass. All48
exported tensors are exact; primary/encoder/rich850 bridge/TTS remain frozen and
disabled language logits match. Training peak21,456MiB stays below21GiB cap.

CPU reconstruction proves the saved126,616-index order/RNG and cursor4,000.
Only524/2,048 virtual recipes were drawn (309sequences,209overlaps,6speech/
background);1,524recipes remain unvisited. Same TRAIN maximum-scene probe loss
falls2.114→0.234, but teacher-forced TRAIN fit is not held-out semantic parity.

Ornith's exact-order13,162→15,696 continuation completed and is REJECTED.
All48F32GGUF values remain exact, frozen-source/disabled-logit checks pass, and
all three actual recovery checkpoints14,162/15,162/15,696 pass tensor/AdamW/order/
RNG audits. Actual final cursor14,136 yields25,644cumulative unique IDs and1,810
virtual scenes: all1,792acoustic recipes, only18/256speech/background recipes.
The forecast matches actual files exactly;238speech/background recipes remain
unseen. This is an acoustic replay cycle, not a complete scene/speech pass.

Native13, fixed16temporal and full160 all completed. Silence and clapping recover;
clock alarm/bark/church bell/siren remain correct. DSP noise/tones directions now
appear in both orders, with incomplete tonal detail. Alarm/chord are still called
melody; voices→crow; printer gives malformed Markdown and incorrect engine/car
content; gunfire→explosion. Event pairs still lose identity/order and mixed speech
fails. Natural speech improves44→40/594errors, still above legacy9; aggregate
237→248/3,336errors. Six nonspeech transcripts remain empty. Keep publication gated.
See `docs/upgrades/ornith15696-scene-review-20261101.json`.

Its CUDA worker actually exited. Normal lease release failed after foreign
contexts changed during the run. Known owner/PID/scope, independent no-CUDA proof
and completed files justified the documented already-gone-owner reclamation.
All peer tokens/scopes/budgets/registered contexts were preserved. The initial
native admission and first full-suite observer failures remain immutable;
recovered native13/temporal16/full160 completed under fresh admission, then
normally released. Do not infer training failure from that wrapper exit code.

The bounded Ornith comparison completed onGPU2, one trunk and sequential arms,
fixed500updates/identical2,000examples from the saved cursor14,136→16,136.
A continues exact15,696LoRA+AdamW with rich850projection frozen (ends16,196).
B freezes the exact15,696LoRA and learns only the existing linear projection
with fresh AdamW/F32masters/BF16forward/effective export. Primary, audio encoder,
TTS, data, prompt, loss4 and LR1e-4 remain unchanged. This is trainability of an
existing component, not a new graph. Forecast788novel IDs,247scene draws and5
speech/background draws; no full-corpus claim. CPU cached extractor, actual
4,096×1,280bridge and all48checkpoint shapes versus24actual HF metadata modules
pass. HF module lookup is `model.layers`; portable adapter keys retain the
`base_model.model.model.language_model.layers` export namespace.

Two zero-update configuration failures are retained: copied extractor name
was wrong, then HF lookup incorrectly used the portable namespace. Both were
corrected before training. Third zero-update preflight passes eight maximum-input
four-backward tests, initial audio logits/loss exact, peak21,106MiB under21.5GiB
Torch/22.5GiB lease. Same source hash6356da65... launched. Forecast0.5GiB of compact
learned/recovery states; no new audio, feature cache or full-projector GGUF.
Observer waits actual trainer exit and normal release, then fixed58raw cases
per source arm (13diagnostic/29natural/16temporal) against the exact completed
15,696native input hashes. Source checks cannot replace native/assigned-target
full parity. Do not change a healthy worker or weaken the original gates.

Both Ornith paired arms and the fixed116source requests are COMPLETE and
REJECTED: A natural40/594errors, projection B50/594, legacy9. A retains correct
silence/alarm/noise/chord/dog/clap/printer/siren, but bells/voices/gunfire and
ordered/mixed sources fail. B loses dog/group/siren and transcribes beside as
behind; projection-only training is insufficient. GPU2 source admission failed
before CUDA because a peer increased its reservation. First GPU0 source attempt
then exposed a copied Qwen5120-wide bridge path; it generated no results. Both
failed records remain. Actual original/learned4096×1280 bridge CPU forwards,
shapes and hashes now pass, and a new scopedGPU0 run completed all116 requests.
Training/checkpoints were not changed. Both actual workers exited and all own
leases released normally. No full projector GGUF or public weights were created.
See `docs/upgrades/paired-projection-review-20261101.json`.

The next bounded audio hypothesis is now ACTIVE: conditional rank8 adapters on
the existing full-attention q/k/v/o projections in Ornith layers27/31. This adds
491,520parameters to the same trunk; it adds no separate reasoning model. A
continues exact16,196FFN+AdamW; B starts from the same FFN/moments and adds
zero-residual-initialized attention adapters with fresh moments. Both receive
the identical next2,000saved-order TRAIN records, cursor16,136→18,136, fixed
500updates, accumulation4, LR1e-4, event factor4 and clip1. Original rich850
projection, primary, audio encoder and TTS stay frozen. Forecast750novel IDs,
27,182cumulative IDs per arm; this is not a full-corpus/convergence claim.

CPU shape/gradient/zero-residual checks pass. The actual CUDA zero-update
preflight passes eight maximum-input four-backward tests, identical initial
audio logits/loss and exact disabled text logits. Peak21,072MiB is below the
23GiB Torch cap/24GiB lease. Source hash29a3ac1e... is pinned in the preflight
and running trial. GPU0 worker3081309 runs the fixed comparison; GPU1 is excluded.
GPU2 remains available only subject to its other owner's reservation. Do not
change or restart the healthy training process.

The finite observer3221201 waits for actual trainer exit and normal lease
release, then runs the same58held-out raw clips per arm on one frozen source
trunk. Actual4096-wide bridge and raw hashes are verified; its WER formula
reproduces the prior40/50errors using the unchanged authoritative normalizer.
Its evaluator/helper hashes are pinned. It never exports/promotes automatically.
B has64tensors and requires new export/binding/runtime qualification: the current
48-tensor component ABI must reject it. Native q/k/v/o shapes and existing LoRA
operators are read-only feasibility evidence, not proof of compatibility or
quality. See `docs/upgrades/attention-adaptation-preflight-20261101.json`.

The original control A completed500updates at16,696, position18,136; its48
AdamW steps and unchanged saved order/RNG are audited. Its checkpoint is valid
and must not be retrained. A loader bug was then found: noncapturable AdamW
CPU step tensors alias the dictionary passed to `load_state_dict`. The in-memory
source counters advanced during A, so original B's first4updates had FFN step
16,700 while its metadata said16,200. Those B4weights are explicitly excluded
from quality and packaging. Original files remain; the known worker cooperatively
stopped after preserving A500, exited, and normally released its lease. No peer
was signaled. This is a control-integrity repair, not a quality rejection of A.

Corrected B-only training now runs onGPU0, PID3784621, from the original16,196
weights and an independent deep copy of the original optimizer. CPU regression
proves source steps/moments stay immutable. The fresh zero-update CUDA preflight
passes all8checks and its initial forward matches the original control exactly.
Script56cdf6d6... is pinned; data/order/LR/loss/precision remain unchanged.
Do not restart this healthy recovered worker. Observer3784091 waits for actual
exit/release, then compares preserved original A500 with recovered B500 on the
fixed116source requests. Its pins and complete lineage are in
`docs/upgrades/attention-optimizer-recovery-20261101.json`.

The64-tensor CPU conversion fixture now preserves every F32 value and exact
base-linear shape, including nonzero query/key/value/output sentinels. The
current48-tensor validator correctly rejects it. This synthetic fixture is
never a model candidate and must never be attached/uploaded. Native generation,
the extended required binding/runtime contract and semantic gates remain open.
See `docs/upgrades/attention-export-feasibility-20261101.json`.

Qwen's scene-data-only7,400→8,400 trial completed and is REJECTED.
Native13, fixed16temporal and full160 all completed against unchanged raw audio,
prompt, F16 KV and explicit audio-adapter scale1. Noise/printer recover, but dog
is meow, bells are guitar/strum, clock alarm is insect, voices are vehicle horn
and clapping is gunfire. Opposite-order pairs collapse to the same observation
or omit a source. Speech-plus-printer fails. Full160:215/3,336word errors,
natural36/594 versus37previous and9legacy; all six nonspeech transcripts empty.
This equals prior6,400 aggregate/natural error counts, not convergence. All48
F32 exported values remain exact; frozen source/disabled logits match. Actual
coverage18,279unique IDs, only4,000draws of64,362saved indices. See
`docs/upgrades/qwen8400-scene-review-20261101.json`.

The next bounded hypothesis compares tail-LoRA continuation against training the
existing final audio projection, which has stayed frozen at rich750 throughout
expanded-data LoRA training. Both arms start from exact8,400 and identical BF16
audio logits/loss, use the same next2,000 saved-order examples/cursor4,000→6,000,
500updates, accumulation4, loss4, LR1e-4 and AdamW. A restored AdamW8,400 and trained
only tail LoRA. It completed8,900/500updates and its48F32weights/AdamWsteps/moments,
unchanged RNG/order/cursor6,000 and19,665seen IDs pass the CPU recovery audit.
Both arms and the matched116-request source assessment are now COMPLETE and
REJECTED. Natural speech errors are39/594 in A and31/594 in B, versus legacy9.
B recovers bells/printer/gunfire but hallucinates noise as speech, dog as rooster,
clock alarm as fan, and loses sequence/mixed-source evidence. Projection learning
improves one measure without restoring full parity. Do not export or publish a
full projector based on this result. See
`docs/upgrades/qwen8900-control-recovery-20261101.json`. B freezes the exact8,400 LoRA and trains only the existing linear
projection with fresh AdamW and F32 masters cast to BF16 on forward/export.
This is one changed trainable component, not a new graph, separate reasoning
model, inference prompt, dataset or sampling policy. One primary trunk serves
the two arms sequentially. CPU effective-BF16 export/gradient checks pass; exact
initial source logits/loss match on CUDA. Eight maximum-input four-backward checks pass55,712MiB, zero optimizer updates,
under61.25GiB Torch/63GiB lease onGPU0. Fixed500updates per arm completed
in `qwen-projection-comparison-1`; its actual training worker exited and normally
released its lease. The initial63.75GiB
lease was refused before starting CUDA; no peer was stopped or reservation
weakened. Forecast0.5GiB; no new audio, feature cache or combined-projector GGUF.
The exact preflight script hash is pinned. A finite observer waits actual
trainer exit and normal lease release, then runs fixed13diagnostic+29natural+
16temporal raw clips on each source arm, same prompt/backend, no labels in
generation. No automatic full-projector export or promotion. Neither arm is a
release candidate until unchanged source/native,
full audio/media, assigned-target/cloning/concurrency gates pass.

An actual deployed Orin clone baseline completed36 serial requests, two bundled
reference voices, repeated A→B→A, same seed42. Every streamed PCM matches final
24kHz mono WAV; all A repeats are exact and B differs. Primary PID/service and
harness stayed unchanged/resident;36different request-scoped TTS CUDA PIDs.
Median first PCM is5.035s female/5.061s male. Minimum MemAvailable3,556.8MiB.
This is a bounded direct-synthesis latency baseline, not speaker similarity,
ASR, physical playback or overall model parity. Deployed cadence is8frames;
the raw continuity diagnostic used2frames, so actual-cadence qualification is
explicitly still open. No audio files were saved. Latest shared-host available
memory5.5GiB is below safe additional TTS headroom; do not start new Orin loads
or restart the healthy service merely to repeat the baseline. See
`docs/upgrades/orin-deployed-clone-baseline-20261101.json`.

The complete Ornith data-only pass ended10,662. Every fixed checkpoint is
rejected despite recovery of alarm, clock alarm, bark, church bells, printer and
siren. Group voices, clapping, gunfire and negative controls still fail. Qwen5,000 and6,400
also fail parity. Preserve `docs/upgrades/completed-audio-checkpoints-20261101.json`.
The user authorizes further evidence-driven training, not weaker gates or failed
experimental publication. The prior finite trials remain immutable; subsequent
hypotheses have separate source/data/policy/checkpoints and bounded assessments.

Fresh target audit: leaf-only supervision reduces provided human-group labels
208→7 and Music2,262→2 on the unchanged17,045 TRAIN rows. A target-only500-update
Ornith ablation ran10,662→11,162 on GPU2: retain annotated parents before
specific categories. Audio, IDs, splits, replay, loss factor, architecture and
optimizer remain fixed. The sampler uses the ORIGINAL immutable manifest view
because its algorithm reads target metadata. Original197 classes, RNG/order and
position3 were independently reconstructed on CPU. Its four-backward maximal
input preflight peaks21,074MiB under21GiB Torch/22GiB lease. Work:
`ornith-annotated-parent-target-pass-1`; its native13 observer waits for trainer
exit to avoid exceeding GPU2's shared promises. Admission-only retries must
never restart an initialized training job or fall back to GPU1.

The parent-first trial completed11,162 and is REJECTED: bells and printer
regress and group/clap/gunfire remain wrong. Full fixed160 replay raises speech
WER7.014%→8.064%. Baseline10,662 natural-speech WER is36/594 versus legacy9/594;
overall lower WER is not natural-speech parity. Preserve
`docs/upgrades/parent-target-and-control-balance-20261101.json`.

The sampling-only trial `ornith-control-balanced-pass-1` completed11,162 and is
REJECTED. Clapping recovers, but alarm and printer regress and group/gunfire/
silence/noise/tones still fail. Six native identities pass. All48 GGUF tensors
are exact; frozen weights and disabled logits remain unchanged. The native
admission retry followed an already-exited-owner lease cleanup: PID898715 was
the sole registered context, its process and CUDA residency were absent, two
normal releases failed on changed foreign allocation, then only that dead
owner's reservation was reclaimed with documented force. Other owner tokens,
scopes, budgets and registered contexts were verified unchanged. Never use this
as permission to force admission or release a live/unverified owner.

The completed bounded DATA-only trial `ornith-full-diversity-pass-1`:
11,162→12,162 on GPU2, fixed1,000 updates. It retains the2event/1speech/1control
sampling rule, rank8 last8 FFNs, event loss factor4, AdamW/LR and all frozen
primary/encoder/projector/TTS weights. Exact48 parameters, optimizer/RNG and the
14,740-record source seen-set are verified. The62,314 TRAIN rows comprise29,174
human event recordings,31,398 speech recordings and1,742 procedural controls.
Load the original/natural/attributed/DSP manifests as a verified union in RAM;
duplicate rows must be identical and protected rows remain unchanged. No audio
or combined-manifest copies, feature caches or full projector exports.

Added DSP controls cover10 varied signal families,128 each, about123MiB total.
A pre-training audit corrected37 captions for single short beeps/nearly steady
sweeps; all audio bytes were preserved and no optimizer ran on those captions.
CPU collator/sampler checks pass, including unknown-transcript masking,
unchanged real encoder inputs and exact2/1/1 groups. The corrected-data
four-backward maximum-input preflight peaks21,066MiB under21GiB Torch/22GiB
lease, zero optimizer updates. Its exact dataset/source hashes gate training.
The observer waited for trainer exit; completed native/source reviews are above. See
`docs/upgrades/data-diversity-convergence-20261101.json`; no release is qualified.

An additional26,539 already-local human-transcribed train-clean100 FLAC clips
are hash/header/speaker-split verified and prepared as a separate data-only
manifest `natural-speech-diversity-data-20261101`,43,584 TRAIN rows total. The
original manifest bytes and all targets/splits are preserved; no audio download,
copy or cache. License CC BY4.0. It was NOT used by the sampling-only trial;
it is activated by the current data-only trial together with attributed events.

An additional17,450 attributed DEV recordings are verified and prepared from the
already-local checksum-verified archives, with raw/normalized hash, protected
uploader/ID and ESC exclusions and per-file creator/title/source/license credits.
Only CC-BY3.0 was added; other license types remain excluded. New bytes are about
4GB and no archive/model downloads or feature caches were created. This corpus
was NOT activated in the target-only ablation. It is now trained in the
data-only trial, so ship its
`attributed-audio-diversity-20261101/ATTRIBUTION.jsonl` with release provenance.
Do not silently combine target and data changes or omit those credits.

See `docs/upgrades/data-diversity-convergence-20261101.json` for current PID and
`docs/upgrades/gpu0-gpu2-convergence-20261101.json` for earlier
lease, resource, restoration and fit evidence. The20-record source fit diagnostic
covers16 TRAIN and four already-used validation clips; it is not broad parity.
One teacher-versus-prefix argmax changed, with cause unproven. Do not infer that
loss or exact export proves a deployable semantic model. Continue the full native,
legacy, ASR/environmental, assigned-target, cloning and operational gates before
publishing and verifying both Hugging Face and Ollama. Do not mark the goal done
merely because a bounded training job, source test or small diagnostic completes.

A separate fixed16 compound VALIDATION suite now covers12 ordered/reversed
sequences and four overlapping sources, including speech plus alarm/printer,
group voices plus siren, and claps plus bells. All samples derive from existing
held-out validation audio. Every source and compound hash is disjoint from the
actual59,800 TRAIN waveforms across all current/prepared corpora, independently
rehashed; all PCM and source timestamps are verified. New bytes4,720,948 only.
Do not train on these derivatives or redistribute their evaluation-only source
audio. Legacy/candidate CPU preparation passes; native semantic/temporal
inference is NOT run or qualified. See
`docs/upgrades/temporal-overlap-validation-20261101.json`. Use this as additional
parity evidence, never replace the original gates or infer quality from setup.

GPU0 serial qualification completed after the actual Qwen6,400 workers exited.
The fixed16 legacy temporal/overlap reference is retained in
`legacy-temporal-overlap-reference-1`. It preserves several event-order reversals
but has source confusions/omissions in other sequences and overlaps. Do not
change fixtures after observing outputs or claim the baseline passes all16.
Raw equal-gain stems are not always equally loud; the level audit is analysis,
not perceptual audibility qualification. Candidate comparison is still pending.
The legacy reference, completed evaluation workers and observer have exited;
GPU0 is now used by the new Qwen data-only trial described above.

The active attributed-data release audit verifies all17,450 exact creator/title/
source/license/change records against the actual TRAIN metadata. Canonical
`ATTRIBUTION.jsonl` SHA256 is
`375ce985679b19c92853ed0bdc5ffb26bd966339489820e202edca145bdcfda9`.
Ship this exact ledger with Hugging Face provenance and carry the full applicable
notices with the Ollama bundle; verify remote artifacts before calling
publication complete. No model publication follows from the attribution audit.

Release notice preparation now verifies the exact17,450-record CC-BY-3.0
TRAIN cohort before writing output. `qwen_omni_adapters.release_notices` rejects
changed ledger hashes, missing/extra/duplicate/held-out credits and mismatched
creator/title/source/license fields. Both HF staging directories share hardlinks
to the exact ledger and complete6,913,636-byte standard Ollama notice. Original
component/LibriSpeech notices remain separate. Stock0.40.11 private create→push→
pull→show preserved the notice bytes, inherited license, sidecar and required
adapter. No CUDA inference or public registry was used. A final resolver-key
mistake in the test was independently recovered against actual downloaded files;
retain that scope in the evidence. Full source gate6,394passes (14new notice
integrity tests). Follow `docs/build-and-release.md` and
`docs/upgrades/training-release-notices-20261101.json`. Notice preparation does
not clear model quality gates or authorize failed previews.

## Resource requirement, clarified 2026-11-01

The user requires completing the upgrades, tracking artifacts and preventing
their accumulation, then publishing the validated final bundles to Ollama and
Hugging Face. Maintain a per-experiment inventory, forecast temporary writes
before starting work, and reclaim completed disposable scratch with receipts.
Do not treat low Linux `MemFree` as exhausted RAM: admission uses `MemAvailable`.

Before any new build, export, training or qualification job, run
`scripts/artifact_guard.py check` for every filesystem it writes. On this A100
host retain at least 10 GiB on the checkout filesystem and 50 GiB on `/srv`,
in addition to forecast writes; keep the combined upgrade artifact roots below
120 GiB including the next job's writes. Declare host RAM requirements and retain
32 GiB of available host RAM. CUDA admission independently requires the existing
GPU broker. These are host build limits, not new portable-runtime defaults.

Preserve canonical model blobs, selected release candidates, all learned
checkpoints, raw training audio, licenses, manifests, source revisions and compact
qualification evidence. Completed rejected experiment exports are disposable
only when their exact learned checkpoint, reconstruction inputs, exporter and
hashes are retained and verified. Disposable feature caches and stopped
materialized views may also be reclaimed. This does not authorize deletion of
pending release candidates or sole copies of learned weights before verified
registry uploads. Never recursively prune caches, repositories or Ollama blobs.

Use the explicit-file cleanup ledger in `scripts/artifact_guard.py`: dry-run
first; require unchanged file identity and SHA-256, retained provenance, no
shared hardlinks or symlinks, and no observed live file handles/mappings. Retain
the cleanup plan and deletion receipt. Finish each job by classifying and
reclaiming its disposable outputs instead of retaining another cache generation.

## Publication requirement

The user explicitly requires publishing both new backbone releases to Hugging
Face and Ollama after validation. Publication is authorized once the release
qualification gates pass. Preserve checkpoint identities, licenses, manifests,
digests and validation receipts; upload the complete corresponding weight
umbrella to each registry and verify the remote artifacts and digests. Candidate
labels, source tests and partial speech benchmarks do not satisfy qualification.
Keep publication in the milestone ledger until both registries are verified.

Hardware assignment, clarified by the user on 2026-10-30: qualify Qwen3.8 on
the current A100 host and Ornith 1.5 on the Orin. Do not force Qwen3.8 onto the
32 GB Orin or treat that host's memory-limited attempt as its release target.

The 2026-10-30 live audit passes operational/cloning checks but fails semantic
environmental-audio parity for both bridges. Qwen3.8's initial mixed-video
transcription failure was repaired and verified on A100, with an Orin
regression check. See `docs/upgrades/release-qualification-20261030.json`. Nonempty
audio observations are contract checks, never sound-identification acceptance.
Complete weight packages are staged locally. Public publication remains gated;
the user explicitly chose **repair parity before publishing** on 2026-10-30.
Do not publish these failed candidates as experimental previews. Repair and
rerun the semantic gates before either registry is promoted.

The repair audit also found native encoder and prompt mismatches. The runtime
now carries valid audio frame counts through CNN padding, matches the Whisper
mel floor, preserves short-tail CNN batching context and supports the explicit
`robit.qwen3a.trained-prefill.v2` projector ABI. See
`docs/upgrades/trained-audio-prefill-20261030.md`. Use all 16 pinned native patches
and the startup ABI guard when evaluating newly exported projections. Exact
framing and passing PCM/source checks do not clear the remaining semantic gate.

Per-clip native video limits now control actual decoded frames and token counts;
see `docs/upgrades/video-sampling-20261030.md`. Completed Ornith training recovers
bark/alarm but still fails music/bells. The richer-target experiment is recorded
in `docs/upgrades/rich-acoustic-targets-20261030.md`; pseudo-captions, loss changes
and repair diagnostics do not substitute for independent release qualification.

The 2026-10-31 UTC richer-target qualification recovers bark and bell/chime
recognition on Ornith/Orin: 14 operational checks, the original five semantic
fixtures, raw PCM and concurrent listening/viewing/cloning pass. Actual original
v1 speech WER improves from 18.645% to 6.894% on the same validation inputs.
Broader environmental errors and Qwen's missing bell recognition still gate
publication. A five-fixture result is not complete feature parity. See
`docs/upgrades/rich-projection-qualification-20261031.md`. The old Orin deployment
was restored and verified healthy after the bounded qualification session.

Indexed v2 publication must include the small hashed binding JSON layer from
`qwen-omni attach`: stock Ollama push strips descriptor annotations. The actual
push/download/resolve/inference repair is recorded in
`docs/upgrades/registry-binding-20261031.md`. Keep annotation-only compatibility,
reject damaged/conflicting bindings, package the edge 16K default explicitly,
and verify the pulled package. This transport gate does not clear audio parity.

Required conditional audio LoRAs now use a separate small GGUF component inside
the same tag and a hashed v3 binding. Bind exact primary/projector digests; keep
all learned tensor values unchanged and the shared TTS sidecar intact. Both
supervisors load the adapter disabled; only audio perception uses scale1, while
vision, language, thinking and tools explicitly use nonempty scale0. Missing,
damaged or transplanted components and older runtimes must fail closed. A global
Ollama ADAPTER would alter text/vision and is forbidden for this route. Both A100 family mechanism gates and the stock-transport gate use rejected weights and do not approve a release.
See `docs/upgrades/conditional-audio-component-20261101.md`. Do not repeat these
mechanism checks absent a relevant new change; semantic qualification remains
mandatory before selecting or publishing any trained adapter.

## Purpose and non-negotiable constraints

The next auditory repair checkpoint is
`docs/upgrades/class-coverage-repair-20261031.md`: the original dataset builder
excluded most human voice classes before sampling. Additional CC0 training
audio and unknown-transcript loss masking are verified, but no new model
quality is claimed. Preserve held-out lines, retain human weak event labels,
and never train unverified teacher transcripts as truth. The rejected 21/24
text grader must not be used to clear the parity gate.

The completed data audit and ongoing projection jobs are recorded in
`docs/upgrades/class-guided-training-20261031.md`. Their manifest is immutable;
all FSD transcript placeholders are unscored. Camera source annotations remain
in metadata but use acoustic-only device labels in training targets. Compare
the preselected 1,000/2,000/3,200 checkpoints by native semantics and ASR, not
wording-sensitive NLL alone. The expanded-data pooled head also failed its
fixed gates and must remain disabled. Neither job's launch proves parity.

The completed Ornith 3,200-step trial rejects every preselected checkpoint.
Qwen's 1,000/2,000/3,200 checkpoints also fail. Its completed final checkpoint
also fails the same core semantic tasks in full BF16, so export/quantization
alone cannot explain the failures.
A single audited 4x event-name-loss continuation on Ornith completed 800 steps;
both fixed 400/800 checkpoints still fail bells/printer recognition and the
final sentence fixture. It is a rejected private experiment, not a runtime
upgrade or a qualified model. Preserve the
[actual fixed-checkpoint review](docs/upgrades/class-guided-trial-review-20261031.json).
Do not promote by NLL, weaken semantic gates, alter inference prompts to match
fixtures, or publish these failed candidates. No held-out test data was used.
The same-64-recording audit after weighting lowers event CE from 2.0889 to
1.9238, but the printer remains wrong. The partial pass processed only 3,200
of 12,458 training rows. One complete-corpus pass completed with the same
audited helper, architecture and learning rate, fixed 1,600/3,200 semantic
checks, and no publication. Both completed checkpoints fail: the final still
calls bells piano, a printer a truck and a siren a train. No further
projection-only pass was launched.

One 1.08M-parameter temporal acoustic head trained over the same frozen encoder
repairs several event diagnostics but fails bells, sirens and its broad
validation gates (AP 0.487; top-one 37.5%). It stays disabled; no native
integration exists. Ordered features reproduce the prior pooled features
exactly. One data-only repeat added 4,587 verified local CC0 training clips
up to 30 seconds, including 35 bells and 20 sirens. The architecture, training
loop and all stricter gates remain unchanged, with the full original 240-clip
validation and all held-out test recordings protected. It also failed: AP
0.50573 and top-one 39.17%, with bell and siren checks still failing. Neither
head may be integrated. A paired raw-versus-projected encoder-tap audit also
failed its preregistered improvement gate (AP delta -0.01153; accuracy delta
-0.025); no raw-tap head was trained. Preserve the existing primary, audio
encoder and cloning weights; do not add another reasoning stack.

A conditional Ornith primary-trunk FFN adapter passes native zero/nonzero and
off/on/off compatibility, plus a bounded seen-training fit gate. Its native
audio diagnostics still fail dog/alarm/printer/silence, so it is private and
must not be enabled or published. One full-corpus trial with the same small
adapter and fixed checkpoints is recorded in
`docs/upgrades/audio-lora-fit-20261031.md`. Existing primary/encoder/projector
and TTS weights remain frozen. Qwen's own exact-checkpoint compatibility and
fit gates also pass, but native bells/crowd/printer diagnostics fail. Its matching
full-corpus trial includes speech/nonspeech replay, the same fixed checkpoints
and passed memory preflights. Adapter fit is not parity for either profile.
The full-corpus Ornith 1,000 checkpoint also fails native bells/printer/siren,
silence and other diagnostics. It is rejected; preserve the unchanged fixed
checkpoint plan and do not promote it because fit or export checks pass. See
`docs/upgrades/audio-lora-fixed-checkpoints-20261031.json`.

The full-corpus Ornith trial completed all 3,200 steps and processed all 12,458
training records. Its 1,000/2,000/3,200 checkpoints all fail mandatory native
sound diagnostics and remain rejected; the final calls bells a guitar chord,
a printer fireworks and a passing siren a boat. No further Ornith trial was
launched. Qwen also completed 3,200 steps and processed all 12,458 training
records; its 1,000/2,000/3,200 checkpoints all fail native bells, voices, printer
and siren semantics. Both bounded trials are complete and rejected. Diagnose a
specific runtime-versus-training quality cause before any further training;
do not promote by training loss. A separate current-encoder English CTC audit passes its fixed text
limits on A100 and an Orin downstream diagnostic; Orin numerical feature
parity still fails. The head remains disabled, with no shared live encoder
residency or streaming qualification. See
`docs/upgrades/current-encoder-ctc-20261031.md`. Private precision probes did
not clear the unchanged numerical gate and must not become production flags.
The same-13-input legacy comparison identifies bells, printer, gunfire, human
voices and a passing siren that the rejected adapters miss. Its silence beep
hallucination is retained; no reference-model words become verified transcripts.
See `docs/upgrades/legacy-expanded-audio-20261031.md`. The legacy model is a
benchmark reference only and must not be restored as a second release reasoner.

A paired 16-recording TRAIN-only audit of final Ornith LoRA verifies unchanged
weights and repeats several semantic errors in both unquantized HF generation
and native inference. Event-name CE is 1.248 versus 0.152 for remaining scored
tokens; the first event argmax is unchanged when gold future tokens are removed.
Export/quantization alone cannot explain the failure. See
`docs/upgrades/audio-lora-seen-training-20261101.json`. Do not stack a new
architecture, new data, sampling and loss changes in response; first test a
bounded learning hypothesis while preserving the existing frozen components.

The bounded Ornith sampling-only continuation completed global6,400; all fixed
4,000/5,000/6,400 native validation checkpoints are rejected. Final6,400 fits all
eight core TRAIN families in both unquantized HF and native inference, while
held-out identities still fail. Event CE improves from1.248 to0.200 on the same
16 TRAIN recordings. This supports a generalization/data-diversity hypothesis,
not a new architecture or quantization experiment. The original Orin service
and harness were restored and verified healthy; do not pause it for a similarly
failing A100 candidate. See `docs/upgrades/class-uniform-6400-review-20261101.json`
and `docs/upgrades/paired-training-fit-review-20261101.json`.

The separate data-only Ornith continuation completed6,400→10,662 with fixed
7,400/8,400/10,662 checkpoints:4,587 additional already-local verified CC0 TRAIN
recordings, same architecture/sampler/loss/optimizer and frozen components.
The old manifest is byte-preserved; source/audio IDs and protected uploaders
are disjoint, unknown transcripts remain unscored. Maximum-input forward/backward
preflight peaks20,966MiB, with zero optimizer steps. That completed run used a
32,768MiB lease/30,720MiB Torch cap and checked RAM/disk reserves.
One class-uniform sampling schedule does not claim every new recording is seen.

Qwen's own3,200 TRAIN audit still misses seen classes (event CE1.480 versus
remaining-token CE0.165). Its separate sampling-only trial was planned3,200→6,400,
fixed4,000/5,000/6,400, on the unchanged12,458-row data; no diversity expansion
is stacked there. Its original GPU1 run used65,536MiB/61,440MiB. The active GPU0 recovery and
subsequent target-only GPU2 ablation are described above. All completed
checkpoints retain native semantic gates and remain unqualified. See
`docs/upgrades/diversity-and-qwen-sampling-trials-20261101.json`. A checkpoint
observer may export and run diagnostics; it cannot approve semantics or publish.

The first Ornith data-only checkpoint7,400 is rejected: Siren recovered and
Church bell became specific, but group voices, clapping, printer, gunfire, clock
alarm and negative controls still fail. Speech and Bark remain correct. Preserve
`docs/upgrades/diversity-7400-review-20261101.json`; the original8,400/10,662
checkpoints remain, with no extra Orin outage or automatic extension.

Qwen's first sampling-only checkpoint4,000 is also rejected: speech, native
silence and alarm pass, but barking, bells, group voices, clapping, printer,
gunfire, siren and other negative controls fail. Preserve
`docs/upgrades/qwen-sampling-4000-review-20261101.json`. Only the original5,000/6,400
checks remain; no model promotion or experimental publication is authorized.



Exact digital silence is handled by a deterministic PCM check in the adapter.
It cannot repair noise or event-recognition errors and must never be used to
relabel failed native-model silence results as a neural quality pass. Preserve
all nonzero acoustic input, including one-LSB samples, and do not inspect voice
references as perceptual input.

Upgrade the existing GGUF/Ollama bundle architecture. Keep the trained Omni
audio bridge, native vision and Qwen3-TTS Base reference-voice cloning. Produce
two independently versioned releases: **Ornith Omni Edge v2** and **Qwen3.8 Omni
Edge v2**. They have different reasoning checkpoints and bridge projections;
never interchange their weights or pretend that they are the same model.

1. **One deployable Ollama identity per backbone.** All required neural weights
   belong to that tag and its versioned sidecar. Do not add separately downloaded
   Kokoro, Piper, standalone ASR, decision, embedding or speech-service weights.
   Existing auxiliary models remain baselines until evaluated replacements exist.
2. **Full feature parity.** Keep thinking, tools, image/video understanding,
   transcription, environmental audio, streaming speech, arbitrary reference
   cloning, saved voices, background tasks, computer use and memory.
3. **One primary reasoning trunk.** Ornith or Qwen3.8 owns semantic reasoning.
   Small trained heads may share its features; another language-model reasoning
   stack may not be introduced. Reduced profiles must not restore the redundant
   Omni Thinker or load another language runner beside the primary trunk.
4. **Measured resource improvement.** Improve real memory use, first audible
   response latency and completed-task time. Fewer GGUF filenames alone prove
   nothing about residency or performance.

One weight umbrella is a packaging boundary, not a single executable graph.
Qwen3-TTS and the reasoning trunk are incompatible architectures. Do not claim
hidden-state fusion, splice their transformer layers, or present a composite
sidecar as a stock Ollama `FROM` target.

The query-independent primary-vision control inventory plus same-trunk semantic
ID selection passes the original 12 synthetic cases, but only 61/86 broader cases
pass. Correct selection is 83/86; coordinate precision and ambiguous intent still
fail. One bounded crop-refinement trial terminates with invalid structured
output after 44 cases. The original pointing helper remains unchanged. Preserve
`docs/upgrades/primary-pointing-inventory-review-20261101.json`; do not repeat
geometry/prompt variants or call the 12-case result computer-use qualification.

## Every meaningful change must meaningfully upgrade the system

Before a patch, name one failing acceptance criterion, its evidence and one
bounded hypothesis. Implement the smallest useful change, exercise its actual
production path, and retain a compact before/after receipt. State what improved
and what remains unverified. Documentation, packaging labels, synthetic inference
results or passing implementation-mirroring tests do not establish model quality,
latency, memory savings or whole-task success.

Preserve existing work before replacement. Do not repair unrelated inherited
failures as an unannounced extension of an upgrade. After a milestone passes,
checkpoint it and report it. Continue the next milestone only under an explicit
next objective or a clearly scoped continuation of the user's request.

## Ordered swap matrix

| Priority | Mechanism | Required direction |
| --- | --- | --- |
| P0 | Multi-view GGUF sidecar | Deduplicated, indexed component views under the same tag; avoid unnecessary reconstruction. |
| P0 | Request-scoped TTS loading | Same Qwen3-TTS Base weights; qualify cheaper initialization and lazy/warm residency. |
| P0 | Repeated voice-reference processing | Cache complete reusable speaker conditioning and acoustic prompt state. |
| P0 | Full-response speech scheduling | Hand off committed speech intentions earlier; begin acoustic work safely. |
| P1 | Separate Laya weights | Train a small task-specific decision head on the exact primary trunk. |
| P1 | Separate memory embedder | Train a compact trunk-derived retrieval projection; retain Nomic as the evaluated baseline. |
| P1 | Repeated acoustic prefill | Streaming Omni encoder state and causal chunks, after compatibility/training validation. |
| P1 | Repeated visual interpretation | Temporal visual-state adapter attached to native vision. |
| P1 | Token decoding | Evaluate weight-free n-gram speculation first; train an exact-checkpoint draft head later. |
| P1 | Repeated text/tool prefill | Versioned reusable prefix KV plus compact typed tool evidence. |
| P2 | Fixed speech packets | Adaptive acoustic streaming windows, qualified for continuity and underruns. |
| P2 | Execution scheduling | One supervisor, component-aware admission and shared resource accounting. |

All neural additions must be namespaced into the matching bundle and enabled by
its versioned runtime. Proposed heads require real training; packaging untrained
parameters is not implementation. Exposing intermediate backbone states without
an extra forward pass requires actual backend work.

## Voice cloning is mandatory

Keep the exact Qwen3-TTS Base cloning checkpoint initially. Preserve both
server-local `speaker_file` and bounded request-local `speaker_audio`, imported
and saved custom profiles, default female/male choices, supported languages,
regional/reference-dependent accents, long-utterance identity, natural pauses,
expression, temperature/top-k/top-p/seed controls, interruption, and both streamed
and complete **24 kHz mono PCM16 WAV** output. Voice changes must not reload the
reasoning trunk. Request-local references do not imply permission to persist them.

The intended reusable voice record includes the original WAV, optional reference
transcript where the checkpoint/backend supports it, speaker embedding, acoustic
prompt/contextual codes, language/locale metadata and conditioning state. Cache
keys include reference-content hash, transcript, exact TTS checkpoint identity,
normalization version, language mode and conditioning-algorithm version. Bound
RAM and retain session/profile isolation. Persist only explicitly saved profiles.

Do not collapse full reference conditioning to a speaker embedding as an
unmeasured shortcut. Qwen's Base cloning path, CustomVoice and VoiceDesign are
distinct checkpoint capabilities. Full description-driven voice design or added
emotional control requires an additional compatible in-bundle checkpoint with
declared costs, or a trained/evaluated Base adaptation. A style embedding alone
does not establish either capability.

Latency work should evaluate one-frame initial emission followed by larger
windows, overlap preparation with the committed first phrase, retain valid codec
state across one uninterrupted utterance, and avoid re-encoding unchanged voice
references between chunks. Preserve request-state reset between utterances and
state at the last real codec frame. Never speak reasoning, unresolved tool calls,
or text that can still become a tool call. Cancel unplayed audio promptly without
disturbing language state. Browser smoothing cannot replace raw-PCM verification.

## Packaging and loading

Evolve the sidecar with a backward-compatible version, component inventory,
hashes, ABI metadata and validated namespaced views. Standard Ollama layers retain
the exact base and matching combined vision/audio projector. The sidecar contains
TTS Base/codec and any future trained adapters; do not duplicate the already
combined frozen audio tower/projection without a measured reason.

Prioritize identical-content deduplication, supported filtered reads and avoiding
extra full component files. Correct filtered loading must restore architecture,
metadata, names, shapes, quantization, alignment and tensor bytes. Stock loaders
cannot consume arbitrary heterogeneous inventories. Do not label materialized
views or a reuse cache as direct filtered mmap support.

Storage savings and GPU-memory savings are separate measurements. Resident memory
improves only by eliminating resident duplication, changing qualified precision,
or admitting/evicting components safely. Do not delete artifacts on an assumption
that they are duplicates. Never manually delete Ollama blobs.

## Profile-specific residency

Both profiles retain the reasoning trunk, its trained audio bridge and native
vision. Ornith is the primary 32 GB Orin target: evaluate warm cloned TTS only
after measuring co-resident peak memory. Qwen3.8 begins conservatively with
request-scoped TTS and cheaper loading/chunk scheduling. Small heads and cached
voice records require bounded admission; larger optional modules remain on
demand. Never simultaneously load a full alternate reasoner.

Preserve qualified Q8 KV/adaptive context tiers and live memory selection rather
than assuming nominal artifact sizes are peak runtime use. Preserve platform
restrictions on unsafe llama.cpp restarts. Approximate 8.15 GiB Ornith and
18.33 GiB Qwen artifact figures are planning inputs, not current-host capacity
proofs. Do not enable warm residency or new precision without measured comparison.

## Attached-head work requires training and acceptance

| Function | Direction and boundary |
| --- | --- |
| Decisions | Exact-trunk feature classifier; bounded judgments, not open-ended reasoning replacement. |
| Partial ASR | Streaming token/CTC-style head over existing audio encoder; preserve full environmental/ambiguous audio path. |
| Retrieval | Compact projection from shared features; beat the existing Nomic baseline before removal. |
| Visual continuity | Temporal entity state; preserve detailed still-image and sampled-video understanding. |
| Salience/VAD | Deterministic DSP where sufficient; bundle any learned detector and retain non-speech evidence. |
| Speculation | Weight-free baseline first; a draft head must match the modified trunk exactly. |
| Endogenous state | Compact recurrent sensory/task updater; avoid continuously running the full 9B/27B trunk. |
| Speech intention | Trained bridge from committed output to existing TTS, preserving arbitrary voice-reference conditioning. |
| Speaker selection | Cached identity/conditioning records with session/profile isolation. |

## Feature-parity release gate

The exact reasoning checkpoint, native boolean `think` and separate thinking
channel, tool contracts/chaining/authorization, browser/desktop/shell/document
tools, durable background task checkpoints/cancellation/recovery, transcription,
environmental sound, image/video/video-audio temporal grounding, reference clones,
saved profiles/languages, streaming/full WAV, interruption/echo handling,
concurrent-session isolation, offline inference, virtual context/durable memory
and qualified quantization/context profiles must all remain supported.

Run the same fixtures on each upgraded bridge and its baseline. Compare reduced
bridges against corresponding legacy full-Omni releases for transcription,
speaker activity, environmental sounds, music, overlapping audio, audio/video
synchronization and temporal questions. Removing a backbone does not itself
prove semantic parity.

## Delivery order and measurements

1. **A — Packaging/loading:** indexed views, proven deduplication and less
   reconstruction; preserve Ollama-facing identity. No neural retraining needed.
2. **B — Cloning/streaming:** complete conditioning reuse, first-packet work and
   measured safe Ornith residency; retain Base and voice/language fidelity.
3. **C — Shared heads:** train/package/evaluate decisions, retrieval and streaming
   audio before retiring existing auxiliary baselines or enabling fast paths.
4. **D — Continuous cognition:** small temporal state over visual/acoustic/task
   changes; primary trunk retains semantic decisions.
5. **E — Orin qualification:** concurrent listening, viewing, reasoning, cloning,
   speaking and tools under measured safe memory limits.

Initial proposed targets are **20% lower median first-audio latency** and **10%
lower peak memory** where the baseline permits, without sacrificing capabilities
or cloning quality. They are targets, not predicted or attained results. Measure
speaker similarity, ASR-based intelligibility, language accuracy, chunk continuity,
conditioning reuse, p50/p95/p99 latency, completed-task accuracy, headroom, loading
time and the full tool/media regression matrix. Record device, checkpoint hashes,
runtime revision, context, precision and concurrency for every comparison.

## Resources, provenance and host discipline

`/srv/fine_tuning_suite` provides existing bridge training, GGUF packaging,
Ollama sidecar tooling, TTS patches and evaluation resources. Inspect its README
and applicable instructions before reuse. Keep this runtime standalone; record
source paths/revisions when adapting code rather than requiring that checkout.

Follow the host GPU broker and `AGENTS.md` before any CUDA lifecycle action.
Existing services are not authorization to restart them for experiments. New
work uses its own runtime state/ports and exactly leased GPU UUIDs. CPU artifact
inspection/materialization is not a live CUDA qualification.

Fork provenance and the assessment of inherited edits are recorded in
`docs/upgrade-baseline.md`. Preserve historical claims as historical; no source
test, fixture or documentation claim closes live model acceptance.
