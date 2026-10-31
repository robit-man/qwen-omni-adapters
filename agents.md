# Qwen Omni Upgrade — durable user ground truth

Accepted 2026-10-29. This is the user's engineering mandate for this repository.
Read this file together with `AGENTS.md` before making changes. Preserve these
requirements across context renewal. Targets and proposed designs below are not
claims of implemented capability, compatible weights, or measured improvement.

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

## Purpose and non-negotiable constraints

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
