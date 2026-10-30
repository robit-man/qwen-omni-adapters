# Edge v2 live release qualification — 2026-10-30

**Release result: FAIL. No public Hugging Face or Ollama publication.**
The complete corresponding weight packages, manifests, component licenses and
model cards are staged locally; their cards explicitly say qualification is
incomplete. Publication remains conditional on the user's parity requirement.
An experimental-preview exception was requested and has not been granted.

The user assigned Ornith 1.5 to the 32 GB Tegra234 Orin and Qwen3.8 to the
current A100 host. The Qwen3.8 32 GB Orin attempt is diagnostic evidence, not
its release qualification target.

| Check | Ornith / Orin | Qwen3.8 / A100 |
| --- | --- | --- |
| Admitted context and KV | 16,384 / Q8 | 16,384 / F16 |
| Text, separate thinking, structured tools | PASS | PASS |
| Unresolved tool call produces no speech | PASS | PASS |
| Fresh red → blue → red image evidence | PASS | PASS |
| Clean speech transcription | Exact fixture transcript | Exact fixture transcript |
| Full-reference clone, complete and streamed WAV | PASS, matching PCM | PASS, matching PCM |
| Streamed format | 24 kHz mono PCM16 | 24 kHz mono PCM16 |
| Concurrent session markers | Isolated | Isolated |
| Actual primary GPU residency | Proven through accelerator module | Proven on leased exact GPU UUID |
| Environmental sound semantics | FAIL | FAIL |
| Speech within video after bounded routing repair | Exact fixture transcript | Exact fixture transcript / WER 0.0 |

The source/contract/browser/installer gate passes **6,270 tests** in 287.93 s.
These tests and the 14 operational checks do not establish semantic parity.
The original operational alarm/music checks accepted nonempty descriptions;
that criterion was insufficient. The separate semantic gate retains actual
outputs and rejects wrong descriptions even when HTTP and tag formatting pass.

## Decisive failures

On real ESC-50 clock-alarm, dog and church-bell fixtures, the trained bridges
omit or misidentify the events. Qwen3.8 predominantly calls them broadband hiss;
Ornith omits the barking and bells. A clean speech/video fixture contains the
sentence “The blue bicycle is parked beside the garden gate.” Qwen3.8 initially
reported no audio. The repair below recovers its exact transcript; both devices
now pass that fixture. Environmental identification still fails.

The original full-Omni audio model/projector was independently materialized
from the installed original sidecar and run on a separate scoped A100 lease.
It identifies the synthetic beeping, static, real dog barking and church bells,
and extracts the exact video speech while preserving red → blue → green order.
This establishes a real full-Omni capability gap, rather than a missing audio
track or invalid fixture. It does not establish a broad dataset accuracy score.

The compact v2 standard backbone and combined projector bytes remain identical
to their existing compact-bridge baselines. An Orin comparison against the old
running adapter also reproduces weak sound descriptions. The disabled CTC head
is not used by these requests. Packaging improvements and preserved TTS weights
do not repair a trained projection's semantic limitations.

## Fixed runtime defect

Qwen3.8 initially rejected describe requests because the perception payload
forwarded a second caller/portal system message after its pinned extraction
policy. The adapter now leaves caller system instructions in the language
stage and uses exactly one pinned perception system message. The actual
patched authenticated HTTP route passes fresh-image checks on the unchanged
loaded Qwen worker; regression tests cover chat, describe and transcribe.

The subsequent mixed-video defect was repaired structurally: trained bridges
submit separate audio and visual perception requests through the same resident
primary. The audio pass uses the bridge's trained prefill. Evidence is combined
by modality, discarding speech/acoustic claims from the visual pass and visual
claims from the audio pass. No keyword routing or additional neural model is
introduced. The actual A100 video-speech gate improves WER from 1.0 to 0.0 while
retaining red → blue → green. One run takes 6.31 s versus the failed 6.63 s
control; the Orin regression takes 59.94 s versus 51.05 s before splitting.
These are single observations, not a claimed latency improvement. The source
tests cover both complete and streaming paths and channel authority.

The first Ornith audio-fixture attempt used an unsupported sample rate. The
fixtures were normalized to the documented 16 kHz PCM16 input and only those
failed contract checks were rerun. Original failed receipts remain in ignored
runtime evidence; the merged operational receipt identifies the correction.

## Scope, restoration and next acceptance

No physical speaker/playback, independent perceptual clone-quality, p95/p99 or
whole-system memory target is claimed. Previously qualified matched-PCM and
reference-conditioning subgates remain separate evidence. Live streaming CTC,
trained decision/retrieval replacements and recurrent cognition remain pending.

Both complete v2 candidate tags were hash-verified on the Orin. The old service
and desktop harness were paused only for the authorized qualification window,
then restored. Their final state is ready/active, with the old Ornith model
and integrated-GPU residency proven. The rollback timer and temporary HTTP
workers were stopped. All task-owned A100 leases were released; peer leases
were preserved.

Before a qualified release can be promoted, repair and evaluate the matching
trained projections' non-speech understanding. Rerun the same semantic fixtures,
retain held-out generalization and ASR
regression checks, and keep arbitrary full-reference cloning intact. Do not
substitute successful requests, keyword routing, another reasoning model, or
new packaging labels for those gates.

Machine-readable result: [release-qualification-20261030.json](release-qualification-20261030.json).
Fixture provenance: ESC-50, pinned source URLs/digests retained in the local
release fixture inventory. ESC-50 recordings are evaluation-only; no recording
was used to train these weights or redistributed with staged model artifacts.
