# B3a — measured packet selection for small full-reference speech

The adapter previously forced eight-frame packets, regardless of the selected
speech context. Its browser also requires at least 250 ms of buffered audio.
Reducing the first packet to one frame therefore does not establish lower
audible latency. The native pilot measured 1, 2, 4, 6 and 8-frame decode windows
with the same Base checkpoint, full-reference prompt, context, sampling and
reference. It retained all arrival timestamps and raw PCM continuity scores.

## Selection and actual route gates

`OMNI_TTS_STREAM_FRAMES=auto` is now the coordinated runtime default. The adapter
omits its packet override; the worker selects four frames for full-reference
requests whose qualified adaptive context is 4K. Other cases retain eight.
Explicit settings retain their value. Mixed-size batches select one cadence
against their largest context tier, preserving a single worker through the
utterance. Speaker-embedding and larger-context profiles are not silently
retuned on evidence obtained from the full-reference/4K pilot.

Actual A100 and Orin synthesis routes selected four frames and passed saved A/B/A,
inline/saved equality, complete-WAV/stream equality, explicit weight shedding,
conditioning reuse, batch concatenation and cancellation/file-cleanup gates.
Reference audio, all sixteen code groups and the transcript remain intact.
No neural checkpoint, sampling control or tool/commit boundary changed.
Target speech still starts only after the language result is committed and
contains no unresolved tool call. Preparation overlap and within-utterance
window ramps remain subsequent B3 work.

## Native arrivals replayed through the real browser scheduler

The replay executes the unmodified production `call_playback.js` pacer and
`portal.js` scheduling functions using recorded native packet arrival times.
Audio buffer durations come from the real packets. Zero arrays stand in only
for amplitudes in the timing replay; actual generated PCM is tested separately.
This measures scheduled WebAudio start under zero network delay, not physical
speaker latency or end-to-end reasoning time. Each median uses the four warm
requests after the initial request in a bounded A/B/A/B/A case.

| Device/prior production rate | Eight-frame scheduled start | Four-frame scheduled start | Reduction |
| --- | ---: | ---: | ---: |
| A100, prior 1.0 | 246 ms | 182 ms | 26.0% |
| A100, prior 0.8 | 404 ms | 269 ms | 33.4% |
| Orin, prior 1.0 | 651 ms | 429 ms | 34.1% |
| Orin, prior 0.8 | 1161 ms | 717 ms | 38.3% |

Four-frame replay produced zero underruns in these cases, including the tested
1.6 prior. One- and two-frame cases produced Orin underruns with the faster
priors. They were not selected. All 50 primary-reference raw PCM cases passed
continuity. Replay completion times are retained as observations but are not
credited as an improvement: existing browser crossfades shorten the scheduled
timeline by an amount that depends on packet count.

## Voice evidence and retained negative results

Both shipped references were evaluated. Eight paired generated recordings from
the two devices and four/eight-frame cadences transcribed exactly through the
already resident Orin primary; no standalone ASR model was added. The existing
same-bundle projected speaker encoder compared both cadences on one matched
A100 backend. Pairwise conditioning cosine was at least 0.999965; the largest
reference-cosine decrease was about 0.000080, within the predeclared 0.001
comparison tolerance. This internal metric is not an independent calibrated
speaker verifier and does not close perceptual cloning qualification.

The alternate-reference A100 B utterance triggered a raw continuity flag at
both four and eight frames. The same five Orin four-frame utterances passed;
its first eight-frame alternate clip was too short for the metric's required
four complete windows. That pilot stopped with a scoring error. These negative
and unscored observations remain in the receipts. No threshold was relaxed and
no universal clean-PCM or arbitrary-voice quality claim follows. They remain
work for the full release-quality evaluation.

The Orin pilot stayed above its 2048 MiB child-cancellation floor (minimum
4508 MiB in the primary-reference sweep). This is sampled host headroom, not a
whole-profile peak-memory improvement claim. The primary service and desktop
harness stayed running; no production restart occurred.

## Reproduction

`runtime/benchmark_tts_cadence.py` records native packets using a precomputed
reference prompt from the corresponding bundled encoder. Its raw-quality gate
fails explicitly on a flagged or too-short utterance while retaining the report.
Use fresh work/readiness paths, an exact scoped CUDA reservation, or the explicit
Tegra memory floor. `runtime/replay_tts_cadence.cjs INPUT.json OUTPUT.json` runs
the existing browser pacer/scheduler with those arrivals and records source hashes.

`runtime/tts_speaker_probe.cpp` and `runtime/qualify_tts_speaker.py` reproduce the
internal speaker-conditioning comparison. Build the helper against the pinned
native libraries, for example from the repository root:

```bash
g++ -std=c++17 -O2 runtime/tts_speaker_probe.cpp \
  -I vendor/llama.cpp/include -I vendor/llama.cpp/ggml/include \
  -I vendor/llama.cpp/common -I vendor/llama.cpp/tools/mtmd \
  -L vendor/llama.cpp/build/bin -Wl,-rpath,"$PWD/vendor/llama.cpp/build/bin" \
  -lllama-common -lllama -lmtmd -lggml -lggml-base \
  -o runtime-data/tts-speaker-probe
```

Run the scoped Python qualifier with that binary, the installed model/projector,
a bounded input JSON, and fresh work/readiness paths. Reference rows declare
`id` and `path`. Each generated row also declares `pair`, `reference_id` and
`frames` (4 or 8). The qualifier proves GPU residency before sending `GO`, and
uses only the existing projected speaker encoder under the same weight umbrella.
Perceptual scoring, all languages, long utterances, complete feature parity and
registry publication remain release gates.
