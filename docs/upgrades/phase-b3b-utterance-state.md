# B3b — decoder continuity inside a committed utterance

Acceptance criterion: full-reference speech blocks belonging to one uninterrupted
utterance must retain causal acoustic decoder history. Previously every block
created a new helper whose `reset()` cleared `c2w_state`. Immutable reference
conditioning was reusable, but target acoustic history was lost between blocks.

The bounded change transfers only waveform decoder state between fresh helpers
under explicit begin/end controls. Language KV, samplers and acoustic sampling
RNG still reset per block. The exact Base checkpoint, all reference acoustic
groups, transcript, speaker conditioning and existing text-commit boundary stay
intact. No second model or trained head was introduced.

## Scope and lifecycle

`OMNI_TTS_CONTINUOUS_BATCH=1` opts into full-reference multi-block batches.
Default is off. Single-block and embedding-only synthesis retain independent
requests. A batch uses one voice/profile, context tier and packet cadence; a
changed reference/checkpoint/profile closes the active worker before more text
can be synthesized. At most 64 blocks and 16 MiB of opaque decoder state are
allowed. The measured decoder state occupies 5,218,948 bytes in host memory;
this is an added allocation, not a memory reduction.

Native `--tts-utterance-controls` requires persistent full-reference synthesis.
Private JSON lines use schema `robit.qwen3tts.utterance-control.v1`, a 32-digit
hexadecimal identifier and `kind` begin/end/abort. A `C` frame acknowledges the
exact control. Text remains base64 and cannot impersonate a control. Invalid
transitions discard state and emit an error. Cancellation kills only this
request's worker and discards unread PCM. The wrapper consumes control frames;
public PCM streaming and event shapes are unchanged.

Target state lives outside the immutable reference cache and is freed at end,
abort or error. Each helper receives state only after full-reference
initialization verifies the exact decoder layout. The state retained by the
existing decoder is from the last real codec frame, excluding rear padding.

## Native evidence

Fresh pinned source accepted all twelve patches sequentially, and both native
builds compiled. A100 and Orin each generated 16 outputs: baseline A/B/A,
candidate independent A/B/A, two repeated two-block utterances, and probes after
end, abort and an invalid identifier. All 32 raw outputs passed continuity.
Ordinary candidate PCM was byte-identical to the corresponding baseline;
standalone A stayed identical after every termination case. Repeated continuous
utterances were deterministic.

| Device | First block decoder position | Second block position | Second-block real frames |
| --- | ---: | ---: | ---: |
| A100 | 66 | 102 | 36 |
| Orin | 73 | 112 | 39 |

The positions include the 19-frame reference. The second block advances from
the first block's position rather than resetting to reference position 19.
Measured phrase joins were already clean in both policies (zero sample jump),
so this gate establishes retained acoustic context, not improved sound quality
or lower latency. The Orin minimum observed available memory was 4,374.5 MiB,
above the unchanged 2,048 MiB qualification floor. No deployed service restarted.

## Actual synthesis routes and voice checks

Both devices' actual HTTP synthesis routes selected the qualified four-frame
automatic cadence. Saved A/B/A, inline/saved equality, complete-WAV/stream
equality, conditioning reuse after weight shedding and cancellation cleanup
passed. Continuous inline batches repeated exactly, retained native decoder
position and left no temporary reference files. The first block matched its
standalone recording; subsequent PCM deliberately differs because it receives
preceding acoustic context. The receipt's `inline_batch_exact:false` denotes
that difference from independent-block concatenation, not a failed repeat gate.

All four reset/continuation B recordings transcribed exactly through the existing
resident Ornith audio bridge with `cache_prompt:false`; no ASR model was added.
The bundled projected speaker encoder compared all recordings on one matched
A100 backend. Continuation versus reset cosine was 0.999140 (A100 recording)
and 0.998496 (Orin recording). Reference cosine decreased by 0.000253 and
0.000247, within the predeclared 0.001 tolerance. This internal metric is not an
independent calibrated speaker verifier or perceptual cloning qualification.

The native pilot used the private precursor of
`runtime/qualify_tts_utterance.py`; its hash is retained in the receipts. The
maintained reproducer accepts baseline/candidate binaries, exact matching
libraries, bundled component paths and a frozen reference prompt. Use an owned
12000 MiB scoped foreground GPU reservation with a fresh readiness/work path,
or a positive Tegra memory floor. `runtime/qualify_tts_reference_http.py
--automatic-packets --continuous-batch` exercises the actual route gate.

Broad voice/language, long-duration and perceptual tests remain release gates.
Committed-text preparation overlap and dynamic within-utterance packet ramps
remain separate B3 work. No whole-model validation or registry publication is
claimed here.
