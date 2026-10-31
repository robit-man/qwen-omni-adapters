# Rich-target projection qualification, 2026-10-31 UTC

Publication remains gated. The repaired Ornith projection passes the original
small semantic fixture gate on Orin, and both real bark and bell/chime recognition
are recovered. Broader environmental validation still exposes errors relative to
full Omni. Qwen's richer projection still misses the bell fixture on A100.
The user chose repair before publication; no experimental-preview exception applies.

## Exact selected Ornith candidate

`robit/ornith-1.5-omni:edge-v2-rich1-best850-20261031` is a **local candidate**.
The richer-target run completed 1,600 steps; minimum validation loss selected
checkpoint 850, not the final checkpoint. Projector SHA-256:
`3ab6b8ce7feda7178092eee36b6c2700dbd256d23ec6f09fa350c9fca055b3f2`.
Only `mm.a.mlp.2.weight` and `.bias` changed; 858 frozen audio/vision tensors
remain byte-identical. Reasoning and Qwen3-TTS Base/codec/reference weights
remain unchanged. The indexed sidecar binding was resolved and verified after
installation on both hosts. No public model upload occurred.

Stock Ollama 0.40.11 loaded this exact candidate under a scoped GPU0 lease and
passed native `think:false`, separated `think:true`, and typed tool-call checks.
The check used an isolated loopback stock server because the shared frontend's
admission queue was waiting on an unrelated hard-GPU request. The shared daemon
and peer requests were preserved. This is a functional result, not a guarantee
that an inherited large default context will fit every deployment device.

## Actual Orin execution

Source `731974f`, all 16 native patches, Tegra234, 16,384-token ceiling and Q8 KV:

- Deployment doctor passed before startup; primary CUDA residency was verified
  through integrated-GPU device and allocator handles.
- All **14 operational checks** passed: native thinking, unresolved-tool speech
  deferral, red→blue→red, speech transcription, non-speech routing, silence
  short-circuit, sampled video, complete and streamed request-local cloning,
  post-TTS transcription, and simultaneous session isolation.
- All **5 original semantic fixtures** passed: synthetic tone, recorded alarm,
  dog bark, bell/chime, and spoken sentence within RGB video. Alarm is still a
  coarse tonal description; matching a permissive tone pattern is not proof of
  exact alarm recognition. Bark and bell/chime outputs identify their event type.
- Complete and streamed clones produced identical PCM SHA-256
  `445949765b845b9bb0e0b724001830a2db733cb31c72c3ea5544cde64328376e`.
  Output is 24 kHz mono PCM16; the streamed request emitted 14 chunks and its
  first PCM arrived in 8.656 seconds. One sample does not establish latency tails
  or the proposed 20% improvement target.
- The independent raw two-frame stream gate passed: median boundary ratio
  0.08276 against 0.35, and one-millisecond shared-prefix ratio 0.20484 against 0.85.
- Three simultaneous independent requests (listening, image analysis, request-local
  cloned speech) passed with correct modality provenance. Minimum sampled
  `MemAvailable` was 10,418 MiB, above the 2,048 MiB hard floor. Primary and TTS
  workers held their required GPU handles. This does not test a physical
  microphone/speaker loop or prove whole-task accuracy.

The old core, harness, and previously active update timers were paused under a
45-minute restoration timer. They were restored after qualification; the old
portal returned HTTP 200 and its supervisor reported ready. The temporary
candidate service stopped and the failsafe timer was canceled after restoration.

## Speech comparison against the actual original runtime

Same A100, 4K F16 KV, 160 validation inputs (154 speech clips, 3,336 reference
words, six non-speech controls), same normalization. The original baseline uses
source `7554ae0`, its original native binaries, and original projector
`7e2d6fe7e80c4538e961e592d954d1c47d64e087139db03043741a0238af2c1c`.
The earlier unreleased label1450 checkpoint is **not** the original baseline.

| Speech source | Original v1 WER | Repaired Ornith 850 | Full Omni comparison |
| --- | ---: | ---: | ---: |
| All 154 clips | 18.645% | 6.894% | 20.504% |
| Synthetic speech with procedural ambience | 22.829% | 1.118% | 8.026% |
| Natural LibriSpeech with local augmentation | 7.239% | 5.556% | 1.515% |
| TTS word-list recovery | 18.985% | 14.730% | 45.254% |

All three runs hallucinated zero transcripts on the six negative controls.
The repair improves each source over original v1, while natural speech still
trails full Omni. Aggregate WER alone hides that gap. These are validation
results, not independent test acceptance or a multilingual claim.

## Broader parity and next bounded experiment

A 240-clip uploader-disjoint FSD50K validation comparison retains raw outputs;
source titles and labels were never sent to either model. Weak labels are not
exhaustive captions, so no fabricated aggregate semantic score is reported.
Clear remaining counterexamples include children interpreted as a dog, a clap
as a gunshot, and a printer as a harmonica, where full Omni identifies the main
event. The full feature-parity gate therefore remains open.

Qwen's validation-selected checkpoint 750 identifies the dog but describes the
church bells as an electronic tone. Its 1,600-step richer-target run continues;
neither lower teacher-forced loss nor a passing HTTP contract clears this failure.

One bounded Ornith experiment trains **both existing qwen3a projection layers**
with the same rich manifest. It adds no graph or reasoner and keeps audio
convolution/attention, vision, the primary trunk and TTS frozen. Before any
training, speech/bark/bell projected tensors matched the prior graph exactly
(relative RMSE zero). The two-layer trainable count is 6,886,656; export changes
only the existing `mm.a.mlp.1` and `.2` weight/bias tensors. The trial is capped
at 400 steps and remains unqualified. It is not deployed or published.

Machine-readable measurements and raw counterexamples are in
[the progress receipt](rich-projection-qualification-20261031.json), with separate
[Orin operational](rich850-ornith-orin-operational-20261031.json),
[semantic](rich850-ornith-orin-semantic-20261031.json),
[PCM](rich850-ornith-orin-pcm-20261031.json),
[concurrency](rich850-ornith-orin-concurrency-20261031.json), and
[restoration](rich850-ornith-orin-restoration-20261031.json) receipts.
