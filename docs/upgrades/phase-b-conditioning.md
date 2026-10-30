# Phase B1 — immutable conditioning reuse

## Verified change

The pinned native Qwen3-TTS worker can now share an immutable conditioning cache
between freshly created audio-generation helpers. It caches one model's decoded
token table/special tokens and one bounded normalized speaker reference plus its
conditioning vector. Generated codes, prompt positions, KV state, PCM, samplers
and codec state remain request-local and reset exactly as before.

The cache belongs to one exact model/projector context. Reference reuse compares
normalized bytes, dimensions, modality and language; its normalized reference
copy is bounded to 16 MiB. Larger references retain the existing synthesis path
without caching that reference. Request-local browser references still use the
existing ephemeral worker and are not persisted. The serial worker owns cache
lifetime; this does not make a llama context safe for concurrent decoding.

The Python worker profile additionally binds reference content, checkpoint
content, normalization version and the existing language/sampling/seed/window
settings. Changing a reference in place or replacing either checkpoint forces a
new TTS worker. Weight hashes are retained only for the two configured public
checkpoint files and invalidated by stable file identity changes. Access-time
changes from ordinary reads do not invalidate otherwise unchanged checkpoints.

Both native builders apply the incremental patch and verify its CLI feature.
`OMNI_TTS_CONDITIONING_CACHE=0` selects the explicit comparison baseline in the
patched binary; the normal mode enables reuse. There are no new public voice
request fields or changed audio/event schemas.

## Real same-checkpoint experiment

[Raw measurement receipt](phase-b-conditioning.json). A scoped 12,000 MiB lease
on one A100 ran the same patched native executable with caching disabled and
enabled. It used the bundled Base Q4_K_M model, codec/projector, both shipped
reference voices, seed 42, two-frame PCM windows and the same A → B → A prompts.
Each mode reused one resident PID for its three utterances. Every child exited
and the task lease was released; the three pre-existing owners were retained.

| Shipped reference | Uncached warm first packet, median | Cached warm first packet, median | Reduction |
| --- | ---: | ---: | ---: |
| Female | 0.7813 s | 0.0663 s | 91.51% |
| Alternate/default | 0.8250 s | 0.0703 s | 91.47% |

Cold first packets remained around 0.87–0.89 s after graph loading. These are
loaded-worker TTS timings, excluding reasoning, tool use and graph startup; they
are not end-to-end first-audio or Orin latency results. The native final counters
were one token-table load, one speaker encode and two speaker hits per cached
three-utterance sequence.

All **12 PCM outputs** passed `runtime.verify_pcm_stream.analyze_pcm` directly on
the raw decoder output. Each cached PCM SHA-256 matched its uncached counterpart;
both A → B → A sequences returned exactly the same A bytes before and after B.
Both references therefore retained the baseline's actual acoustic output. This
is stronger than comparing browser playback or WAV headers alone.

The cached female worker's peak CPU RSS was 2,052,948 KiB versus 2,233,920 KiB
uncached; the alternate measured 2,057,660 versus 2,233,852 KiB. These are worker
CPU peaks, not whole-system or GPU peaks. Further memory reduction remains needed.

## Capability boundaries and remaining qualification

The inspected pinned GGUF backend implements reference-speaker conditioning via
its speaker encoder. Its existing public voice-profile metadata calls that mode
`speaker_embedding`. This patch preserves that exact computation; it does not
implement reference-transcript/acoustic-code conditioning, VoiceDesign or a new
style control. The official
[Qwen3-TTS implementation](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py)
distinguishes reusable full clone prompts from its x-vector-only path. Supporting
additional full-prompt behavior requires actual backend/codec-encoder work and
qualification, not relabeling this cache.

Tests cover same-path reference changes, both checkpoint replacements, stable
PID reuse, normal access-time updates, cache cleanup, framed cancellation and
existing TTS behavior. Native build fixtures now include the new required patch
and exercise applying each independent patch exactly once without changing the
backend pin. The first complete source run exposed those fixtures' missing new
prerequisite; the fixture was updated and its selected-patch coverage expanded.
Final source-gate results are recorded in the milestone status receipt.

The operator-specified Orin is accessible and was inspected read-only. It is a
32 GB Tegra234 on L4T R36.3.0, serving the standard Ornith bridge with a 65,536
context and Q8 KV; initial available memory was about 3.1 GB. No extra model was
loaded there and its existing runtime was not restarted. Warm co-residency,
multilingual/clone-quality comparison and whole-service peaks remain separate
qualification gates. The next bounded loading hypothesis targets the current
1,269,825,536-byte float expansion of a 178,569,216-byte quantized token table.
