# B4b3 — reference transcripts through runtime requests and saved profiles

Optional `reference_text` now reaches the Base full-reference prompt through the
adapter, TTS wrapper, Python/JavaScript clients, portal voice dialog and saved
desktop profiles. No transcript keeps the original speaker-embedding behavior.
The shipped default leaves the transcript blank so legacy bundles retain their
existing behavior. Missing encoder weights fail explicitly when full conditioning
is requested; no downgrade or runtime weight download is used.

The installed indexed sidecar supplies model, codec and frozen reference encoder
under one identity. The encoder runs lazily and exits before TTS loading. The
wrapper snapshots the reference so speaker/acoustic conditioning cannot read
different versions of an in-place edit. It removes snapshots and native JSON
after readiness. One RAM entry, limited to a 128 KiB serialized conditioning
payload, keys audio content, transcript, language, exact Base/codec/encoder
identities, normalization version and conditioning algorithm. It can survive
speech-weight eviction. Native embedding/reference-decoder reuse remains bounded
separately by B4b2. Inline audio gets its own worker and cache for the utterance;
request completion or interruption closes that worker and deletes its files.

Saved presets bind transcript and audio together. Selecting a preset without a
transcript clears the previous transcript; uploaded audio cannot inherit one.
The indicator's explicit **Reference transcript** action saves a voice's text.
The portal textarea is request-local. Schemas, adapter metadata, clients,
supervisor component paths and protocol documentation are synchronized.

## Actual synthesis-route qualification

`runtime/qualify_tts_reference_http.py` exercises the real Flask routes and native
CUDA children with the installed shared sidecar digest
`7838f27ad60ed5181c60e9b6e7263df944933f07099602aa70a5e96c8640f388`.
It starts no public service and restarts no production process. A scoped A100
reservation covers the initial encoder load; later stages use prepare/ready.
On Orin, the runtime proves integrated-GPU residency through device handles and
retains its memory floor. Library mappings and producer PIDs are recorded.

The gates cover saved-reference A/B/A equality, speech-weight shedding/reload,
encoded-conditioning reuse, inline/saved PCM equality, complete-WAV/stream
equality, inline batch concatenation, cancellation, saved voice reuse after an
inline request, raw continuity, and zero remaining temporary reference files.
The same short reference/transcript and seed 42 are used throughout. This is a
bounded functional qualification, not arbitrary-voice perceptual scoring.
Both A100 and Orin passed these route gates on the final normalization path.

An initial Orin run failed continuity. Investigation found that its older
FFmpeg/soxr integer output differed from the validated normalized input by one
PCM16 step at 11,933 of 35,834 samples. Float output with explicit nearest-even
rounding reduced this to 47 samples; the generated Orin PCM then matched the
passing B4b2 native control exactly. The normalization key is now
`24khz-soxr-precision20-float-pcm16-v2`. Cross-platform normalized audio is not
claimed universally bit-identical. The gate was retained without relaxation.

The Orin synthesis-route run passed all functional gates, while its normal
6144 MiB idle floor declined warm weight retention under current memory pressure.
The saved reference encoded once and its RAM conditioning was reused across
worker reloads; no retention floor was lowered. Warm/native figures from B4b2
remain separate from these complete route timings. Request-local references
still pay their own encoding/loading cost, as required for isolation.

Remaining release gates include all-language/full-clone perceptual quality,
speaker similarity, accents, long target utterances, complete backbone/media/tool
parity, concurrent-session resource qualification and trained shared heads. These
receipts do not authorize publication ahead of those gates.

## Reproduction

Supply a private transcript file rather than putting reference text in argv:

```bash
python runtime/qualify_tts_reference_http.py \
  --binary BUILD/bin/llama-tts \
  --model INSTALLED_SIDECAR --projector INSTALLED_SIDECAR \
  --encoder INSTALLED_SIDECAR --reference VOICE.wav \
  --transcript-file PRIVATE_TRANSCRIPT.txt \
  --work-dir NEW_DIRECTORY --ready-file NEW_READY_PATH --out REPORT.json
```

On discrete CUDA, use `docker gpu run` with a fresh readiness path, an explicit
owner, one exact UUID and at least 12000 MiB reserved. The qualifier registers
only its own process tree after actual encoder residency, then signals readiness.
On Tegra, use the integrated device without a discrete lease. The new checkout
and installed components must be independent of the production service's
disposable component directory.
