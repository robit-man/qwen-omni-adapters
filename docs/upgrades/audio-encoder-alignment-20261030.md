# Frozen audio encoder alignment — 2026-10-30

The native Qwen3A path did not reproduce the frozen encoder used to train the
compact bridges. Four-second inputs generated 65 native audio tokens instead
of 52 reference tokens. Padded CNN positions participated in attention. The
native mel floor was also `5.960464477539063e-08`, whereas the reference Whisper
extractor clamps at `1e-10`.

The new native patch drops the final centered STFT frame, retains valid frame
lengths, introduces CNN batching padding inside the graph, removes padded CNN
positions **before attention**, and matches the reference mel floor and valid
normalization range. Weights are unchanged. Both platform builders apply the
patch to the pinned backend.

On the same eight speech, silence, procedural sound and real ESC-50 fixtures,
all native token counts now equal the reference counts. Final projected feature
cosine similarity is 0.999830–0.999946, versus 0.876415–0.992168 before the fix.
Relative RMSE falls to 1.07–1.92%, versus 12.49–49.00%. The reference uses BF16
PyTorch execution; native execution is not claimed bit-identical. Every native
encoder was proven resident on the exact leased GPU before encoding.

All 13 native patches apply in order to a fresh checkout of
`458681e1d5d4a29a1463c4732e03226cf384b997`. Its three changed encoder files match
the compiled sources. The rebuilt runtime also passes six full-reference
A → B → A TTS outputs: cache-on/off PCM is identical and every raw PCM stream
passes `runtime/verify_pcm_stream.py:analyze_pcm`.

This is an encoder-alignment milestone, **not environmental semantic parity**.
Eight fixtures do not qualify short-window tails, arbitrary long recordings,
all languages, or both installed model bundles. A separate prompt mismatch
also remains: the training collator omits audio boundary tokens that the
native Qwen3A path inserts. The attempted 400-step Ornith projection must not
be published merely because its validation loss improved.

The full source/contract/browser/installer gate passes **6,271 tests** in
286.25 seconds. This does not change the failed release-quality status.

Receipts:

- [Paired feature measurements](audio-encoder-alignment-20261030.json)
- [Pinned patch composition](audio-encoder-patch-composition-20261030.json)
- [Rebuilt full-reference PCM regression](audio-encoder-tts-regression-20261030.json)
- [Training attempt and remaining quality gate](environmental-repair-training-20261030.md)
