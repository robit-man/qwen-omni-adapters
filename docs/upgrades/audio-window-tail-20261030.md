# Audio CNN tail windows — 2026-10-30

The first encoder-alignment patch passes eight ordinary fixtures, but an
8.5-second recording exposes a separate mismatch. The reference audio tower
pads CNN subchunks according to the original recording's maximum subchunk
width. The native implementation previously treated a short final inference
window as a separate, shorter CNN batch. Removing padded output tokens after
the CNN does not repair the different convolution boundary conditions.

The follow-up patch carries the original maximum CNN chunk width through
preprocessing into the graph. It keeps valid frame counts and discards padded
positions before attention. The new execution-only fields are excluded from
placeholder serialization; cached token counts depend on valid frame lengths,
and placeholders are never encoded.

Across 13 deterministic tone/noise recordings from 0.25 to 24.1 seconds, every
token count remains exact. At 8.5 seconds, minimum token cosine similarity
against the BF16 training reference improves from 0.780992 to 0.999127; relative
RMSE falls from 8.52% to 1.80%. The 16.1- and 24.1-second cases also improve.
The paired measurements retain controls whose numerical differences remain.
An additional F32 reference comparison puts every token above 0.995 cosine
similarity; execution is not claimed bit-identical across precision/backends.

All 14 native patches apply in order to the pinned source, and the changed
files match the compiled runtime. Six newly generated full-reference TTS
outputs again pass raw PCM continuity and cache-on/off equality. No neural
weights change in this patch. These checks do not qualify environmental
recognition or authorize publication.

The full source/contract/browser/installer gate passes **6,272 tests** in
292.13 seconds.

See [the duration and composition receipt](audio-window-tail-20261030.json).
The next bounded defect is the independently observed mismatch between the
trained prompt and the native audio boundary/directive framing.
