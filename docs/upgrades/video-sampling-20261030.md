# Bounded native video sampling

Acceptance: per-clip `sampling.fps` and `sampling.max_frames` must change the
frames actually decoded and counted, with independent limits for multiple clips.
The adapter's context-overflow retries cannot work if the native server discards
those fields. Before this patch, limits of 1, 3 and 16 all counted 1,101 tokens for
the same six-second color/speech clip.

Patch 16 validates sampling on each video media part, carries it through both
native inference and input-token counting, and applies a decoder frame cap.
For known-duration clips, reduced budgets span the original timeline; each
selected frame retains a timestamp. The caller cannot inject the server's
internal per-file table. Defaults retain legacy sampling. Unknown-duration media
has a bounded prefix; this check does not establish full-span sampling for it.

Actual native parser/tokenizer/FFmpeg results on the same clip:

| Request | Decoded frames | Input tokens | Timestamps |
| --- | ---: | ---: | --- |
| Legacy default | 24 | 1,101 | Legacy cadence |
| 1 fps, maximum 1 | 1 | 108 | 0 s |
| 1 fps, maximum 3 | 3 | 290 | 0, 2, 4 s |
| 1 fps, maximum 16 | 6 | 563 | 0 through 5 s |

Decoded RGB means prove red, blue and green are sampled in source order. Two
clips retain independent 1/3-frame caps; a repeated request has identical counts.
Invalid rates, zero/fractional limits, boolean rates and malformed options are
rejected. A spoofed internal table leaves default behavior unchanged. These
counts use the legacy baseline vocabulary/projector and are not universal model
or image-resolution measurements.

A separate actual Ornith primary run at 4K context, using the repaired projection,
preserves the red → blue → green timeline and exact sentence at both 3- and
16-frame caps. Its visual and acoustic passes share one resident primary. This
is an A100 runtime regression check; Ornith's release qualification remains on
Orin. Six seeded full-reference TTS outputs remain PCM-identical with caching
on/off and pass raw-PCM continuity checks.

The complete source gate passes 6,281 tests. All sixteen patches apply in order
to the pinned backend; all changed files
match the compiled tree. See [the native receipts](video-sampling-20261030.json)
and `runtime/qualify_video_sampling.cpp`. Native helpers must be rebuilt against
the updated headers. No environmental-audio or full-release acceptance follows
from passing these video and PCM controls.
