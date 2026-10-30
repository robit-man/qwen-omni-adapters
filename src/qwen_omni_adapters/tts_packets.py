"""Runtime packet overrides; automatic selection stays with the speech worker."""
from __future__ import annotations


def stream_frame_setting(value: str | None) -> int | None:
    if value is None or value.strip().lower() in {"", "auto"}:
        return None
    frames = int(value)
    if not 1 <= frames <= 72:
        raise ValueError("OMNI_TTS_STREAM_FRAMES must be auto or an integer from 1 to 72")
    return frames
