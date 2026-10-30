"""Shared validation for an explicitly supplied voice-reference transcript."""
from __future__ import annotations

MAX_REFERENCE_TEXT_CHARS = 2048


def reference_transcript(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("reference_text must be a string")
    if len(value) > MAX_REFERENCE_TEXT_CHARS or "\x00" in value:
        raise ValueError("reference_text must be at most 2048 characters without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("reference_text must contain valid Unicode") from exc
    return value.strip()
