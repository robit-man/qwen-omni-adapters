"""Language and regional-locale validation for Qwen3-TTS speech requests."""

from __future__ import annotations

import re

QWEN3_TTS_LANGUAGES = frozenset(
    {"zh", "en", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"}
)

_BCP47_LOCALE = re.compile(r"^[A-Za-z]{2}(?:-[A-Za-z0-9]{2,8})*$")


class VoiceLocaleError(ValueError):
    """Raised when a speech locale cannot be mapped to Qwen3-TTS."""


def normalize_voice_locale(value: object) -> tuple[str, str]:
    """Return ``(canonical_locale, qwen_language_code)`` for a BCP 47 tag.

    Qwen3-TTS exposes ISO 639-1 language tokens, while the adapter accepts a
    regional locale such as ``en-GB``. The full locale stays in the adapter
    layer and the primary language subtag is sent to the TTS graph.
    """

    candidate = str(value or "").strip()
    if not candidate or not _BCP47_LOCALE.fullmatch(candidate):
        raise VoiceLocaleError(
            "locale must be a BCP 47 tag such as en-US or en-GB"
        )
    subtags = candidate.split("-")
    language = subtags[0].lower()
    if language not in QWEN3_TTS_LANGUAGES:
        supported = ", ".join(sorted(QWEN3_TTS_LANGUAGES))
        raise VoiceLocaleError(
            f"locale language must be one of the Qwen3-TTS languages: {supported}"
        )
    canonical = [language]
    for subtag in subtags[1:]:
        if len(subtag) == 4 and subtag.isalpha():
            canonical.append(subtag.title())
        elif (len(subtag) == 2 and subtag.isalpha()) or (
            len(subtag) == 3 and subtag.isdigit()
        ):
            canonical.append(subtag.upper())
        else:
            canonical.append(subtag.lower())
    return "-".join(canonical), language
