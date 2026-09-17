"""Decide whether a spoken turn actually asked to be looked at.

Attaching a camera frame to every turn makes the picture the subject. Asked
"can you hear me okay?", a model handed an image answers the question and then
starts describing the room, which is not how anyone talks.

So the cameras are not offered unless the words reach for them. This is a
deliberately plain rule -- a list of ways people ask to be looked at -- rather
than a model call, because it runs before every turn and has to be instant, and
because being wrong in the quiet direction is cheap: the follow-up pass can
still look, and the person can always ask again more plainly.
"""

from __future__ import annotations

import re

# Verbs and nouns that only make sense about something visible.
_LOOK = r"(?:look|see|seeing|watch|watching|show|showing|view|glance|check out)"
_VISUAL_NOUN = (
    r"(?:camera|cameras|screen|screens|monitor|picture|image|photo|video|"
    r"footage|room|desk|whiteboard|board|sign|label|text|writing|colou?r|"
    r"shirt|face|hand|hands|object|thing)"
)

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Direct requests to look.
    re.compile(rf"\b(?:can|could|would|will)\s+you\s+{_LOOK}\b"),
    re.compile(rf"\b{_LOOK}\s+at\b"),
    re.compile(r"\btake\s+a\s+(?:look|picture|photo|snapshot)\b"),
    # Questions about what is present.
    re.compile(rf"\bwhat\s+(?:do|can)\s+you\s+{_LOOK}\b"),
    re.compile(r"\bwhat(?:'s| is| are)\s+(?:this|that|these|those|here|in front)\b"),
    re.compile(r"\bwhat\s+am\s+i\s+(?:holding|wearing|pointing)\b"),
    re.compile(r"\bwho(?:'s| is| are)\s+(?:here|in the room|in front)\b"),
    re.compile(r"\bhow\s+many\b.{0,24}\b(?:here|in the room|on the desk)\b"),
    # Reading something visible.
    re.compile(r"\b(?:read|says?)\b.{0,20}\b(?:this|that|screen|sign|label|it)\b"),
    # Anything that names a visual thing alongside a visual verb.
    re.compile(rf"\b{_LOOK}\b.{{0,30}}\b{_VISUAL_NOUN}\b"),
    re.compile(rf"\b{_VISUAL_NOUN}\b.{{0,30}}\b{_LOOK}\b"),
    # Explicitly about what just happened in view.
    re.compile(r"\bwhat\s+just\s+happened\b"),
    re.compile(r"\bdid\s+you\s+(?:see|catch)\s+that\b"),
)

# Asking about a stretch of time wants a clip rather than a still.
_MOTION = re.compile(
    r"\b(?:what just happened|did (?:you )?(?:see|catch) that|"
    r"happening|moving|movement|motion|gesture|waving|just did)\b"
)


def wants_vision(transcript: str) -> bool:
    """Whether the words asked to be looked at."""

    text = " ".join((transcript or "").lower().split())
    if not text:
        return False
    return any(pattern.search(text) for pattern in _PATTERNS)


def wants_motion(transcript: str) -> bool:
    """Whether the question is about a stretch of time rather than a moment."""

    text = " ".join((transcript or "").lower().split())
    return bool(text) and bool(_MOTION.search(text))
