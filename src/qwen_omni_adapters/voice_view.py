"""A local, read-only grant for the device owner's speech excerpts."""

from __future__ import annotations

import os
import secrets
from pathlib import Path


def local_voice_view_key(state_root: Path, *, create: bool = False) -> str:
    path = state_root / "voice-view.key"
    try:
        with path.open(encoding="ascii") as source:
            key = source.read(65).strip()
        return key if len(key) == 43 and all(char.isalnum() or char in "-_" for char in key) else ""
    except FileNotFoundError:
        if not create:
            return ""
    except (OSError, UnicodeError):
        return ""

    key = secrets.token_urlsafe(32)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as target:
            target.write(key + "\n")
    except FileExistsError:
        return local_voice_view_key(state_root)
    except OSError:
        return ""
    return key
