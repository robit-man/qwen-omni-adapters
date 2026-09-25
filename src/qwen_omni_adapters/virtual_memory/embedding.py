"""Small, model-agnostic embedding providers for the external memory indexes."""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Sequence

_WORD_RE = re.compile(r"[\w.$:/-]+", re.UNICODE)


class HashingEmbedder:
    """Deterministic dense lexical fallback with no resident model weights.

    This is deliberately not advertised as a learned semantic encoder.  It
    supplies an always-available fuzzy/dense retrieval channel while keeping
    the embedding interface replaceable by a learned local or remote model.
    Character features make identifiers, error strings, and small spelling
    variations less brittle than exact/BM25 search alone.
    """

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 64:
            raise ValueError("hash embedding dimensions must be at least 64")
        self.dimensions = int(dimensions)

    def __call__(self, text: str) -> list[float]:
        normalized = " ".join(str(text or "").casefold().split())
        vector = [0.0] * self.dimensions
        if not normalized:
            return vector
        words = _WORD_RE.findall(normalized)
        features = [f"w:{word}" for word in words]
        features.extend(
            f"b:{left}\0{right}" for left, right in zip(words, words[1:], strict=False)
        )
        compact = f"  {normalized}  "
        features.extend(f"c:{compact[index:index + 3]}" for index in range(len(compact) - 2))
        for feature in features:
            digest = zlib.crc32(feature.encode("utf-8"))
            index = digest % self.dimensions
            sign = 1.0 if digest & 0x80000000 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def validate_embedding(vector: Sequence[float] | None) -> list[float] | None:
    """Normalize an injected learned embedding without coupling to its backend."""

    if vector is None:
        return None
    values = [float(value) for value in vector]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("embedding must contain finite values")
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values
