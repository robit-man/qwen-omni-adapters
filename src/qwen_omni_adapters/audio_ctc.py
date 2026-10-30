"""Bounded training-lexicon decoding for the English shared-audio-head pilot.

This is an offline candidate decoder, not the authoritative audio-comprehension
path. It loads no language model and enables no partial-ASR or streaming route.
Unknown words require the existing full audio bridge; the lexicon is not a
general multilingual recognition vocabulary.
"""

from __future__ import annotations

import functools
import math
import re

import numpy as np


def _add(x, y):
    if x == -math.inf:
        return y
    if y == -math.inf:
        return x
    maximum = max(x, y)
    return maximum + math.log1p(math.exp(min(x, y) - maximum))


class CTCLexiconDecoder:
    def __init__(self, words, pieces):
        if (
            not 1 <= len(words) <= 100_000
            or sum(len(w) for w in words) > 2_000_000
            or any(
                not isinstance(w, str) or not re.fullmatch(r"[a-z0-9']{1,128}", w) for w in words
            )
        ):
            raise ValueError("invalid bounded English training lexicon")
        if not 2 <= len(pieces) <= 2048 or any(
            not isinstance(p, str) or len(p) > 128 for p in pieces
        ):
            raise ValueError("invalid CTC token pieces")
        self.words = frozenset(words)
        self.prefixes = {word[:i] for word in words for i in range(1, len(word) + 1)} | {""}
        self.pieces = tuple(pieces)
        self.advance = functools.lru_cache(maxsize=65_536)(self._advance)

    def _advance(self, current, symbol):
        piece = self.pieces[symbol]
        if not piece:
            return None
        for char in piece:
            if char == " ":
                if current and current not in self.words:
                    return None
                current = ""
            elif current + char in self.prefixes:
                current += char
            else:
                return None
        return current

    def decode(self, logs, *, width=16, top=32):
        logs = np.asarray(logs)
        if (
            logs.ndim != 2
            or logs.shape[1] != len(self.pieces)
            or not 1 <= logs.shape[0] <= 640
            or not np.isfinite(logs).all()
            or not 1 <= width <= 256
            or not 1 <= top <= 2048
        ):
            raise ValueError("invalid bounded CTC lattice")
        beams = {((), ""): (0.0, -math.inf)}
        for frame in logs:
            next_beams = {}

            def update(key, *, blank=None, nonblank=None, current=next_beams):
                pb, pn = current.get(key, (-math.inf, -math.inf))
                current[key] = (
                    pb if blank is None else _add(pb, blank),
                    pn if nonblank is None else _add(pn, nonblank),
                )

            symbols = {int(i) for i in np.argsort(-frame, kind="stable")[:top]} - {0}
            for (prefix, word), (pb, pn) in beams.items():
                total = _add(pb, pn)
                update((prefix, word), blank=total + float(frame[0]))
                candidates = symbols | ({prefix[-1]} if prefix else set())
                for symbol in candidates:
                    value = float(frame[symbol])
                    same = prefix and symbol == prefix[-1]
                    if same:
                        update((prefix, word), nonblank=pn + value)
                    current = self.advance(word, symbol)
                    if current is not None:
                        update(
                            ((*prefix, symbol), current), nonblank=(pb if same else total) + value
                        )
            beams = dict(
                sorted(next_beams.items(), key=lambda item: _add(*item[1]), reverse=True)[:width]
            )
        complete = {key: p for key, p in beams.items() if not key[1] or key[1] in self.words}
        return max(complete, key=lambda key: _add(*complete[key]))[0] if complete else None
