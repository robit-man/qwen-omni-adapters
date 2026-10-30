import collections
import itertools
import math

import numpy as np
import pytest

from qwen_omni_adapters.audio_ctc import CTCLexiconDecoder


def test_lexicon_prefix_beam_matches_independent_exhaustive_ctc_paths():
    rng = np.random.default_rng(7)
    pieces = ["", "a", "b", " "]
    decoder = CTCLexiconDecoder({"a", "b", "ab"}, pieces)
    for _ in range(8):
        probabilities = rng.random((4, 4))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        exact = collections.defaultdict(float)
        for path in itertools.product(range(4), repeat=4):
            collapsed = tuple(c for i, c in enumerate(path) if c and (i == 0 or c != path[i - 1]))
            text = "".join(pieces[c] for c in collapsed)
            if all(word in decoder.words for word in text.split()):
                exact[collapsed] += math.prod(probabilities[i, c] for i, c in enumerate(path))
        assert decoder.decode(np.log(probabilities), width=256, top=4) == max(exact, key=exact.get)


def test_blank_repeats_and_word_boundaries_are_distinct_ctc_events():
    decoder = CTCLexiconDecoder({"a", "aa"}, ["", "a", " "])
    probabilities = np.full((5, 3), 1e-8)
    probabilities[np.arange(5), [1, 0, 1, 2, 1]] = 1
    assert decoder.decode(np.log(probabilities)) == (1, 1, 2, 1)
    assert decoder.decode(np.log(np.array([[0.999, 0.0005, 0.0005]]))) == ()


def test_candidate_decoder_rejects_unbounded_or_nonfinite_input():
    decoder = CTCLexiconDecoder({"a"}, ["", "a"])
    for value in (np.zeros((641, 2)), np.zeros((3, 3)), np.array([[0.0, np.nan]])):
        with pytest.raises(ValueError):
            decoder.decode(value)
    with pytest.raises(ValueError):
        CTCLexiconDecoder({"arbitrary unsupported punctuation!"}, ["", "a"])
