"""Unknown words must not become empty ASR labels or corrupt acoustic targets."""
from types import SimpleNamespace

import numpy as np
import pytest

from qwen_omni_adapters.partial_supervision import install_partial_speech_collator

# Captured from the actual supported Qwen2 tokenizers. Token 1721 spans the
# speech closing bracket and the next opening bracket (character offsets38:40).
TARGET = '<speech_transcript></speech_transcript><audio_observation>Audible sound events: Laughter.</audio_observation>'
IDS = [27, 85186, 7714, 1186, 1419, 85186, 7714, 1186, 1721, 16245,
       29415, 8503, 22733, 643, 1196, 4950, 4216, 25, 4770, 7109,
       3786, 16245, 29415, 8503, 29]
OFFSETS = [(0, 1), (1, 7), (7, 13), (13, 18), (18, 21), (21, 27),
           (27, 33), (33, 38), (38, 40), (40, 45), (45, 49), (49, 57),
           (57, 59), (59, 61), (61, 65), (65, 71), (71, 78), (78, 79),
           (79, 82), (82, 88), (88, 91), (91, 96), (96, 100),
           (100, 108), (108, 109)]


class GoldenTokenizer:
    def __call__(self, target, *, add_special_tokens, return_offsets_mapping):
        assert target == TARGET and not add_special_tokens and return_offsets_mapping
        return {'input_ids': IDS.copy(), 'offset_mapping': OFFSETS.copy()}


class OriginalCollator:
    tokenizer = GoldenTokenizer()
    suffix_prompt_ids = [11, 12]
    end_ids = [151645, 198]

    def __call__(self, records):
        suffix = self.suffix_prompt_ids + IDS + self.end_ids
        return {'suffix_input_ids': np.tile(suffix, (len(records), 1)),
                'suffix_labels': np.tile([-100, -100] + IDS + self.end_ids, (len(records), 1)),
                'input_features': np.ones((len(records), 3, 5), np.float32)}


def record(**kwargs):
    return {'assistant_target': TARGET, 'transcript': '', **kwargs}


def install(collator=OriginalCollator):
    training = SimpleNamespace(AudioBridgeCollator=collator)
    original, partial = install_partial_speech_collator(training)
    assert original is collator and training.AudioBridgeCollator is partial
    return partial()


def test_unknown_transcript_masks_boundary_token_and_keeps_acoustic_stop_targets():
    row = record(transcript_supervision='unavailable')
    before = OriginalCollator()([row])
    after = install()([row])
    np.testing.assert_array_equal(after['suffix_input_ids'], before['suffix_input_ids'])
    np.testing.assert_array_equal(after['input_features'], before['input_features'])
    expected = [-100] * 11 + IDS[9:] + [151645, 198]
    assert after['suffix_labels'][0].tolist() == expected
    assert before['suffix_labels'][0, 10] == 1721
    assert after['suffix_labels'][0, 10] == -100
    assert after['suffix_labels'][0, 11] == 16245


@pytest.mark.parametrize('supervision', [None, 'verified'])
def test_ordinary_transcribed_replay_is_unchanged(supervision):
    row = record(transcript='Verified words', transcript_supervision=supervision)
    before = OriginalCollator()([row])
    after = install()([row])
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])


def test_partial_labels_are_isolated_within_a_batch():
    rows = [record(), record(transcript_supervision='unavailable'), record()]
    before = OriginalCollator()(rows)
    after = install()(rows)
    np.testing.assert_array_equal(after['suffix_labels'][[0, 2]], before['suffix_labels'][[0, 2]])
    assert np.all(after['suffix_labels'][1, :11] == -100)


@pytest.mark.parametrize('overrides', [
    {'transcript': 'Unverified words'},
    {'assistant_target': '<speech_transcript>Unverified words</speech_transcript><audio_observation>Voices</audio_observation>'},
    {'assistant_target': '<audio_observation>Voices</audio_observation>'},
])
def test_unknown_transcript_with_unverified_words_or_missing_placeholder_is_rejected(overrides):
    with pytest.raises(ValueError, match='empty unscored placeholder'):
        install()([record(transcript_supervision='unavailable', **overrides)])


def test_token_alignment_mismatch_is_rejected_before_any_training_step():
    class MisalignedCollator(OriginalCollator):
        def __call__(self, rows):
            batch = super().__call__(rows)
            batch['suffix_input_ids'][0, 4] += 1
            return batch

    with pytest.raises(ValueError, match='token alignment'):
        install(MisalignedCollator)([record(transcript_supervision='unavailable')])
