"""Build-only audio labels; no Torch or fine-tuning dependency at runtime."""


def install_partial_speech_collator(training):
    """Add unknown-transcript masking to an audio-bridge trainer's collator.

    Call before constructing the collator. A record explicitly marked
    ``transcript_supervision: unavailable`` must contain an empty transcript
    placeholder and an acoustic observation. Its speech target, including any
    token spanning the following tag boundary, is unscored. Other records and
    all encoder inputs retain the original collator's behavior.

    ``training`` supplies AudioBridgeCollator, whose batch exposes
    suffix_input_ids/suffix_labels and whose tokenizer supports offsets. This
    optional build integration neither imports the trainer nor changes an
    inference request. The returned classes allow callers to inspect the
    original and installed collators.
    """
    original = training.AudioBridgeCollator

    class PartialSpeechCollator(original):
        def __call__(self, records):
            batch = super().__call__(records)
            for index, record in enumerate(records):
                if record.get('transcript_supervision') != 'unavailable':
                    continue
                target = str(record['assistant_target'])
                boundary_text = '<speech_transcript></speech_transcript>'
                if record.get('transcript') or not target.startswith(boundary_text + '<audio_observation>'):
                    raise ValueError('Unknown transcript supervision requires an empty unscored placeholder')
                encoded = self.tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
                expected = encoded['input_ids'] + self.end_ids
                start = len(self.suffix_prompt_ids)
                actual = batch['suffix_input_ids'][index, start:start+len(expected)].tolist()
                if actual != expected:
                    raise ValueError('Partial-label token alignment differs from the real collator')
                # One BPE token may contain the speech closing bracket and the
                # following observation opening bracket. Mask it in full.
                count = sum(begin < len(boundary_text) for begin, end in encoded['offset_mapping'])
                if not 0 < count < len(encoded['input_ids']):
                    raise ValueError('Partial supervision left no acoustic target')
                batch['suffix_labels'][index, start:start+count] = -100
            return batch

    training.AudioBridgeCollator = PartialSpeechCollator
    return original, PartialSpeechCollator
