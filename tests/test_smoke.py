from __future__ import annotations

import pytest

from portal.smoke import require_input_transcript


def test_audio_smoke_requires_attributed_speech_not_an_audio_observation() -> None:
    result = {
        "message": {"content": "A fan is audible."},
        "adapter": {"audio_observation": "steady mechanical hum"},
    }

    with pytest.raises(RuntimeError, match="no tagged speech transcript"):
        require_input_transcript(result)


def test_audio_smoke_accepts_and_returns_a_tagged_transcript() -> None:
    result = {
        "message": {"content": "hello"},
        "adapter": {"input_transcript": "  A deliberately uncommon utterance.  "},
    }

    assert require_input_transcript(result) == "A deliberately uncommon utterance."
