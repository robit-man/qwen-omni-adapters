"""Where the machine is, told to the model the way the time is.

Asked about the weather or what is nearby, a model with no location invents
one. A coarse city is enough to reason from -- and has to be described as
coarse, or the model answers as though it knows the street.

Everything here is best-effort by design: a machine with no internet still has
a microphone, and no conversation should wait on a geolocation service.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.place import Place, PlaceLookup  # noqa: E402


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def test_a_successful_lookup_is_remembered(monkeypatch) -> None:
    import harness.place as place_module

    monkeypatch.setattr(
        place_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            b'{"status":"success","city":"Portland","regionName":"Oregon",'
            b'"country":"United States","timezone":"America/Los_Angeles"}'
        ),
    )

    lookup = PlaceLookup()
    found = lookup.refresh()

    assert found.city == "Portland"
    assert found.timezone == "America/Los_Angeles"
    assert found.describe() == "Portland, Oregon, United States"


def test_being_offline_leaves_the_location_unknown(monkeypatch) -> None:
    """And must not raise: the microphone works without a network."""

    import harness.place as place_module

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("Network is unreachable")

    monkeypatch.setattr(place_module.urllib.request, "urlopen", refuse)

    lookup = PlaceLookup()

    assert not lookup.refresh()
    assert not lookup.place


def test_a_failing_service_is_not_retried_by_every_turn(monkeypatch) -> None:
    """A slow lookup on each turn would be a delay before every answer."""

    import harness.place as place_module

    calls: list[int] = []

    def refuse(*_args: object, **_kwargs: object) -> None:
        calls.append(1)
        raise OSError("timed out")

    monkeypatch.setattr(place_module.urllib.request, "urlopen", refuse)

    lookup = PlaceLookup(refresh_s=3600)
    lookup.refresh()
    # Asking again inside the refresh window must not trigger another request.
    assert not lookup.place
    assert not lookup.place

    assert len(calls) == 1


def test_an_unsuccessful_payload_is_ignored(monkeypatch) -> None:
    import harness.place as place_module

    monkeypatch.setattr(
        place_module.urllib.request,
        "urlopen",
        lambda *_a, **_k: FakeResponse(b'{"status":"fail","message":"private range"}'),
    )

    assert not PlaceLookup().refresh()


def test_the_location_is_offered_as_approximate() -> None:
    """A model told a city without the caveat answers as if it knows the street."""

    from harness.call import grounding_preamble

    preamble = grounding_preamble(place=Place("Portland", "Oregon", "United States"))

    assert "Portland, Oregon, United States" in preamble
    assert "approximate" in preamble
    assert "never as the speaker's exact position" in preamble


def test_without_a_location_the_grounding_says_nothing_about_place() -> None:
    from harness.call import grounding_preamble

    preamble = grounding_preamble()

    assert "The current date and time is" in preamble
    assert "This machine is in" not in preamble
