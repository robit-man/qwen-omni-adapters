"""Client-side decorated frames match only the exact surface minus declared margins."""
import pytest

from portal import native_ui
from portal.native_ui import NativeUiUnavailable

SURFACE = {"x": 94, "y": 4, "width": 977, "height": 1064}
FIREFOX_EXTENTS = {"left": 26, "right": 26, "top": 23, "bottom": 29}  # Observed on the Egg.
CONTENT = {"x": 120, "y": 27, "width": 925, "height": 1012}


@pytest.fixture
def x11(monkeypatch):
    state = {"extents": dict(FIREFOX_EXTENTS), "extent_reads": 0}
    monkeypatch.setattr(native_ui, "_x11_geometry", lambda _identity, *a, **k: dict(SURFACE))

    def extents(_identity, *a, **k):
        state["extent_reads"] += 1
        if state["extents"] is None:
            raise NativeUiUnavailable("native_window_frame_extents_unavailable")
        return dict(state["extents"])

    monkeypatch.setattr(native_ui, "_x11_frame_extents", extents)
    return state


def test_surface_and_exact_csd_content_match(x11):
    identity = {"window_id": "17", "process_id": 42}
    assert native_ui._frame_matches_surface(dict(SURFACE), identity) and x11["extent_reads"] == 0
    assert native_ui._frame_matches_surface(dict(CONTENT), identity) and x11["extent_reads"] == 1
    assert native_ui._surface_frames(identity) == [SURFACE, CONTENT]


@pytest.mark.parametrize("rect", [
    {**CONTENT, "x": 121}, {**CONTENT, "width": 924}, {**CONTENT, "y": 4},
    {"x": 107, "y": 15, "width": 951, "height": 1038},  # Half margins: never guessed.
])
def test_any_other_offset_or_size_is_rejected(x11, rect):
    assert not native_ui._frame_matches_surface(rect, {"window_id": "17", "process_id": 42})


def test_without_declared_extents_only_the_surface_matches(x11):
    x11["extents"] = None
    identity = {"window_id": "17", "process_id": 42}
    assert native_ui._surface_frames(identity) == [SURFACE]
    assert not native_ui._frame_matches_surface(dict(CONTENT), identity)
    assert native_ui._frame_matches_surface(dict(SURFACE), identity)


def test_degenerate_margins_add_no_rectangle(x11):
    x11["extents"] = {"left": 600, "right": 600, "top": 0, "bottom": 0}
    assert native_ui._surface_frames({"window_id": "17", "process_id": 42}) == [SURFACE]
