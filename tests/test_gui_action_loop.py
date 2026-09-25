from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from portal.gui import GuiAutomation
from runtime.verify_gui_action_loop import ChallengeState, _is_stale_gui_fixture, _page


def test_gui_fixture_reclaims_only_its_own_stale_task() -> None:
    fixture = {
        "status": "running",
        "objective": (
            "Open http://127.0.0.1:36867/ in the visible Chromium window. "
            "Complete all three instructions drawn inside its canvas using fresh "
            "browser_interact viewport screenshots and normalized_1000 visual_click "
            "actions."
        ),
        "completion_criteria": "The canvas shows GUI-ACTION-PASS-AB89BFE9.",
    }

    assert _is_stale_gui_fixture(fixture) is True
    assert _is_stale_gui_fixture({**fixture, "status": "completed"}) is False
    assert _is_stale_gui_fixture(
        {
            "status": "running",
            "objective": "Open a real customer page in Chromium.",
            "completion_criteria": "Submit the requested form.",
        }
    ) is False


def _decoded_image(result: dict[str, Any]) -> Image.Image:
    encoded = result["screenshot"]["data"]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


class SyntheticDesktop(GuiAutomation):
    """Exercise the production capture and input transform against real pixels."""

    def __init__(
        self,
        *,
        display: tuple[int, int],
        window: tuple[int, int, int, int],
        target: tuple[int, int],
    ) -> None:
        super().__init__()
        self.display = display
        self.window = window
        self.target = target
        self.commands: list[list[str]] = []
        self.clicks: list[tuple[int, int]] = []
        self.hit = False
        self.desktop = Image.new("RGB", display, "#161b22")
        draw = ImageDraw.Draw(self.desktop)
        x, y, width, height = window
        draw.rectangle((x, y, x + width - 1, y + height - 1), fill="#f7f7f7")
        draw.rectangle((x, y, x + width - 1, y + 31), fill="#30363d")
        tx, ty = target
        draw.rectangle((tx - 12, ty - 12, tx + 12, ty + 12), fill="#0969da")

    def _desktop_state(self) -> tuple[int, int, dict[str, Any]]:
        x, y, width, height = self.window
        return (
            self.display[0],
            self.display[1],
            {
                "id": "synthetic-browser",
                "title": "Visual click fixture",
                "bounds": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "screen": 0,
                },
            },
        )

    def _run(self, command: list[str]) -> str:
        self.commands.append(command)
        if command[0] == "gnome-screenshot":
            self.desktop.save(Path(command[-1]), format="PNG")
        elif command[:3] == ["xdotool", "mousemove", "--sync"] and "click" in command:
            x, y = int(command[3]), int(command[4])
            self.clicks.append((x, y))
            tx, ty = self.target
            if abs(x - tx) <= 12 and abs(y - ty) <= 12:
                self.hit = True
                draw = ImageDraw.Draw(self.desktop)
                draw.rectangle(
                    (tx - 12, ty - 12, tx + 12, ty + 12), fill="#1a7f37"
                )
        return ""


@pytest.mark.parametrize(
    ("display", "window", "target_relative"),
    [
        ((1280, 900), (54, 37, 1042, 800), (341, 473)),
        ((1920, 1080), (420, 110, 720, 640), (615, 92)),
        ((1080, 1920), (9, 503, 1010, 760), (41, 701)),
    ],
)
def test_active_window_pixels_and_clicks_share_one_exact_frame(
    monkeypatch: pytest.MonkeyPatch,
    display: tuple[int, int],
    window: tuple[int, int, int, int],
    target_relative: tuple[int, int],
) -> None:
    monkeypatch.setattr("portal.gui.shutil.which", lambda _name: "/usr/bin/tool")
    wx, wy, width, height = window
    rx, ry = target_relative
    target = (wx + rx, wy + ry)
    gui = SyntheticDesktop(display=display, window=window, target=target)

    observed = gui.act("fixture", {"action": "snapshot"})
    image = _decoded_image(observed)

    assert image.size == (width, height)
    assert observed["coordinate_space"] == {
        "name": "active_window",
        "origin_x": wx,
        "origin_y": wy,
        "width": width,
        "height": height,
        "coordinate_units": ["pixels", "normalized_1000"],
    }
    assert image.getpixel((rx, ry)) == (9, 105, 218)

    clicked = gui.act(
        "fixture",
        {"action": "click", "x": rx, "y": ry, "wait_ms": 0},
    )

    assert gui.clicks == [target]
    assert gui.hit is True
    assert clicked["visual_change"]["comparable"] is True
    assert clicked["visual_change"]["materially_changed"] is True
    assert _decoded_image(clicked).getpixel((rx, ry)) == (26, 127, 55)


def test_active_window_normalized_point_maps_into_the_observed_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("portal.gui.shutil.which", lambda _name: "/usr/bin/tool")
    gui = SyntheticDesktop(
        display=(1280, 900),
        window=(54, 37, 1042, 800),
        target=(574, 437),
    )
    gui.act("fixture", {"action": "snapshot"})

    gui.act(
        "fixture",
        {
            "action": "click",
            "x": 500,
            "y": 500,
            "coordinate_unit": "normalized_1000",
            "wait_ms": 0,
        },
    )

    assert gui.clicks == [(574, 437)]
    assert gui.hit is True


def test_missed_target_is_reported_as_unchanged_not_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("portal.gui.shutil.which", lambda _name: "/usr/bin/tool")
    gui = SyntheticDesktop(
        display=(1280, 900),
        window=(80, 60, 900, 700),
        target=(500, 400),
    )
    gui.act("fixture", {"action": "snapshot"})

    result = gui.act(
        "fixture",
        {"action": "click", "x": 100, "y": 100, "wait_ms": 0},
    )

    assert gui.clicks == [(180, 160)]
    assert gui.hit is False
    assert result["visual_change"] == {
        "comparable": True,
        "changed_sample_count": 0,
        "changed_sample_fraction": 0.0,
        "materially_changed": False,
    }


def test_full_screen_observation_keeps_screen_coordinates_for_next_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("portal.gui.shutil.which", lambda _name: "/usr/bin/tool")
    gui = SyntheticDesktop(
        display=(1280, 900),
        window=(80, 60, 900, 700),
        target=(1100, 820),
    )
    initial = gui.act(
        "fixture", {"action": "snapshot", "coordinate_space": "screen"}
    )
    assert _decoded_image(initial).size == (900, 700)
    observed = gui.act(
        "fixture", {"action": "snapshot", "coordinate_space": "screen"}
    )

    assert _decoded_image(observed).size == (1280, 900)
    clicked = gui.act(
        "fixture", {"action": "click", "x": 1100, "y": 820, "wait_ms": 0}
    )

    assert clicked["coordinate_space"]["name"] == "screen"
    assert gui.clicks == [(1100, 820)]
    assert gui.hit is True


def test_live_canvas_fixture_requires_each_distinct_visual_target() -> None:
    state = ChallengeState()

    assert state.attempt(10, 10)["accepted"] is False
    assert state.snapshot()["stage"] == 0
    for expected_stage, (x, y) in enumerate(state.expected_points, start=1):
        result = state.attempt(x, y)
        assert result["accepted"] is True
        assert result["stage"] == expected_stage

    assert state.snapshot() == {
        "stage": 3,
        "complete": True,
        "marker": state.marker,
        "hits": [
            {"stage": 0, "x": 10, "y": 10, "accepted": False},
            {"stage": 0, "x": 710, "y": 265, "accepted": True},
            {"stage": 1, "x": 630, "y": 380, "accepted": True},
            {"stage": 2, "x": 275, "y": 505, "accepted": True},
        ],
    }


def test_live_canvas_fixture_keeps_targets_out_of_dom_controls() -> None:
    state = ChallengeState()
    html = _page(state.marker).decode()

    assert '<canvas id="board"' in html
    assert "BLUE TRIANGLE" in html
    assert "AMBER STAR" in html
    assert "PURPLE DIAMOND" in html
    assert state.marker in html
    assert "<button" not in html
    assert "<a " not in html
