from __future__ import annotations

from runtime.verify_browser_gauntlet import (
    _FORM_EXPECTED,
    _PHOTO_RECORDS,
    GauntletState,
    _form_page,
    _grid_page,
    _interface_page,
    _is_stale_browser_gauntlet,
)


def test_browser_gauntlet_reclaims_only_its_own_stale_task() -> None:
    fixture = {
        "status": "running",
        "objective": (
            "Open http://127.0.0.1:31991/ in the visible Chromium window. "
            "Complete the dispatch form, every image-selection round, and transfer."
        ),
        "completion_criteria": "Observe BROWSER-GAUNTLET-PASS-A1B2C3D4.",
    }

    assert _is_stale_browser_gauntlet(fixture) is True
    assert _is_stale_browser_gauntlet({**fixture, "status": "completed"}) is False
    assert (
        _is_stale_browser_gauntlet(
            {
                "status": "running",
                "objective": "Open a customer form.",
                "completion_criteria": "Submit it.",
            }
        )
        is False
    )


def test_exact_form_state_rejects_then_accepts() -> None:
    state = GauntletState(seed=11)
    expected = dict(_FORM_EXPECTED)

    assert state.submit_form({**expected, "dispatch_code": "ORIN-4828"}) is False
    assert state.submit_form(expected) is True
    assert state.snapshot()["form_passed"] is True


def test_randomized_visual_rounds_require_targets_and_replacement() -> None:
    for seed in range(8):
        state = GauntletState(seed=seed)
        total_hits = 0
        for expected_round in range(3):
            while True:
                view = state.grid_view()
                assert view["round"] == expected_round
                assert "target" not in view
                targets = state.answer_indices()
                if not targets:
                    break
                result = state.click_grid(targets[0])
                assert result["accepted"] is True
                total_hits += 1
            assert state.verify_grid() is True
        assert state.grid_view()["complete"] is True
        assert total_hits == state.expected_grid_hits == 7
        assert state.snapshot()["grid_misses"] == []


def test_visual_grid_records_miss_and_refuses_early_verify() -> None:
    state = GauntletState(seed=17)
    view = state.grid_view()
    answers = set(state.answer_indices())
    miss = next(index for index in range(len(view["tiles"])) if index not in answers)

    assert state.click_grid(miss)["accepted"] is False
    assert state.verify_grid() is False
    snapshot = state.snapshot()
    assert len(snapshot["grid_misses"]) == 1
    assert snapshot["grid_verifications"] == [{"round": 0, "accepted": False}]


def test_nested_interface_is_server_ordered() -> None:
    state = GauntletState(seed=19)

    assert state.interface_event("routing") is False
    for event in ["operations", "routing", "orbital", "gale", "advanced", "delta-7"]:
        assert state.interface_event(event) is True
    assert state.interface_event("delta-7") is False
    assert state.snapshot()["interface_passed"] is True


def test_final_state_requires_correct_drag_and_inferred_seal() -> None:
    state = GauntletState(seed=23)
    assert state.submit_form(dict(_FORM_EXPECTED))
    for event in ["operations", "routing", "orbital", "gale", "advanced", "delta-7"]:
        assert state.interface_event(event) is True
    for _round in range(3):
        while True:
            targets = state.answer_indices()
            if not targets:
                break
            state.click_grid(targets[0])
        assert state.verify_grid() is True

    assert state.record_drop("amber") is False
    assert state.submit_final("K7-MARS") is False
    assert state.record_drop("purple") is True
    assert state.submit_final("T4-DUST") is False
    assert state.submit_final("K7-MARS") is True
    assert state.snapshot()["complete"] is True


def test_pages_keep_grid_targets_out_of_dom_text() -> None:
    form = _form_page().decode()
    interface = _interface_page().decode()
    grid = _grid_page().decode()

    for input_type in (
        "text",
        "email",
        "password",
        "search",
        "tel",
        "url",
        "number",
        "date",
        "time",
        "datetime-local",
        "month",
        "week",
        "range",
        "color",
        "radio",
        "checkbox",
        "file",
        "hidden",
        "button",
        "reset",
        "submit",
        "image",
    ):
        assert f'type="{input_type}"' in form
    assert "multiple" in form
    assert "Operations ▾" in interface
    assert "Advanced clearance" in interface
    assert "traffic lights" not in grid
    assert "Nine unlabeled real photograph tiles" in grid
    assert "trafficLight(" not in grid
    assert "objectScene(" not in grid
    assert "data-crate" not in grid


def test_photo_manifest_is_opaque_pinned_and_real() -> None:
    ids = {str(item["id"]) for item in _PHOTO_RECORDS}
    kinds = {str(item["kind"]) for item in _PHOTO_RECORDS}

    assert len(ids) == len(_PHOTO_RECORDS) == 16
    assert {"traffic_light", "bicycle", "crosswalk"} <= kinds
    assert all(str(item["id"]).startswith("p") for item in _PHOTO_RECORDS)
    assert all(len(str(item["sha256"])) == 64 for item in _PHOTO_RECORDS)
    assert all(str(item["url"]).startswith("https://thumb.wikimedia.org/") for item in _PHOTO_RECORDS)
    assert all(str(item["source"]).startswith("https://commons.wikimedia.org/") for item in _PHOTO_RECORDS)
