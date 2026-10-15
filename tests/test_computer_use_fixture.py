from __future__ import annotations

import copy
import json
import threading
from http.server import ThreadingHTTPServer

import httpx
import pytest

from qwen_omni_adapters.computer_use import ComputerUseError
from runtime.verify_computer_use import (
    FixtureState,
    fixture_handler,
    fixture_page,
    prescribed_plan,
    validate_plan,
)


def phases(state: FixtureState) -> None:
    assert state.submit("validate", {"values": state.initial})["accepted"]
    assert state.submit("preview", {"values": state.initial})["accepted"]
    assert state.submit("approve", {"values": state.expected})["accepted"]


def test_exact_phases_and_pixel_hit_are_required_for_authoritative_completion() -> None:
    state = FixtureState()
    phases(state)
    result = state.submit("apply", {"values": state.expected, "point": [310, 100]})
    assert result == {"accepted": True, "marker": state.marker}
    assert state.summary()["completed"] is True
    assert state.summary()["phase"] == "complete"


@pytest.mark.parametrize("phase", ["preview", "approve", "apply", "unknown"])
def test_missing_prior_phase_never_emits_completion_marker(phase: str) -> None:
    state = FixtureState()
    result = state.submit(phase, {"values": state.expected, "point": [310, 100]})
    assert result == {"accepted": False, "marker": ""}
    assert not state.summary()["completed"]


@pytest.mark.parametrize("field,value", [
    ("Recipient", "Wrong person"), ("Reference", "OMNI-4828"),
    ("Quantity", "4"), ("Route", "east"), ("Priority", "standard"),
    ("Inspect", False), ("Notify", False), ("Nonce", "replayed"),
])
def test_wrong_initial_values_fail_validation(field, value) -> None:
    state = FixtureState()
    result = state.submit("validate", {"values": {**state.initial, field: value}})
    assert not result["accepted"] and not result["marker"]
    assert state.phase == "initial"


def test_changed_preview_and_review_cannot_pass_final_submission() -> None:
    state = FixtureState()
    phases(state)
    assert not state.submit("apply", {"values": {**state.expected, "Reviewed": False}, "point": [310, 100]})["accepted"]
    state.preview_digest = "different-preview"
    assert not state.submit("apply", {"values": state.expected, "point": [310, 100]})["accepted"]
    assert not state.completed


@pytest.mark.parametrize("point", [[90, 100], [530, 100], [310, 40], [True, 100], [float("nan"), 100], [310], None, "310,100"])
def test_wrong_or_invalid_canvas_coordinates_never_complete(point) -> None:
    state = FixtureState()
    phases(state)
    assert not state.submit("apply", {"values": state.expected, "point": point})["accepted"]
    assert not state.completed


def test_semantic_baseline_is_an_explicit_distinct_gate() -> None:
    state = FixtureState(semantic_only=True)
    phases(state)
    assert not state.submit("apply", {"values": state.expected, "point": [310, 100]})["accepted"]
    assert state.submit("apply", {"values": state.expected, "semantic": True})["accepted"]
    assert state.completed


def test_canvas_target_has_no_actionable_semantic_control() -> None:
    state = FixtureState()
    page = fixture_page(state)
    assert '<canvas id="motor" width="620" height="200" hidden>' in page
    assert 'name="Apply"' not in page and '>Apply</button>' not in page
    assert 'role="button"' not in page
    assert state.marker not in page
    assert "Required: " + state.nonce in page
    assert 'name="Recipient" type="text" autocomplete="off"></label><small>' in page


def test_prescribed_plan_is_bounded_and_uses_observed_effects() -> None:
    state = FixtureState()
    plan = validate_plan(prescribed_plan(state), state)
    assert len(plan["steps"]) == 16
    assert plan["steps"][-1]["target"]["name"] == "blue Apply button"
    assert all(step["expect"] for step in plan["steps"])
    assert plan["final_predicates"] == [{"kind": "text", "text": state.marker}]


@pytest.mark.parametrize("change", ["code", "short", "duplicate", "fake-final", "extra"])
def test_model_plan_cannot_widen_fixture_contract(change: str) -> None:
    state = FixtureState()
    plan = copy.deepcopy(prescribed_plan(state))
    if change == "code":
        plan["steps"][0]["args"]["code"] = "arbitrary input"
    elif change == "short":
        plan["steps"] = plan["steps"][:2]
    elif change == "duplicate":
        plan["steps"][1]["id"] = plan["steps"][0]["id"]
    elif change == "fake-final":
        plan["final_predicates"] = [{"kind": "text", "text": "Approved dispatch"}]
    else:
        plan["permissions"] = "allow"
    with pytest.raises(ComputerUseError):
        validate_plan(plan, state)


def test_real_http_fixture_retains_server_truth() -> None:
    state = FixtureState(semantic_only=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), fixture_handler(state))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}") as client:
            assert client.get("/").status_code == 200
            for phase in ("validate", "preview", "approve", "apply"):
                values = state.initial if phase in {"validate", "preview"} else state.expected
                response = client.post("/" + phase, json={"values": values, "semantic": phase == "apply"})
                assert response.status_code == 200
                assert response.json()["accepted"]
            assert state.completed
            assert state.marker == json.loads(response.text)["marker"]
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)
