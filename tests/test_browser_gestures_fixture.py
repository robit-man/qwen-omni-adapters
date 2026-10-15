"""Fixture integrity tests. These do not claim real browser/model acceptance."""

from __future__ import annotations

import argparse
import copy
import json
import threading
from http.server import ThreadingHTTPServer

import httpx
import pytest

from qwen_omni_adapters.computer_use import ComputerUseError
from runtime.verify_browser_gestures import (
    CONFIRMATIONS,
    PHASES,
    SELECTION_TEXT,
    GestureState,
    drop_identity,
    fixture_handler,
    fixture_page,
    prescribed_plan,
    run,
    validate_plan,
)


def event(state: GestureState, phase: str) -> dict:
    details = {
        "double": {"event": "dblclick", "detail": 2, "button": 0, "click_details": [1, 2]},
        "context": {"event": "contextmenu", "button": 2},
        "drop": {"event": "drop", "source": "drag-source", "destination": "drop-area",
                 "drag_nonce": state.nonce, "dragstart_trusted": True},
        "selection": {"event": "select", "focused": True, "start": 0, "end": 5,
                      "selected_text": "Alpha", "value": SELECTION_TEXT},
        "shortcut": {"event": "keydown", "focused": True, "key": "k", "ctrl": True,
                     "alt": False, "shift": False, "meta": False},
        "scroll": {"event": "scroll", "top": 300, "height": 110, "content_height": 1024},
        "finish": {"event": "click", "button": 0},
    }
    return {"nonce": state.nonce, "trusted": True, **details[phase]}


def preceding(state: GestureState, phase: str) -> None:
    for prior in PHASES[:PHASES.index(phase)]:
        assert state.submit(prior, event(state, prior))["accepted"]


def test_exact_server_event_sequence_creates_an_independently_readable_artifact(tmp_path) -> None:
    state = GestureState(tmp_path / "server.json")
    assert not state.verify_artifact()["verified"]
    for phase, confirmation in zip(PHASES, CONFIRMATIONS, strict=True):
        result = state.submit(phase, event(state, phase))
        assert result["accepted"] and result["confirmation"] == confirmation
        if phase != "finish":
            assert not result["marker"] and not state.artifact.exists()
    assert result["marker"] == state.marker
    receipt = state.verify_artifact()
    assert receipt["verified"] and receipt["bytes"] > 0 and len(receipt["sha256"]) == 64
    saved = json.loads(state.artifact.read_bytes())
    assert saved["events"] == state.summary()["events"]
    assert [item["phase"] for item in saved["events"]] == list(PHASES)
    assert [item["sequence"] for item in saved["events"]] == list(range(1, 8))
    saved["events"][2]["event"]["event"] = "mousemove"
    state.artifact.write_text(json.dumps(saved))
    assert not state.verify_artifact()["verified"]


@pytest.mark.parametrize("phase", PHASES[1:])
def test_out_of_order_gestures_never_advance_or_create_a_receipt(phase, tmp_path) -> None:
    state = GestureState(tmp_path / "server.json")
    assert not state.submit(phase, event(state, phase))["accepted"]
    assert not state.completed and not state.artifact.exists()
    assert state.summary()["events"] == []


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("field,value", [("trusted", False), ("nonce", "another-run")])
def test_untrusted_or_replayed_browser_event_cannot_satisfy_a_phase(phase, field, value) -> None:
    state = GestureState()
    preceding(state, phase)
    payload = {**event(state, phase), field: value}
    assert not state.submit(phase, payload)["accepted"]
    assert state.summary()["phase"] == phase


@pytest.mark.parametrize("phase,field,value", [
    ("double", "detail", 1), ("double", "click_details", [1, 1]), ("double", "event", "click"),
    ("context", "event", "click"), ("context", "button", 0),
    ("drop", "event", "mouseup"), ("drop", "destination", "another-area"),
    ("drop", "drag_nonce", "another-drag"), ("drop", "dragstart_trusted", False),
    ("selection", "start", 1), ("selection", "end", 6), ("selection", "selected_text", "bravo"),
    ("selection", "focused", False), ("selection", "value", "Changed text"),
    ("shortcut", "focused", False), ("shortcut", "ctrl", False), ("shortcut", "key", "j"),
    ("shortcut", "alt", True), ("shortcut", "shift", True),
    ("scroll", "top", 0), ("scroll", "top", -1), ("scroll", "top", True),
    ("scroll", "top", float("nan")), ("scroll", "top", 10000), ("scroll", "height", 1024),
    ("finish", "event", "keydown"),
])
def test_dispatch_claims_without_precise_observed_effects_are_rejected(phase, field, value) -> None:
    state = GestureState()
    preceding(state, phase)
    assert not state.submit(phase, {**event(state, phase), field: value})["accepted"]
    assert not state.completed


def test_fixture_preserves_browser_generated_events_and_hides_terminal_marker() -> None:
    state = GestureState()
    page = fixture_page(state)
    assert state.marker not in page
    assert "dispatchEvent" not in page and "new MouseEvent" not in page
    assert "event.isTrusted" in page
    assert "'dblclick'" in page and "click_details:clicks" in page
    assert "'contextmenu'" in page and "button:event.button" in page
    assert "'dragstart'" in page and "'drop'" in page and "event.dataTransfer.getData" in page
    assert "selection.selectionStart" in page and "selection.selectionEnd" in page
    assert "document.activeElement===shortcut" in page
    assert "top:panel.scrollTop" in page and 'id="finish" disabled' in page


def test_plan_uses_observed_destination_and_independent_predicates_within_bound() -> None:
    observation = {"nodes": [{"id": "opaque-e4", "role": "region", "name": "Drop area",
                               "visible": True, "enabled": True}]}
    destination = drop_identity(observation)
    plan = validate_plan(prescribed_plan(destination), destination)
    assert len(plan["steps"]) == 7 <= 25
    assert plan["steps"][2]["args"] == {"to_node_id": "opaque-e4"}
    assert plan["steps"][3]["expect"][1] == {
        "kind": "text_selection", "node_id": "$target", "start": 0, "end": 5}
    assert all(step["expect"][0]["text"] == text for step, text in zip(plan["steps"], CONFIRMATIONS, strict=True))
    plan["steps"][4]["args"]["keys"] = ["Control", "k"]
    plan["steps"][-1]["verb"] = "click"
    assert validate_plan(plan, destination)


@pytest.mark.parametrize("change", ["code", "destination", "order", "range", "shortcut", "effect", "selection-effect", "duplicate", "terminal", "steps", "extra"])
def test_model_plan_cannot_bypass_fixture_verb_args_or_effect_contract(change) -> None:
    plan = copy.deepcopy(prescribed_plan("opaque-e4"))
    if change == "code":
        plan["steps"][0]["args"]["code"] = "arbitrary JavaScript"
    elif change == "destination":
        plan["steps"][2]["args"]["to_node_id"] = "foreign-target"
    elif change == "order":
        plan["steps"][0], plan["steps"][1] = plan["steps"][1], plan["steps"][0]
    elif change == "range":
        plan["steps"][3]["args"]["end"] = 6
    elif change == "shortcut":
        plan["steps"][4]["args"]["keys"] = ["CTRL", "W"]
    elif change == "effect":
        plan["steps"][0]["expect"] = [{"kind": "text", "text": "Ready for gestures"}]
    elif change == "selection-effect":
        plan["steps"][3]["expect"].pop()
    elif change == "duplicate":
        plan["steps"][1]["id"] = plan["steps"][0]["id"]
    elif change == "terminal":
        plan["final_predicates"] = [{"kind": "text", "text": "Scroll verified"}]
    elif change == "steps":
        plan["steps"].pop()
    else:
        plan["script"] = "arbitrary JavaScript"
    with pytest.raises(ComputerUseError):
        validate_plan(plan, "opaque-e4")


@pytest.mark.parametrize("nodes", [[], [{"id": "e4", "role": "region", "name": "Drop area", "visible": True, "enabled": False}]])
def test_destination_must_exist_in_current_eligible_semantic_observation(nodes) -> None:
    with pytest.raises(ComputerUseError):
        drop_identity({"nodes": nodes})


def test_http_server_records_events_and_rejects_malformed_or_wrong_paths(tmp_path) -> None:
    state = GestureState(tmp_path / "server.json")
    server = ThreadingHTTPServer(("127.0.0.1", 0), fixture_handler(state))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}") as client:
            assert client.get("/").status_code == 200
            assert client.get("/status").status_code == 404
            for path, data in [("/event/double", b"bad json"), ("/event/double", b"[]"), ("/wrong", b"{}")]:
                assert client.post(path, content=data).json()["accepted"] is False
            for phase in PHASES:
                assert client.post("/event/" + phase, json=event(state, phase)).json()["accepted"] is True
        assert state.verify_artifact()["verified"]
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


@pytest.mark.parametrize("workflow_error", [False, True])
def test_controller_complete_claim_without_real_server_events_cannot_pass(tmp_path, monkeypatch, workflow_error) -> None:
    real_client = httpx.Client
    nodes = [{"id": "e4", "role": "region", "name": "Drop area", "visible": True, "enabled": True}]

    def fake_portal(request: httpx.Request) -> httpx.Response:
        arguments = json.loads(request.content)["arguments"]
        if request.url.path.endswith("computer_use/call"):
            if arguments["operation"] == "observe":
                result = {"state": {"nodes": nodes, "text": "All gestures verified"}}
            else:
                result = {"status": "complete", "terminal_receipt": {"claim": "complete"},
                          "outcomes": [{"status": "verified"}] * 7}
                if workflow_error:
                    result.update(error="ui_workflow_unresolved", policy_receipts=[{"diagnostic": "retained"}],
                                  point_receipt={"diagnostic": "retained"})
        else:
            result = {"status": "claimed"}
        return httpx.Response(200, json={"result": result})

    monkeypatch.setattr("runtime.verify_browser_gestures.httpx.Client", lambda **kwargs: real_client(
        transport=httpx.MockTransport(fake_portal), **kwargs))
    token = tmp_path / "token"
    token.write_text("a" * 32)
    output = tmp_path / "report.json"
    args = argparse.Namespace(endpoint="http://127.0.0.1:8920", token_file=token,
                              model="test-only", output=output, speech=False)
    assert run(args) == 2
    report = json.loads(output.read_text())
    assert report["status"] == "complete" and report["verified_steps"] == 7
    assert report["passed"] is False and report["authoritative_artifact"]["verified"] is False
    assert report["authoritative_server"]["events"] == []
    assert report["marker_observed"] is False
    if workflow_error:
        assert report["workflow"]["error"] == "ui_workflow_unresolved"
        assert report["workflow"]["policy_receipts"] == [{"diagnostic": "retained"}]
        assert report["workflow"]["point_receipt"] == {"diagnostic": "retained"}
