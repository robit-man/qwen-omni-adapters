from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from runtime.verify_native_computer_use import (
    NativeFixtureState,
    _scope_steps,
    artifact_bytes,
    baseline_plan,
    editor_contract,
    expected_data,
    model_plan,
    tool,
    validate_plan,
    verify_artifact,
)


def test_native_fixture_requires_exact_data_and_order_before_publishing_bytes(tmp_path: Path) -> None:
    fixture = NativeFixtureState(tmp_path, "FRESH-MARKER")
    expected = expected_data()
    assert fixture.advance("Save", expected) == "FIXTURE-REJECTED"
    assert fixture.advance("Preview", expected) == "FIXTURE-REJECTED"
    bad = copy.deepcopy(expected)
    bad["fields"]["Count"] = "4"
    assert fixture.advance("Validate", bad) == "FIXTURE-REJECTED"
    assert fixture.stage == "editing"
    assert not (tmp_path / "native-report.json").exists()
    assert fixture.advance("Validate", expected) == "FRESH-MARKER:validated"
    assert fixture.advance("Preview", expected) == "FRESH-MARKER:previewed"
    assert fixture.advance("Save", bad) == "FIXTURE-REJECTED"
    assert not (tmp_path / "native-report.json").exists()
    assert fixture.advance("Save", expected) == "FRESH-MARKER:saved"
    receipt = verify_artifact(tmp_path, "FRESH-MARKER")
    raw = artifact_bytes(expected)
    assert receipt == {"path": str(tmp_path / "native-report.json"), "sha256": hashlib.sha256(raw).hexdigest(),
                       "size_bytes": len(raw), "verified": True}


def test_successful_native_submission_and_a_forged_receipt_cannot_replace_readback(tmp_path: Path) -> None:
    fixture = NativeFixtureState(tmp_path, "MARKER")
    for operation in ("Validate", "Preview", "Save"):
        fixture.advance(operation, expected_data())
    (tmp_path / "native-report.json").write_bytes(b"wrong but submitted\n")
    receipt = json.loads((tmp_path / "fixture-receipt.json").read_text())
    receipt["sha256"] = hashlib.sha256(b"wrong but submitted\n").hexdigest()
    receipt["size_bytes"] = len(b"wrong but submitted\n")
    (tmp_path / "fixture-receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="Independent native artifact"):
        verify_artifact(tmp_path, "MARKER")


def test_native_baseline_is_sixteen_distinct_typed_actions_with_real_effects() -> None:
    plan = baseline_plan("MARKER")
    assert validate_plan(plan, "MARKER") == plan
    assert len(plan["steps"]) == 16
    assert len({step["id"] for step in plan["steps"]}) == 16
    assert [step["target"]["name"] for step in plan["steps"][-3:]] == ["Validate", "Preview", "Save"]
    assert [step["expect"][0]["text"] for step in plan["steps"][-3:]] == [
        "MARKER:validated", "MARKER:previewed", "MARKER:saved"]


@pytest.mark.parametrize("change", ["foreign", "wrong-value", "missing-effect", "missing-action", "duplicate-id", "wrong-final", "executable"])
def test_model_plan_guard_rejects_actions_outside_exact_fixture_authority(change) -> None:
    plan = baseline_plan("MARKER")
    if change == "foreign":
        plan["steps"][0]["target"]["scope_id"] = "foreign-window"
    elif change == "wrong-value":
        plan["steps"][0]["args"]["value"] = "silently different outcome"
    elif change == "missing-effect":
        plan["steps"][0]["expect"] = [{"kind": "text", "text": "Existing easy text"}]
    elif change == "missing-action":
        plan["steps"].pop()
    elif change == "duplicate-id":
        plan["steps"][1]["id"] = plan["steps"][0]["id"]
    elif change == "wrong-final":
        plan["final_predicates"] = [{"kind": "text", "text": "MARKER:previewed"}]
    else:
        plan["command"] = "arbitrary model code"
    with pytest.raises(ValueError):
        validate_plan(plan, "MARKER")


def test_plan_guard_preserves_order_for_execution_to_verify_instead_of_repairing_model_output() -> None:
    plan = baseline_plan("MARKER")
    plan["steps"][-3:] = reversed(plan["steps"][-3:])
    validated = validate_plan(plan, "MARKER")
    assert validated["steps"][-3]["target"]["name"] == "Save"
    # The application/controller must reject premature Save; validation cannot fake a corrected plan.


def test_model_plan_makes_exactly_one_nonexecuting_request_to_selected_model() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert request.headers["Authorization"] == "Bearer secret-fixture-token"
        assert body["model"] == "explicit/ornith:q4km"
        assert body["think"] is False and body["stream"] is False
        assert body["portal_auto_tools"] is False
        assert "tools" not in body
        return httpx.Response(200, json={"model": body["model"], "message": {
            "role": "assistant", "content": json.dumps(baseline_plan("MARKER"))}})

    with httpx.Client(transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer secret-fixture-token"}) as client:
        plan = model_plan(client, "http://portal.fixture", "explicit/ornith:q4km", "MARKER", None)
    assert plan == baseline_plan("MARKER")
    assert len(calls) == 1


@pytest.mark.parametrize("body", [
    {"model": "different/model", "message": {"content": "{}"}},
    {"model": "selected/model", "message": {"content": "{}", "tool_calls": [{"function": {"name": "shell"}}]}},
    {"model": "selected/model", "message": {"content": "not JSON"}},
])
def test_planner_identity_and_no_tool_execution_are_required(body) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=body))) as client, pytest.raises(ValueError):
        model_plan(client, "http://portal.fixture", "selected/model", "MARKER", None)


def test_native_tool_calls_use_the_authenticated_desktop_surface() -> None:
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        assert request.url.path == "/api/tools/computer_use/call"
        return httpx.Response(200, json={"result": {"status": "complete"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert tool(client, "http://portal.fixture", {"operation": "workflow", "steps": []})["status"] == "complete"
    assert seen == [{"arguments": {"surface": "desktop", "operation": "workflow", "steps": []}}]


def test_failed_native_workflow_preserves_the_actual_dispatch_and_verification_evidence() -> None:
    failure = {"error": "ui_workflow_unresolved", "status": "unresolved",
               "outcomes": [{"status": "unresolved", "reason": "effect_not_observed",
                             "dispatch": {"status": "dispatched"}}]}
    with httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"result": failure}))) as client:
        assert tool(client, "http://portal.fixture", {"operation": "workflow"}) == failure


def test_native_plan_scope_is_host_pinned_without_mutating_model_plan() -> None:
    plan = baseline_plan("MARKER")
    scoped = _scope_steps(plan["steps"], {"scope": {"id": "owned-window"}})
    assert all(step["target"]["scope_id"] == "owned-window" for step in scoped)
    assert all("scope_id" not in step["target"] for step in plan["steps"])


def test_gedit_save_targets_semantic_window_despite_modified_title() -> None:
    state: dict[str, Any] = {"nodes": [
        {"role": "textbox", "name": "Document", "operations": ["set_value"], "sensitive": False, "visible": True, "enabled": True},
        {"role": "window", "name": "editor-note.txt — Text Editor", "operations": ["key_chord"], "sensitive": False, "visible": True, "enabled": True},
    ]}
    editor = editor_contract(state, "Exact note.\n")
    assert editor["steps"][1]["target"] == {"role": "window"}
    assert editor["steps"][1]["expect"] == [{"kind": "visible", "role": "window", "name": "editor-note.txt — Text Editor", "visible": True}]
    plan = {**baseline_plan("MARKER"), "editor_steps": copy.deepcopy(editor["steps"])}
    assert validate_plan(plan, "MARKER", editor) == plan
    plan["editor_steps"][1]["args"] = {"keys": ["ctrl", "q"]}
    with pytest.raises(ValueError, match="Editor actions"):
        validate_plan(plan, "MARKER", editor)


def test_gedit_requires_a_unique_unprotected_editor() -> None:
    with pytest.raises(RuntimeError, match="unique editable"):
        editor_contract({"nodes": []}, "Exact note.\n")


def test_gedit_hidden_search_editor_does_not_make_the_visible_document_ambiguous() -> None:
    state = {"nodes": [
        {"role": "textbox", "name": "", "operations": ["set_value", "focus"], "sensitive": False, "visible": True, "enabled": True},
        {"role": "textbox", "name": "Search", "operations": ["invoke", "set_value", "focus"], "sensitive": False, "visible": False, "enabled": True},
        {"role": "window", "name": "Owned note - gedit", "operations": ["key_chord"], "sensitive": False, "visible": True, "enabled": True},
    ]}
    editor = editor_contract(state, "Exact note.\n")
    assert editor["steps"][0]["target"] == {"role": "textbox", "name": ""}
    state["nodes"][0]["enabled"] = False
    with pytest.raises(RuntimeError, match="unique editable"):
        editor_contract(state, "Exact note.\n")
