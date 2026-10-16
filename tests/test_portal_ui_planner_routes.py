"""Real portal routes and controller with CPU-only transport/input doubles."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import httpx
import pytest

import portal.tools as tools_module
from portal.app import (
    DEFAULT_MODEL,
    SESSION_COOKIE_NAME,
    PortalConfig,
    _model_tool_result,
    create_app,
)
from portal.computer_use import ComputerUseStore
from portal.tools import PortalToolHarness, tool_schemas

TOKEN = "ui-planner-route-test-token-at-least-24-characters"
HEADERS = {"Authorization": "Bearer " + TOKEN}


def _config(tmp_path):
    return PortalConfig(adapter_url="http://adapter/api/chat", adapter_health_url="http://adapter/healthz",
        comprehension_health_url="http://comprehension/health", tts_health_url="http://tts/healthz",
        ollama_health_url="http://ollama/api/tags", model=DEFAULT_MODEL, access_token=TOKEN,
        timeout_s=30, max_body_bytes=1024 * 1024, session_log_dir=tmp_path / "sessions")


class Plane:
    shadow_mode = False

    def __init__(self, name):
        self.config = {"tool_families": {"desktop": [name]}}

    def evaluate(self, **kwargs):
        return SimpleNamespace(results={"tool_family": SimpleNamespace(
            value="desktop", confidence=1.0, latency_ms=0.1, band="accepted", source="test",
            checkpoint="test", shadow=False, escalated=False, fast_path_taken=True)})


class Adapter:
    def __init__(self, session):
        self.session, self.value, self.saved = session, "", ""
        self.calls = []

    def observe(self):
        nodes = [{"id": "document", "role": "textbox", "name": "Document", "operations": ["set_value"], "value": self.value},
                 {"id": "save", "role": "button", "name": "Save", "operations": ["invoke"]},
                 {"id": "saved", "role": "label", "name": "Saved state", "operations": [], "value": self.saved}]
        return {"session_id": self.session, "revision": self.value + ":" + self.saved,
                "scope": {"id": "owned-document", "kind": "desktop"},
                "nodes": [{"generation": "1", "scope_id": "owned-document", "visible": True,
                           "enabled": True, "sensitive": False, **node} for node in nodes]}

    def perform(self, action):
        self.calls.append(copy.deepcopy(action))
        if action["verb"] == "set_value":
            self.value = action["args"]["value"]
        else:
            self.saved = self.value
        return {"status": "dispatched"}


def _step():
    return {"id": "write", "verb": "set_value", "target": {"role": "textbox", "name": "Document"},
            "args": {"value": "Owned note"},
            "expect": [{"kind": "value", "node_id": "$target", "value": "Owned note"}]}


def _workflow(name):
    return {"operation": "workflow", **({"surface": "desktop"} if name == "computer_use" else {}),
            "steps": [_step(), {"id": "save", "verb": "invoke", "target": {"role": "button", "name": "Save"},
                                  "expect": [{"kind": "value", "node_id": "saved", "value": "Owned note"}]}],
            "final_predicates": [{"kind": "value", "node_id": "saved", "value": "Owned note"}]}


def _call(name, args, call_id):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


def _answer(stream, calls=()):
    body = {"model": DEFAULT_MODEL, "message": {"role": "assistant", "content": "" if calls else "Phase finished.",
            **({"tool_calls": list(calls)} if calls else {})}}
    return (httpx.Response(200, content=json.dumps({"type": "final", "response": body}) + "\n",
                           headers={"content-type": "application/x-ndjson"}) if stream else httpx.Response(200, json=body))


def _request(stream, name):
    return {"model": DEFAULT_MODEL, "stream": stream, "think": False,
            "messages": [{"role": "system", "content": "Keep this host policy."},
                         {"role": "user", "content": "Write and save the owned note."}],
            "omni": {"schema": "robit.ollama.omni-adapter.v1", "task": "chat"},
            "response_modalities": ["text"], "speech_mode": "never", "portal_auto_tools": True,
            "portal_tool_allowlist": [name], "tool_choice": "auto"}


def _post(client, body, stream):
    response = client.post("/api/chat/stream" if stream else "/api/chat", json=body, headers=HEADERS)
    assert response.status_code == 200, response.data
    if not stream:
        return response.json
    events = [json.loads(line) for line in response.data.splitlines() if line]
    return next(event["response"] for event in reversed(events) if event.get("type") == "final")


def _assert_profile(payload, name):
    function = next(tool["function"] for tool in payload["tools"] if tool["function"]["name"] == name)
    parameters = function["parameters"]
    assert set(parameters["properties"]["operation"]["enum"]) == {"observe", "workflow", "verify", "cancel", "inspect_visual"}
    assert not {"action_id", "step"} & parameters["properties"].keys()
    assert {branch["properties"]["operation"]["const"] for branch in parameters["oneOf"]} == {"observe", "workflow", "verify", "cancel", "inspect_visual"}
    assert payload["think"] is False
    assert any("Keep this host policy." in message.get("content", "") for message in payload["messages"] if message["role"] == "system")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("name", ["native_ui", "computer_use"])
def test_automatic_phase_executes_two_inputs_between_model_requests_and_preserves_raw_trace(tmp_path, monkeypatch, stream, name):
    adapters, requests, executions, raw_results = [], [], [], []

    def factory(session, surface):
        assert surface == "desktop"
        adapter = Adapter(session)
        adapters.append(adapter)
        return adapter

    store = ComputerUseStore(None, adapter_factory=factory)
    monkeypatch.setattr(tools_module, "ComputerUseStore", lambda *args, **kwargs: store)
    original_execute = PortalToolHarness.execute

    def execute(self, session, actual_name, arguments):
        result = original_execute(self, session, actual_name, arguments)
        executions.append(copy.deepcopy(arguments))
        raw_results.append(copy.deepcopy(result))
        return result

    monkeypatch.setattr(PortalToolHarness, "execute", execute)

    def upstream(request):
        payload = json.loads(request.content)
        requests.append(payload)
        _assert_profile(payload, name)
        if len(requests) == 1:
            assert not adapters
            return _answer(stream, [_call(name, {"operation": "observe", **({"surface": "desktop"} if name == "computer_use" else {})}, "inspect")])
        if len(requests) == 2:
            assert adapters[0].calls == []
            return _answer(stream, [_call(name, _workflow(name), "phase")])
        assert len(requests) == 3
        assert [action["verb"] for action in adapters[0].calls] == ["set_value", "invoke"]
        assert [action["revision"] for action in adapters[0].calls] == [":", "Owned note:"]
        result = json.loads(next(message["content"] for message in reversed(payload["messages"]) if message["role"] == "tool"))
        assert result["phase_handoff"]["terminal_verified"] is True
        assert result["phase_handoff"]["goal_verified"] is False
        return _answer(stream)

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)), decision_plane=Plane(name))
    final = _post(app.test_client(), _request(stream, name), stream)
    assert len(requests) == 3 and len(executions) == 2 and len(adapters[0].calls) == 2
    traces = final["portal"]["safe_tools_executed"]
    assert [trace["id"] for trace in traces] == ["inspect", "phase"]
    assert [trace["result"] for trace in traces] == [_model_tool_result(result, tool_name=name)[0] for result in raw_results]
    assert "nodes" in json.loads(traces[0]["result"])["state"]  # Raw audit never retires its map.


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("name", ["native_ui", "computer_use"])
@pytest.mark.parametrize("arguments", [
    {"operation": "execute", "action_id": "genuine-host-offer"},
    {"operation": "observe", "step": _step()},
    {"operation": ["execute"], "action_id": "genuine-host-offer"},
])
def test_automatic_per_action_and_malformed_calls_rejected_before_harness(tmp_path, monkeypatch, stream, name, arguments):
    requests = []

    def execute(*args, **kwargs):
        pytest.fail("A model per-action call must never reach the harness")

    def upstream(request):
        payload = json.loads(request.content)
        requests.append(payload)
        _assert_profile(payload, name)
        if len(requests) == 1:
            return _answer(stream, [_call(name, arguments, "blocked")])
        assert len(requests) == 2
        result = json.loads(next(message["content"] for message in payload["messages"] if message["role"] == "tool"))
        assert result["error"] == "ui_planner_phase_required"
        assert result["dispatch_status"] == "not_dispatched"
        return _answer(stream)

    monkeypatch.setattr(PortalToolHarness, "execute", execute)
    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)), decision_plane=Plane(name))
    final = _post(app.test_client(), _request(stream, name), stream)
    assert len(requests) == 2
    assert json.loads(final["portal"]["safe_tools_executed"][0]["result"])["dispatch_status"] == "not_dispatched"


@pytest.mark.parametrize("stream", [False, True])
def test_even_genuine_session_bound_action_is_not_dispatched_by_automatic_planner(tmp_path, monkeypatch, stream):
    adapters, executions, requests = [], [], []

    def factory(session, surface):
        adapter = Adapter(session)
        adapters.append(adapter)
        return adapter

    store = ComputerUseStore(None, adapter_factory=factory)
    monkeypatch.setattr(tools_module, "ComputerUseStore", lambda *args, **kwargs: store)
    original_execute = PortalToolHarness.execute

    def execute(self, session, name, arguments):
        executions.append(copy.deepcopy(arguments))
        return original_execute(self, session, name, arguments)

    monkeypatch.setattr(PortalToolHarness, "execute", execute)
    offer = None

    def upstream(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return _answer(stream, [_call("native_ui", {"operation": "execute", "action_id": offer}, "genuine-but-blocked")])
        assert len(requests) == 2
        return _answer(stream)

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)), decision_plane=Plane("native_ui"))
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, "owned-same-controller-session-12345")
    response = client.post("/api/tools/native_ui/call", json={"operation": "observe", "step": _step()}, headers=HEADERS)
    offer = response.json["result"]["selected_action_id"]
    assert offer and response.json["result"]["selected_kind"] == "action"
    final = _post(client, _request(stream, "native_ui"), stream)
    assert executions == [{"operation": "observe", "step": _step()}]
    assert adapters[0].calls == []
    assert json.loads(final["portal"]["safe_tools_executed"][0]["result"])["error"] == "ui_planner_phase_required"


def test_direct_tool_client_retains_genuine_observe_step_execute_protocol(tmp_path, monkeypatch):
    adapters = []

    def factory(session, surface):
        adapter = Adapter(session)
        adapters.append(adapter)
        return adapter

    store = ComputerUseStore(None, adapter_factory=factory)
    monkeypatch.setattr(tools_module, "ComputerUseStore", lambda *args, **kwargs: store)

    def no_inference(request):
        pytest.fail("Direct controller clients do not request language inference")

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(no_inference)))
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, "owned-direct-client-session-12345")
    public = client.get("/api/tools", headers=HEADERS).json["tools"]
    assert next(item for item in public if item["function"]["name"] == "native_ui") == tool_schemas(["native_ui"])[0]
    observation = client.post("/api/tools/native_ui/call", headers=HEADERS,
                              json={"operation": "observe", "step": _step()})
    assert observation.status_code == 200
    offered = observation.json["result"]["selected_action_id"]
    executed = client.post("/api/tools/native_ui/call", headers=HEADERS,
                           json={"operation": "execute", "action_id": offered})
    assert executed.status_code == 200 and executed.json["result"]["status"] == "verified"
    assert len(adapters[0].calls) == 1
