from __future__ import annotations

import base64
import io
import json
import wave

import httpx
import pytest

from portal.app import DEFAULT_MODEL, PortalConfig, _pop_tool_allowlist, create_app
from portal.tools import PortalToolHarness

TOKEN = "scoped-portal-test-token-with-24-characters"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def _config(tmp_path):
    return PortalConfig(adapter_url="http://adapter/api/chat", adapter_health_url="http://adapter/healthz",
                        comprehension_health_url="http://comprehension/health", tts_health_url="http://tts/healthz",
                        ollama_health_url="http://ollama/api/tags", model=DEFAULT_MODEL, access_token=TOKEN,
                        timeout_s=30, max_body_bytes=1024 * 1024, session_log_dir=tmp_path / "sessions")


def _request(stream, *, media=False, **fields):
    message = {"role": "user", "content": "Inspect the owned desktop."}
    if media:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\0\0" * 1600)
        message["audios"] = [{"mime_type": "audio/wav", "encoding": "base64",
                              "data": base64.b64encode(buffer.getvalue()).decode()}]
    return {"model": DEFAULT_MODEL, "stream": stream, "think": False, "messages": [message],
            "omni": {"schema": "robit.ollama.omni-adapter.v1", "task": "chat"},
            "response_modalities": ["text"], "speech_mode": "never", "portal_auto_tools": True, **fields}


def _answer(stream, calls=(), **fields):
    body = {"model": DEFAULT_MODEL, "message": {"role": "assistant", "content": "" if calls else "Complete.",
             **({"tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}
                                for name, arguments in calls]} if calls else {})}, **fields}
    if stream:
        return httpx.Response(200, content=json.dumps({"type": "final", "response": body}) + "\n",
                              headers={"Content-Type": "application/x-ndjson"})
    return httpx.Response(200, json=body)


def _result(response, stream):
    if not stream:
        return response.json
    events = [json.loads(line) for line in response.data.splitlines() if line]
    return next(item["response"] for item in reversed(events) if item.get("type") == "final")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("scope", [None, "native_ui", {}, [True], [{}], ["unknown"],
                                   ["NATIVE_UI"], ["native_ui", "native_ui"]])
def test_malformed_request_scope_is_rejected_before_upstream(tmp_path, stream, scope):
    def upstream(request):
        pytest.fail("Invalid scope reached the adapter")

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    response = app.test_client().post("/api/chat/stream" if stream else "/api/chat", headers=HEADERS,
                                      json=_request(stream, portal_tool_allowlist=scope))
    assert response.status_code == 400
    assert "portal_tool_allowlist" in response.json["error"]


def test_scope_is_immutable_and_always_retains_discovery():
    names = ["native_ui"]
    payload = {"portal_tool_allowlist": names}
    scope = _pop_tool_allowlist(payload)
    names.append("shell")
    assert scope == frozenset({"native_ui", "tool_search"})
    assert payload == {}
    assert _pop_tool_allowlist({"portal_tool_allowlist": []}) == frozenset({"tool_search"})
    assert _pop_tool_allowlist({}) is None


@pytest.mark.parametrize("stream,detached", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("media", [False, True])
def test_scope_survives_all_generations_and_filters_discovery_and_forged_calls(
    tmp_path, monkeypatch, stream, detached, media,
):
    requests, executions = [], []
    original_execute = PortalToolHarness.execute
    forbidden = {"shell", "workspace_file", "web_fetch", "browser_interact", "background_task", "subagent_delegate"}
    scope = {"native_ui", "memory_write", "memory_search", "tool_search"}

    def execute(self, session, name, arguments):
        assert name not in forbidden, "Forbidden model call reached the host executor"
        executions.append(name)
        if name == "native_ui":
            return {"status": "verified", "evidence_authority": "inspection"}
        return original_execute(self, session, name, arguments)

    monkeypatch.setattr(PortalToolHarness, "execute", execute)

    def upstream(request):
        body = json.loads(request.content)
        requests.append(body)
        assert "portal_tool_allowlist" not in body
        names = {item["function"]["name"] for item in body.get("tools", [])}
        assert names <= scope
        index = len(requests)
        if index == 1:
            if media:
                assert "native_ui" in names and "memory_write" in names
            return _answer(stream, [("tool_search", {"family": "desktop"})])
        if index == 2:
            assert names == {"native_ui"}
            return _answer(stream, [(name, {}) for name in sorted(forbidden)],
                           portal_tool_allowlist=["shell"])
        if index == 3:
            return _answer(stream, [("native_ui", {"operation": "observe"}),
                                   ("memory_write", {"key": "scoped", "value": "Exact evidence"})])
        if index == 4:
            return _answer(stream, [("tool_search", {"family": "uncertain"})])
        if index == 5:
            return _answer(stream, [("memory_search", {"query": "Exact evidence"})])
        assert index == 6
        return _answer(stream)

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    body = _request(stream, media=media, portal_tool_allowlist=["native_ui", "memory_write", "memory_search"])
    if detached:
        body.update(portal_detached_turn=True, portal_turn_id="scoped-detached-12345")
    response = app.test_client().post("/api/chat/stream" if stream else "/api/chat", headers=HEADERS, json=body)
    assert response.status_code == 200
    result = _result(response, stream)
    assert result["portal"]["tool_allowlist"] == sorted(scope)
    assert len(requests) == 6
    assert set(executions) == scope
    traces = result["portal"]["safe_tools_executed"]
    denied = [json.loads(item["result"]) for item in traces if item["name"] in forbidden]
    assert len(denied) == len(forbidden)
    assert all(item["error"] == "tool_not_allowed_by_request"
               and item["dispatch_status"] == "not_dispatched" for item in denied)
    desktop = next(json.loads(item["result"]) for item in traces
                   if item["name"] == "tool_search" and item["arguments"]["family"] == "desktop")
    assert desktop["available_tools"] == ["native_ui"]
    families = next(json.loads(item["result"]) for item in traces
                    if item["name"] == "tool_search" and item["arguments"]["family"] == "uncertain")
    assert {family["name"] for family in families["available_families"]} == {"desktop", "memory"}


@pytest.mark.parametrize("stream", [False, True])
def test_empty_scope_rejects_leaf_calls_and_does_not_leak_into_next_request(tmp_path, monkeypatch, stream):
    calls, executed = [], []

    def execute(self, session, name, arguments):
        executed.append(name)
        return {"stdout": "Test executor marker", "exit_code": 0}

    monkeypatch.setattr(PortalToolHarness, "execute", execute)

    def upstream(request):
        calls.append(json.loads(request.content))
        return _answer(stream, [("shell", {"intent": "inspect", "command": "never actually run"})]) if len(calls) % 2 else _answer(stream)

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    client = app.test_client()
    endpoint = "/api/chat/stream" if stream else "/api/chat"
    first = client.post(endpoint, headers=HEADERS, json=_request(stream, portal_tool_allowlist=[]))
    assert first.status_code == 200
    assert _result(first, stream)["portal"]["tool_allowlist"] == ["tool_search"]
    assert executed == []
    second = client.post(endpoint, headers=HEADERS, json=_request(stream))
    assert second.status_code == 200
    assert "tool_allowlist" not in _result(second, stream)["portal"]
    assert executed == ["shell"]


@pytest.mark.parametrize("stream", [False, True])
def test_request_scope_cannot_lift_existing_execution_profile_denial(tmp_path, monkeypatch, stream):
    calls = []

    def execute(self, session, name, arguments):
        pytest.fail("An existing denied shell profile reached the executor")

    monkeypatch.setattr(PortalToolHarness, "execute", execute)

    def upstream(request):
        calls.append(json.loads(request.content))
        return _answer(stream, [("shell", {"intent": "inspect", "command": "never run"})]) if len(calls) == 1 else _answer(stream)

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    response = app.test_client().post("/api/chat/stream" if stream else "/api/chat", headers=HEADERS,
        json=_request(stream, portal_tool_allowlist=["shell", "background_task"],
                      portal_background_bridge=True, portal_shell_bridge=False))
    assert response.status_code == 200
    trace = _result(response, stream)["portal"]["safe_tools_executed"]
    assert len(trace) == 1
    assert json.loads(trace[0]["result"])["error"] == "tool_not_available_in_this_execution_profile"


@pytest.mark.parametrize("stream", [False, True])
def test_capability_discovery_reports_only_applied_scope(tmp_path, stream):
    calls = []

    def upstream(request):
        calls.append(json.loads(request.content))
        return _answer(stream, [("get_portal_capabilities", {})]) if len(calls) == 1 else _answer(stream)

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    response = app.test_client().post("/api/chat/stream" if stream else "/api/chat", headers=HEADERS,
        json=_request(stream, portal_tool_allowlist=["native_ui", "get_portal_capabilities"]))
    assert response.status_code == 200
    trace = _result(response, stream)["portal"]["safe_tools_executed"]
    assert set(json.loads(trace[0]["result"])["safe_tools"]) == {"native_ui", "get_portal_capabilities", "tool_search"}
