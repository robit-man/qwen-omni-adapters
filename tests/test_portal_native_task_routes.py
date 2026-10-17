"""Native-task portal integration with CPU transport and guarded adapter doubles."""
from __future__ import annotations

import copy
import json

import httpx
import pytest
from test_ui_task import NoEffectAdapter, no_effect_phase

import portal.app as app_module
import portal.tools as tools_module
from portal.app import DEFAULT_MODEL, SESSION_COOKIE_NAME, PortalConfig, create_app
from portal.computer_use import ComputerUseStore
from portal.native_ui import NativeUiAdapter
from portal.ui_task import NativePhaseCompilationError, phase_response_schema

TOKEN = "native-task-route-token-at-least-24-characters"
HEADERS = {"Authorization": "Bearer " + TOKEN}
GOAL = "Use the actual owned editor.\nWrite and commit the reference note, then report any unmet criteria."


def _config(tmp_path):
    return PortalConfig(adapter_url="http://adapter/api/chat", adapter_health_url="http://adapter/healthz",
        comprehension_health_url="http://comprehension/health", tts_health_url="http://tts/healthz",
        ollama_health_url="http://ollama/api/tags", model=DEFAULT_MODEL, access_token=TOKEN,
        timeout_s=30, max_body_bytes=1024 * 1024, session_log_dir=tmp_path / "sessions")


def _request(stream, think=False):
    return {"model": DEFAULT_MODEL, "stream": stream, "think": think,
            "messages": [{"role": "system", "content": "Preserve this exact caller policy."},
                         {"role": "user", "content": "Earlier user context."},
                         {"role": "assistant", "content": "Earlier reply."},
                         {"role": "user", "content": GOAL}],
            "omni": {"schema": "robit.ollama.omni-adapter.v1", "task": "chat"},
            "response_modalities": ["text"], "speech_mode": "never",
            "portal_auto_tools": True, "portal_execution_profile": "native_task",
            "portal_tool_allowlist": ["native_ui"],
            "options": {"num_predict": 437, "temperature": 0.1, "seed": 73}}


def _phase():
    return {"kind": "ui", "steps": [
        {"id": "write", "verb": "set_value", "target": {"role": "textbox", "name": "Draft"},
         "args": {"value": "Reference note"}, "expect": [{"kind": "value", "node_id": "$target", "value": "Reference note"}]},
        {"id": "commit", "verb": "invoke", "target": {"role": "button", "name": "Commit"},
         "expect": [{"kind": "value", "node_id": "receipt", "value": "Reference note"}]}],
        "final_predicates": [{"kind": "value", "node_id": "receipt", "value": "Reference note"}]}


def _working_reply(request, plan):
    """Format test-authored replies against the actual private request envelope."""
    packet = request if isinstance(request, dict) else json.loads(json.loads(request.content)["messages"][1]["content"])
    state = packet["working_state"]
    decision = ("blocked" if plan["kind"] == "blocked" else "verify" if plan["kind"] == "complete" else
                "start" if state["source_phase"] == 1 else "recover" if state["recovery_mode"] else "continue")
    objective = plan.get("objective", plan.get("subgoal", {}).get("objective", "Write and commit the reference note"))
    return {"assessment": {"source_epoch": state["source_epoch"], "source_phase": state["source_phase"],
                            "decision": decision, "summary": "Assess the latest host result; this is a fixture claim."},
            "commitment": {**({} if "objective" in plan or plan["kind"] == "ui_subgoal" else {"objective": objective}),
                           "intended_resource": "owned fixture resource", "unmet_constraints": []},
            "phase": copy.deepcopy(plan)}


class Adapter:
    def __init__(self, session):
        self.session, self.value, self.saved = session, "", ""
        self.calls, self.planning_calls, self.ordinary_calls = [], 0, 0

    def _state(self, include_windows=False):
        nodes = [{"id": "draft", "role": "textbox", "name": "Draft", "value": self.value, "operations": ["set_value"]},
                 {"id": "commit", "role": "button", "name": "Commit", "operations": ["invoke"]},
                 {"id": "receipt", "role": "label", "name": "Receipt", "value": self.saved, "operations": []}]
        nodes = [{"generation": "1", "scope_id": "form", "visible": True, "enabled": True,
                  "sensitive": False, **node} for node in nodes]
        if include_windows:
            nodes.append({"id": "other-window", "generation": "1", "scope_id": "other-window", "role": "window",
                          "name": "Already-open owned application", "visible": True, "enabled": True,
                          "sensitive": False, "operations": ["switch_window"]})
        return {"session_id": self.session, "revision": self.value + ":" + self.saved,
                "scope": {"id": "form", "kind": "desktop", "window_id": "43001"}, "nodes": nodes}

    def observe(self):
        self.ordinary_calls += 1
        return self._state()

    def planning_observe(self):
        self.planning_calls += 1
        return self._state(True)

    def perform(self, action):
        self.calls.append(copy.deepcopy(action))
        if action["verb"] == "set_value":
            self.value = action["args"]["value"]
        else:
            self.saved = self.value
        return {"status": "dispatched"}


def _install_store(monkeypatch):
    adapters = []

    def factory(session, surface):
        assert surface == "desktop"
        adapter = Adapter(session)
        adapters.append(adapter)
        return adapter

    store = ComputerUseStore(None, adapter_factory=factory)
    monkeypatch.setattr(tools_module, "ComputerUseStore", lambda *args, **kwargs: store)
    return store, adapters


def _post(client, body, stream):
    response = client.post("/api/chat/stream" if stream else "/api/chat", json=body, headers=HEADERS)
    assert response.status_code == 200, response.data
    if stream:
        events = [json.loads(line) for line in response.data.splitlines() if line]
        result = next(event["response"] for event in reversed(events) if event.get("type") == "final")
    else:
        events, result = [], response.json
    activity = client.get("/api/activity", headers=HEADERS).json
    assert activity["active"] == activity["inflight"] == activity["queued"] == 0
    return result, events


@pytest.mark.parametrize("stream", [False, True])
def test_native_text_knowledge_reaches_real_compiler_and_phase_feedback(tmp_path, monkeypatch, stream):
    _install_store(monkeypatch)
    original_state = Adapter._state

    def state_with_knowledge(self, include_windows=False):
        state = original_state(self, include_windows)
        state["nodes"][0]["text_observation"] = {
            "character_count": len(self.value), "selection_count": 0,
            "captured_range": {"start": 0, "end": len(self.value)}, "truncated": False}
        return state

    monkeypatch.setattr(Adapter, "_state", state_with_knowledge)
    packets = []

    def upstream(request):
        payload = json.loads(request.content)
        packet = json.loads(payload["messages"][1]["content"])
        packets.append(packet)
        assert packet["goal"] == GOAL
        assert not {"text_observation", "selection_count", "character_count"} & payload.keys()
        draft = next(node for node in packet["current_state"]["state"]["nodes"] if node["id"] == "draft")
        assert draft["text_observation"]["selection_count"] == 0
        assert draft["text_observation"]["captured_range"]["end"] == len(draft["value"])
        if len(packets) == 1:
            plan = _phase()
        else:
            effect = packet["last_result"]["outcomes"][0]["observed_effects"][0]
            assert effect["text_observation"]["character_count"] == len("Reference note")
            assert effect["text_observation"]["selection_count"] == 0
            plan = {"kind": "complete", "message": "Fixture claim only"}
        return httpx.Response(200, json={"model": DEFAULT_MODEL,
            "message": {"role": "assistant", "content": json.dumps(_working_reply(request, plan))}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _ = _post(app.test_client(), _request(stream), stream)
    assert len(packets) == 2
    assert result["portal"]["native_task"]["task_verified"] is False


@pytest.mark.parametrize("stream", [False, True])
def test_metadata_only_changes_do_not_renew_zero_effect_budget_through_actual_route(tmp_path, monkeypatch, stream):
    packets, adapters = [], []

    def factory(session, surface):
        assert surface == "desktop"
        adapter = NoEffectAdapter()
        observe = adapter.observe

        def clarified():
            state = observe()
            state["session_id"] = session
            state["nodes"][0].update(value="a" * 18, text_observation={
                "character_count": 18 + len(packets), "selection_count": 1,
                "captured_range": {"start": 0, "end": 18}, "truncated": bool(packets)})
            return state

        adapter.observe = adapter.planning_observe = clarified
        adapters.append(adapter)
        return adapter

    store = ComputerUseStore(None, adapter_factory=factory)
    monkeypatch.setattr(tools_module, "ComputerUseStore", lambda *_args, **_kwargs: store)

    def upstream(request):
        packet = json.loads(json.loads(request.content)["messages"][1]["content"])
        packets.append(packet)
        current = packet["current_state"]
        draft = next(node for node in current["state"]["nodes"] if node["id"] == "draft")
        assert draft["text_observation"]["character_count"] == 17 + len(packets)
        plan = no_effect_phase() if len(packets) <= 4 else {"kind": "complete", "message": "Unverified claim"}
        return httpx.Response(200, json={"message": {"content": json.dumps(_working_reply(request, plan))}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _ = _post(app.test_client(), _request(stream), stream)
    assert result["portal"]["native_task"]["status"] == "blocked"
    assert len(packets) == 4 and len(adapters[0].calls) == 2
    for packet in packets:
        current = packet["current_state"]
        assert current["content_sha256"] == current["state"]["model_projection"]["raw_source_sha256"]
    feedback = packets[2]["last_result"]
    assert feedback["error"] == "ui_phase_repeated_without_new_effect"
    assert feedback["phase_handoff"]["new_effect_count"] == 0
    assert [item["effect_receipt"]["effect_state"] for item in feedback["outcomes"]] == ["absent", "replayed", "replayed"]
    assert all(item["status"] == "verified" for item in feedback["outcomes"])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("think", [False, True])
def test_native_profile_compiles_strict_phase_and_executes_two_inputs_internally(tmp_path, monkeypatch, stream, think):
    store, adapters = _install_store(monkeypatch)
    requests = []
    body = _request(stream, think)

    def upstream(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert str(request.url) == "http://adapter/api/chat"  # Compiler uses one nonstream response, on either portal route.
        assert payload["stream"] is False and payload["think"] is think
        assert payload["options"] == body["options"]
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["schema"]["oneOf"]
        assert not {"tools", "tool_choice", "format", "portal_execution_profile", "portal_auto_tools", "portal_tool_allowlist"} & payload.keys()
        assert "Preserve this exact caller policy." in payload["messages"][0]["content"]
        packet = json.loads(payload["messages"][1]["content"])
        assert packet["goal"] == GOAL
        assert packet["response_schema"] == payload["response_format"]["json_schema"]["schema"]
        predicates = {item["properties"]["kind"]["const"]: item for item in
                      packet["response_schema"]["$defs"]["predicate"]["oneOf"]}
        canonical = {item["properties"]["kind"]["const"]: item for item in
                     phase_response_schema(["native_ui"])["$defs"]["predicate"]["oneOf"]}
        for kind in ("focused", "active_window"):
            assert predicates[kind]["description"] == canonical[kind]["description"]
            assert predicates[kind]["description"].strip()
        assert any(node["id"] == "other-window" for node in packet["current_state"]["state"]["nodes"])
        if len(requests) == 1:
            assert adapters[0].planning_calls == 1 and adapters[0].calls == []
            plan = _phase()
        else:
            assert len(requests) == 2
            assert [action["verb"] for action in adapters[0].calls] == ["set_value", "invoke"]
            assert packet["last_result"]["phase_handoff"]["new_effect_count"] == 2
            assert not any("before" in outcome or "after" in outcome for outcome in packet["last_result"]["outcomes"])
            plan = {"kind": "complete", "message": "Planner completion claim."}
        return httpx.Response(200, json={"model": DEFAULT_MODEL, "message": {"role": "assistant", "content": json.dumps(_working_reply(request, plan))}})

    class NoDiscovery:
        shadow_mode = False
        config = {}

        def evaluate(self, **kwargs):
            pytest.fail("Mandatory native planner must not enter generic family discovery")

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)), decision_plane=NoDiscovery())
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, "native-phase-owned-session-12345")
    result, events = _post(client, body, stream)
    assert len(requests) == 2 and adapters[0].planning_calls == 2
    assert adapters[0].ordinary_calls >= 5
    assert result["message"]["content"] == "Planner completion claim."
    assert result["portal"]["native_task"]["status"] == "planner_claim"
    assert result["portal"]["native_task"]["task_verified"] is False
    assert result["portal"]["tool_allowlist"] == ["native_ui"]
    traces = result["portal"]["safe_tools_executed"]
    assert len(traces) == 1 and traces[0]["name"] == "native_ui"
    raw = json.loads(traces[0]["result"])
    assert len(raw["outcomes"]) == 2 and raw["terminal_receipt"]["status"] == "verified"
    assert all("before" in outcome and "after" in outcome for outcome in raw["outcomes"])
    assert store._owned("native-phase-owned-session-12345", "desktop").cancel.is_set() is False
    if stream:
        assert [event["phase"] for event in events if event["type"] == "tool"] == ["start", "complete"]


@pytest.mark.parametrize("stream", [False, True])
def test_native_profile_session_memory_uses_only_explicit_scope(tmp_path, monkeypatch, stream):
    _install_store(monkeypatch)
    plans = iter([{"kind": "memory_write", "topic": "reference", "key": "owned", "value": "Source-bound note"},
                  {"kind": "memory_search", "query": "Source-bound note", "max_results": 1},
                  {"kind": "complete", "message": "Memory claim."}])
    requests = []

    def upstream(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 3:
            packet = json.loads(payload["messages"][1]["content"])
            assert packet["last_result"]["results"][0]["value"] == "Source-bound note"
        return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps(_working_reply(request, next(plans)))}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, "native-memory-owned-session-12345")
    body = _request(stream)
    body["portal_tool_allowlist"] = ["native_ui", "memory_write", "memory_search"]
    result, _ = _post(client, body, stream)
    assert [item["name"] for item in result["portal"]["safe_tools_executed"]] == ["memory_write", "memory_search"]
    other = app.test_client()
    other.set_cookie(SESSION_COOKIE_NAME, "native-memory-other-session-12345")
    recalled = other.post("/api/tools/memory_search/call", headers=HEADERS, json={"query": "Source-bound note"})
    assert recalled.json["result"]["results"] == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("invalid", ["no_auto", "scope", "audio", "image", "detached", "background", "unknown", "bad_think", "no_goal", "bad_schema"])
def test_invalid_native_profile_fails_before_model_or_native_dispatch(tmp_path, monkeypatch, stream, invalid):
    _, adapters = _install_store(monkeypatch)

    def no_upstream(request):
        pytest.fail("Invalid profile must not contact a model")

    body = _request(stream)
    if invalid == "no_auto":
        body["portal_auto_tools"] = False
    elif invalid == "scope":
        body["portal_tool_allowlist"] = ["memory_write"]
    elif invalid in {"audio", "image"}:
        body["messages"][-1]["audios" if invalid == "audio" else "images"] = ["attached"]
    elif invalid == "detached":
        body.update(portal_detached_turn=True, portal_turn_id="native-detached-turn-12345")
    elif invalid == "background":
        body["portal_background_worker"] = True
    elif invalid == "unknown":
        body["portal_execution_profile"] = "unrecognized"
    elif invalid == "bad_think":
        body["think"] = "false"
    elif invalid == "no_goal":
        body["messages"] = [{"role": "system", "content": "No task."}]
    elif invalid == "bad_schema":
        body["omni"]["schema"] = "wrong.schema"
    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(no_upstream)))
    response = app.test_client().post("/api/chat/stream" if stream else "/api/chat", headers=HEADERS, json=body)
    assert response.status_code == 400, response.data
    assert adapters == []


def test_native_planning_read_uses_window_inspection_without_visual_policy_or_input(monkeypatch):
    requests = []
    active = Adapter("session")._state()
    windows = Adapter("session")._state(True)
    windows["nodes"].append({"id": "protected", "generation": "1", "scope_id": "form", "role": "textbox",
                             "name": "Private", "value": "Protected value", "visible": True, "enabled": True,
                             "sensitive": True, "operations": ["type_text"]})

    def bridge(request):
        requests.append(copy.deepcopy(request))
        assert request["op"] == "observe"
        return copy.deepcopy(windows if request["include_windows"] else active)

    adapter = NativeUiAdapter("session", bridge_runner=bridge, pointing_url="http://127.0.0.1:8940")
    monkeypatch.setattr(adapter, "_visual_capture", lambda *_: pytest.fail("Planning observation cannot call vision"))
    store = ComputerUseStore(None, adapter_factory=lambda *_: adapter)

    class NoPolicy:
        def decide(self, *_args, **_kwargs):
            pytest.fail("Read-only planning cannot request candidate policy")

    store.provider = NoPolicy()
    ordinary = store.act("session", {"operation": "observe", "surface": "desktop"})
    planned = store.planning_observation("session")
    assert requests == [{"op": "observe", "include_windows": False}, {"op": "observe", "include_windows": True}]
    assert all(node["id"] != "other-window" for node in ordinary["state"]["nodes"])
    assert any(node["id"] == "other-window" and node["operations"] == ["switch_window"] for node in planned["state"]["nodes"])
    protected = next(node for node in planned["state"]["nodes"] if node["id"] == "protected")
    assert protected["name"] == "[protected]" and "value" not in protected and protected["operations"] == []
    assert "candidates" not in planned and planned["evidence_authority"] == "inspection"
    assert store._owned("session", "desktop").controller._actions == {}


@pytest.mark.parametrize("stream", [False, True])
def test_malformed_compiler_content_stops_at_three_failures_without_input(tmp_path, monkeypatch, stream):
    store, adapters = _install_store(monkeypatch)
    requests = []
    original_goal = app_module.run_native_goal
    errors = []

    def checked_goal(*args, compile_phase, **kwargs):
        def compile_without_raw_reply(packet):
            try:
                return compile_phase(packet)
            except NativePhaseCompilationError as error:
                errors.append(error)
                assert error.args == () and error.__context__ is None
                raise

        return original_goal(*args, compile_phase=compile_without_raw_reply, **kwargs)

    monkeypatch.setattr(app_module, "run_native_goal", checked_goal)

    def upstream(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": '{"private-partial-canary"'}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, "native-malformed-owned-session-12345")
    result, _ = _post(client, _request(stream), stream)
    assert len(requests) == 3 and adapters[0].planning_calls == 3 and adapters[0].calls == []
    assert len(errors) == 3
    assert result["portal"]["native_task"]["status"] == "blocked"
    assert result["portal"]["safe_tools_executed"] == []
    assert store._owned("native-malformed-owned-session-12345", "desktop").cancel.is_set()
    assert "private-partial-canary" not in json.dumps(result)
    for payload in requests[1:]:
        packet = json.loads(payload["messages"][1]["content"])
        assert packet["last_result"]["error"] == "invalid_native_phase_json"
        assert packet["last_result"]["input_attempted"] is False
        assert "private-partial-canary" not in json.dumps(packet)


@pytest.mark.parametrize("stream", [False, True])
def test_malformed_compiler_content_recovers_with_fresh_observation_and_real_workflow(tmp_path, monkeypatch, stream):
    _, adapters = _install_store(monkeypatch)
    requests = []
    body = _request(stream)

    def upstream(request):
        payload = json.loads(request.content)
        requests.append(payload)
        packet = json.loads(payload["messages"][1]["content"])
        assert packet["goal"] == GOAL and payload["options"] == body["options"]
        assert payload["think"] is False and not payload.get("tools")
        if len(requests) == 1:
            assert adapters[0].calls == []
            adapters[0].value = "Fresh independently observed marker"
            content = '{"private-partial-canary"'
        elif len(requests) == 2:
            assert adapters[0].planning_calls == 2 and adapters[0].calls == []
            assert next(node for node in packet["current_state"]["state"]["nodes"] if node["id"] == "draft")["value"] == "Fresh independently observed marker"
            assert packet["last_result"]["error"] == "invalid_native_phase_json"
            assert "private-partial-canary" not in json.dumps(packet)
            content = json.dumps(_working_reply(request, _phase()))
        else:
            assert len(requests) == 3 and len(adapters[0].calls) == 2
            content = json.dumps(_working_reply(request, {"kind": "complete", "message": "Planner claim after recovery."}))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _ = _post(app.test_client(), body, stream)
    assert result["portal"]["native_task"]["status"] == "planner_claim"
    assert [action["verb"] for action in adapters[0].calls] == ["set_value", "invoke"]
    assert len(result["portal"]["safe_tools_executed"]) == 1
    assert result["portal"]["tool_allowlist"] == ["native_ui"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fault", ["transport", "HTTP", "response_JSON", "missing_content"])
def test_non_compiler_JSON_faults_are_fatal_without_replanning(tmp_path, monkeypatch, stream, fault):
    _, adapters = _install_store(monkeypatch)
    requests = []

    def upstream(request):
        requests.append(request)
        if fault == "transport":
            raise httpx.ReadTimeout("private transport canary")
        if fault == "HTTP":
            return httpx.Response(500, json={"error": "private upstream canary"})
        if fault == "response_JSON":
            return httpx.Response(200, text="private invalid envelope canary")
        return httpx.Response(200, json={"message": {"role": "assistant"}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _ = _post(app.test_client(), _request(stream), stream)
    assert len(requests) == 1 and adapters[0].calls == []
    assert result["portal"]["native_task"]["status"] == "blocked"
    assert result["portal"]["safe_tools_executed"] == []
    assert "private" not in json.dumps(result)


def test_stream_disconnect_before_observation_cancels_exact_scope_and_releases_queue(tmp_path, monkeypatch):
    store, adapters = _install_store(monkeypatch)

    def no_upstream(request):
        pytest.fail("Disconnected task cannot start its compiler")

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(no_upstream)))
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, "native-disconnected-owned-session-12345")
    response = client.post("/api/chat/stream", headers=HEADERS, json=_request(True), buffered=False)
    assert response.status_code == 200
    first = json.loads(next(iter(response.response)))
    assert first["type"] == "progress" and first["stage"] == "native_task_observe"
    response.close()
    activity = client.get("/api/activity", headers=HEADERS).json
    assert activity["active"] == activity["inflight"] == activity["queued"] == 0
    assert store._owned("native-disconnected-owned-session-12345", "desktop").cancel.is_set()
    assert all(adapter.calls == [] for adapter in adapters)


def test_crowded_native_planning_projection_keeps_mapped_inactive_windows(tmp_path, monkeypatch):
    _install_store(monkeypatch)
    original = Adapter._state

    def crowded(self, include_windows=False):
        state = original(self, include_windows)
        inactive = [node for node in state["nodes"] if node["id"] == "other-window"]
        controls = [node for node in state["nodes"] if node["id"] != "other-window"]
        controls.extend({"id": f"control-{index}", "generation": "1", "scope_id": "form", "role": "button",
                         "name": f"Visible control {index}", "description": "Observed interface detail " * 20,
                         "visible": True, "enabled": True, "sensitive": False, "operations": ["invoke"]}
                        for index in range(28))  # 31 active controls, then mapped inactive windows.
        state["nodes"] = controls + inactive
        return state

    monkeypatch.setattr(Adapter, "_state", crowded)

    def upstream(request):
        payload = json.loads(request.content)
        observation = json.loads(payload["messages"][1]["content"])["current_state"]
        assert any(node["id"] == "other-window" for node in observation["state"]["nodes"])
        assert observation["state"]["model_projection"]["omitted_visible_nodes"] > 0
        return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps(_working_reply(request, {
            "kind": "complete", "message": "Read-only planning claim."}))}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _ = _post(app.test_client(), _request(False), False)
    assert result["portal"]["native_task"]["status"] == "planner_claim"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("external", ["cancel", "trash"])
def test_external_native_cancel_latches_goal_before_late_memory_dispatch(tmp_path, monkeypatch, stream, external):
    store, adapters = _install_store(monkeypatch)
    session_id = "native-cancelled-owned-session-12345"
    calls, events = [], []
    cancel_client = None

    def upstream(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            events.append(store._owned(session_id, "desktop").cancel)
            plan = {"kind": "memory_write", "topic": "test", "key": "one", "value": "Before cancellation"}
        elif len(calls) == 2:
            if external == "cancel":
                response = cancel_client.post("/api/tools/native_ui/call", headers=HEADERS, json={"operation": "cancel"})
                assert response.status_code == 200
            else:
                assert cancel_client.delete("/api/diagnostics", headers=HEADERS).status_code == 204
            plan = {"kind": "memory_write", "topic": "test", "key": "two", "value": "Forbidden after cancellation"}
        else:
            pytest.fail("Cancelled native task cannot start another compiler call")
        return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps(_working_reply(request, plan))}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    client, cancel_client = app.test_client(), app.test_client()
    for item in (client, cancel_client):
        item.set_cookie(SESSION_COOKIE_NAME, session_id)
    body = _request(stream)
    body["portal_tool_allowlist"] = ["native_ui", "memory_write", "memory_search"]
    result, _ = _post(client, body, stream)
    assert result["portal"]["native_task"]["status"] == "cancelled"
    assert len(calls) == 2 and events[0].is_set()
    assert [trace["arguments"]["key"] for trace in result["portal"]["safe_tools_executed"]] == ["one"]
    assert len(adapters) == 1 and adapters[0].calls == []
    recalled = cancel_client.post("/api/tools/memory_search/call", headers=HEADERS, json={"query": "Forbidden after cancellation"})
    assert all(item["key"] != "two" and item["value"] != "Forbidden after cancellation"
               for item in recalled.json["result"]["results"])


def test_external_cancel_at_stream_tool_start_prevents_callback_and_next_compiler(tmp_path, monkeypatch):
    store, adapters = _install_store(monkeypatch)
    calls = []

    def upstream(request):
        calls.append(request)
        assert len(calls) == 1
        return httpx.Response(200, json={"message": {"content": json.dumps(_working_reply(request, _phase()))}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    client, canceller = app.test_client(), app.test_client()
    session_id = "native-tool-start-cancel-session-12345"
    for item in (client, canceller):
        item.set_cookie(SESSION_COOKIE_NAME, session_id)
    response = client.post("/api/chat/stream", headers=HEADERS, json=_request(True), buffered=False)
    iterator = iter(response.response)
    for chunk in iterator:
        event = json.loads(chunk)
        if event.get("type") == "tool" and event.get("phase") == "start":
            break
    assert canceller.post("/api/tools/native_ui/call", headers=HEADERS, json={"operation": "cancel"}).status_code == 200
    events = [json.loads(chunk) for chunk in iterator]
    response.close()
    result = next(event["response"] for event in events if event.get("type") == "final")
    assert result["portal"]["native_task"]["status"] == "cancelled"
    assert result["portal"]["safe_tools_executed"] == []
    assert len(calls) == 1 and all(adapter.calls == [] for adapter in adapters)
    assert store._owned(session_id, "desktop").cancel.is_set()
    activity = client.get("/api/activity", headers=HEADERS).json
    assert activity["active"] == activity["inflight"] == activity["queued"] == 0


@pytest.mark.parametrize("stream", [False, True])
def test_cancel_and_external_renew_before_store_lookup_cannot_dispatch_into_new_epoch(tmp_path, monkeypatch, stream):
    import contextvars

    store, adapters = _install_store(monkeypatch)
    session_id = "native-lookup-race-session-12345"
    original = tools_module.PortalToolHarness.execute
    old_events, renewed_events, compiler_calls = [], [], []

    def execute(harness, session, name, arguments):
        if name == "native_ui":
            old_events.append(store._owned(session, "desktop").cancel)
            store.clear(session)
            # An independent request may renew the current session after Trash.
            contextvars.Context().run(lambda: store.planning_observation(session))
            renewed_events.append(store._owned(session, "desktop").cancel)
        return original(harness, session, name, arguments)

    monkeypatch.setattr(tools_module.PortalToolHarness, "execute", execute)

    def upstream(request):
        compiler_calls.append(request)
        assert len(compiler_calls) == 1
        return httpx.Response(200, json={"message": {"content": json.dumps(_working_reply(request, _phase()))}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, session_id)
    result, _ = _post(client, _request(stream), stream)
    assert result["portal"]["native_task"]["status"] == "cancelled"
    assert len(adapters) == 2 and all(adapter.calls == [] for adapter in adapters)
    assert old_events[0].is_set() and old_events[0] is not renewed_events[0]
    assert renewed_events[0].is_set() is False  # Old task cleanup cannot cancel the new controller.
    trace = result["portal"]["safe_tools_executed"]
    assert len(trace) == 1 and json.loads(trace[0]["result"])["status"] == "cancelled"


def test_late_timer_from_completed_goal_cannot_cancel_next_goal_or_replay_old_receipts(tmp_path, monkeypatch):
    store, adapters = _install_store(monkeypatch)
    timers, task_events = [], []
    session_id = "native-timer-epoch-session-12345"

    class Timer:
        def __init__(self, _seconds, callback, args=(), kwargs=None):
            self.callback = lambda: callback(*args, **(kwargs or {}))
            if callback.__name__ == "cancel":
                timers.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr("portal.app.threading.Timer", Timer)

    def upstream(_request):
        task_events.append(store._owned(session_id, "desktop").cancel)
        if len(task_events) == 2:
            timers[0].callback()  # Simulate a timer already admitted before cancel().
            assert task_events[1].is_set() is False
        return httpx.Response(200, json={"message": {"content": json.dumps(_working_reply(_request, {"kind": "complete", "message": "Claim only"}))}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, session_id)
    first, _ = _post(client, _request(False), False)
    second, _ = _post(client, _request(False), False)
    assert first["portal"]["native_task"]["status"] == second["portal"]["native_task"]["status"] == "planner_claim"
    assert len(adapters) == 2 and task_events[0] is not task_events[1]
    assert task_events[0].is_set() and not task_events[1].is_set()


def test_new_native_goal_keeps_existing_session_memory(tmp_path, monkeypatch):
    _install_store(monkeypatch)
    plans = iter([
        {"kind": "memory_write", "topic": "source", "key": "kept", "value": "Preserved source note"},
        {"kind": "complete", "message": "First claim"},
        {"kind": "memory_search", "query": "Preserved source note", "max_results": 1},
        {"kind": "complete", "message": "Second claim"},
    ])

    def upstream(_request):
        return httpx.Response(200, json={"message": {"content": json.dumps(_working_reply(_request, next(plans)))}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, "native-memory-epoch-session-12345")
    body = _request(False)
    body["portal_tool_allowlist"] = ["native_ui", "memory_write", "memory_search"]
    _post(client, body, False)
    second, _ = _post(client, body, False)
    traces = second["portal"]["safe_tools_executed"]
    assert len(traces) == 1 and traces[0]["name"] == "memory_search"
    assert json.loads(traces[0]["result"])["results"][0]["value"] == "Preserved source note"


@pytest.mark.parametrize('stream', [False, True])
def test_cancel_renew_at_scope_entry_cannot_admit_memory_executor(tmp_path, monkeypatch, stream):
    from contextlib import contextmanager

    store, adapters = _install_store(monkeypatch)
    original = store.task_scope
    entries, next_events = [], []

    @contextmanager
    def scope(session_id, surface, event):
        entries.append(event)
        if len(entries) == 2:
            store.clear(session_id)
            next_events.append(store.begin_task(session_id, surface))
        with original(session_id, surface, event):
            yield

    monkeypatch.setattr(store, 'task_scope', scope)
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(200, json={'message': {'content': json.dumps(_working_reply(request, {
            'kind': 'memory_write', 'topic': 'test', 'key': 'forbidden', 'value': 'Must never be stored'}))}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, 'native-scope-entry-session-12345')
    body = _request(stream)
    body['portal_tool_allowlist'] = ['native_ui', 'memory_write', 'memory_search']
    result, _ = _post(client, body, stream)
    assert result['portal']['native_task']['status'] == 'cancelled'
    traces = result['portal']['safe_tools_executed']
    assert not any(json.loads(trace['result']).get('stored') for trace in traces)
    assert len(calls) == 1
    assert next_events and not next_events[0].is_set()
    assert all(adapter.calls == [] for adapter in adapters)
