from __future__ import annotations

import json
import threading

import httpx
import pytest

from portal.app import _InferenceQueue
from portal.episodic_memory import AdapterMemoryEncoder, EpisodicMemoryService, MemoryCapture


def service(root, **overrides):
    return EpisodicMemoryService(root, owner_session="owner", generate=lambda *_: {},
        embed=lambda _: [1.0, 0.0], admit=lambda: True, start_worker=False, **overrides)


def capture(memory, session, text, *, generation=None):
    return memory.capture(session, generation=memory.generation(session) if generation is None else generation,
        episode="episode", key=text[:20], modality="text", authority="user_statement", actor="user", text=text)


def test_perception_only_and_older_tagged_records_show_as_typed_evidence(tmp_path):
    import time

    memory = service(tmp_path)
    text = "Only keystroke-like clicks are audible, with no intelligible speech."
    markup = f"<speech_transcript></speech_transcript><audio_observation>{text}</audio_observation>"
    turn = MemoryCapture(memory, "one", "checkpoint", media={"has_audio_input": True})
    turn.response({"message": {"content": markup}, "adapter": {
        "task": "transcribe", "audio_observation": text,
        "evidence_provenance": {"current_media_modalities": ["audio"], "current_visual_input": False},
    }})
    memory.capture("one", generation=memory.generation("one"), episode="checkpoint", key="legacy-duplicate",
        modality="text", authority="assistant_generated", actor="agent", text=markup,
        provenance={"origin": "completed_language_response"}, occurred_at=time.time())
    memory.capture("one", generation=memory.generation("one"), episode="older", key="legacy-only",
        modality="text", authority="assistant_generated", actor="agent", text=markup,
        provenance={"origin": "completed_language_response"}, occurred_at=time.time() - 1)
    rows = memory.activity("one")["sources"]
    assert len(rows) == 2
    assert {(record["lane"], record["label"], record["text"]) for record in rows} == {
        ("heard", "Sound observation", text),
    }
    assert all(not record["audio_generated"] for record in rows)
    memory.close()


def test_states_activity_preserves_transcript_attribution_tool_outcomes_and_generated_speech(tmp_path):
    memory = service(tmp_path)
    turn = MemoryCapture(memory, "one", "request-one", user_text="Typed input", live_turn=True)
    turn.observation({"transcript": "heard exact words", "speech_addressee": "OTHER",
        "evidence_provenance": {"current_media_modalities": ["audio"], "current_visual_input": False}})
    turn.tool("mcp_call", {"ok": False}, ok=False)
    turn.response({"message": {"content": "Reply with formatting", "thinking": "hidden reasoning marker",
        "audio": {"transcript": "Exact synthesized words"}},
        "adapter": {"speech_synthesized": True, "speech_addressee": "OTHER"}})
    capture(memory, "owner", "private owner marker")
    capture(memory, "two", "other browser marker")
    activity = memory.activity("one")
    assert {item["lane"] for item in activity["sources"]} == {"heard", "done", "said"}
    speech = next(item for item in activity["sources"] if item["text"] == "heard exact words")
    assert speech["actor"].startswith("speaker:") and speech["addressee"] == "OTHER"
    assert any(item["text"] == "mcp_call · failed" for item in activity["sources"])
    reply = next(item for item in activity["sources"] if item["lane"] == "said")
    assert reply["text"] == "Exact synthesized words" and reply["audio_generated"] is True
    assert reply["playback_verified"] is False
    assert all(marker not in json.dumps(activity) for marker in ["hidden reasoning marker", "private owner marker", "other browser marker"])
    memory.clear("one")
    assert memory.activity("one")["sources"] == []
    memory.close()


def test_reported_speech_cannot_masquerade_as_validated_transcript(tmp_path):
    memory = service(tmp_path)
    memory.capture("one", generation=0, episode="report", key="report", modality="speech", authority="client_report",
        actor="client", text="reported words", provenance={"origin": "client_event"})
    assert memory.activity("one")["sources"][0]["label"] == "Reported speech"
    memory.close()


def test_foreground_cancels_optional_encoding_even_at_full_admission_capacity():
    queue = _InferenceQueue(1, 1)
    cancellation = threading.Event()
    background = queue.acquire_background("memory", cancellation)
    assert background is not None
    assert queue.snapshot()["background"] == 1
    result = []
    thread = threading.Thread(target=lambda: result.append(queue.acquire("person", 2)))
    thread.start()
    assert cancellation.wait(1)
    queue.release(background)
    thread.join(1)
    assert result and result[0] is not None
    assert queue.acquire_background("another", threading.Event()) is None
    queue.release(result[0])
    assert queue.snapshot()["inflight"] == queue.snapshot()["active"] == 0


def test_device_memory_is_aggregate_for_other_cookies_and_survives_owner_rebinding(tmp_path):
    memory = service(tmp_path)
    capture(memory, "owner", "private owner marker")
    capture(memory, "one", "first browser marker")
    capture(memory, "two", "second browser marker")
    view = memory.snapshot("one")
    assert view["device"]["events"] == 1 and view["device"]["content_available"] is False
    assert "private owner marker" not in json.dumps(view)
    assert "first browser marker" in json.dumps(view)
    assert "second browser marker" not in json.dumps(view)
    memory.close()
    rebound = EpisodicMemoryService(tmp_path, owner_session="new-owner", generate=lambda *_: {},
        embed=lambda _: [1.0, 0.0], admit=lambda: True, start_worker=False)
    assert "private owner marker" in json.dumps(rebound.snapshot("new-owner")["session"])
    assert "private owner marker" not in json.dumps(rebound.snapshot("owner"))
    rebound.close()


def test_trash_invalidates_inflight_encoding_and_stale_captures_without_recreating_old_sources(tmp_path):
    started, release = threading.Event(), threading.Event()
    def encode(*_):
        started.set()
        release.wait(2)
        return {}
    memory = EpisodicMemoryService(tmp_path, owner_session="owner", generate=encode,
        embed=lambda _: [1.0, 0.0], admit=lambda: True, start_worker=False)
    capture(memory, "one", "old private marker")
    old_generation = memory.generation("one")
    key = memory._key("one")
    current = memory._scope(key)
    thread = threading.Thread(target=lambda: memory._encode_one(key, current))
    thread.start()
    assert started.wait(1)
    memory.clear("one")
    assert capture(memory, "one", "stale private marker", generation=old_generation) == []
    capture(memory, "one", "new retained marker")
    release.set()
    thread.join(2)
    view = memory.snapshot("one")
    assert "new retained marker" in json.dumps(view)
    assert "old private marker" not in json.dumps(view)
    assert "stale private marker" not in json.dumps(view)
    memory.close()


def test_long_unicode_encoding_packets_cover_the_original_source_losslessly(tmp_path):
    memory = service(tmp_path)
    original = "記録された感覚と会話。" * 1500
    parts = list(memory._segments(original))
    assert len(parts) > 1
    assert "".join(part for _, part, _ in parts) == original
    assert all(not complete for _, _, complete in parts[:-1]) and parts[-1][2]
    memory.close()


def test_encoder_uses_native_boolean_closed_tools_and_deterministic_structured_output():
    observed = []
    def handler(request):
        observed.append(json.loads(request.content))
        return httpx.Response(200, content=b'{"type":"delta","message":{"content":"{\\"summary\\":\\"\\",\\"summary_support\\":[],\\"entities\\":[],\\"facts\\":[]}"}}\n')
    queue = _InferenceQueue(1, 1)
    encoder = AdapterMemoryEncoder("http://adapter/api/chat", "model", queue,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    assert encoder("one", '{"current":{}}', threading.Event())["facts"] == []
    payload = observed[0]
    assert payload["think"] is False
    assert payload["tools"] == [] and payload["tool_choice"] == "none"
    assert payload["response_modalities"] == ["text"] and payload["speech_mode"] == "never"
    assert payload["options"]["temperature"] == 0
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert queue.snapshot()["active"] == 0


def test_encoder_factory_failure_releases_the_optional_inference_lane():
    queue = _InferenceQueue(1, 1)
    def unavailable():
        raise RuntimeError("client unavailable")
    encoder = AdapterMemoryEncoder("http://adapter/api/chat", "model", queue, client_factory=unavailable)
    with pytest.raises(ValueError):
        encoder("one", "{}", threading.Event())
    assert queue.snapshot()["inflight"] == 0


def test_incoming_person_closes_a_running_encoding_stream_and_receives_the_lane():
    class BlockingStream(httpx.SyncByteStream):
        def __init__(self):
            self.started, self.closed = threading.Event(), threading.Event()
        def __iter__(self):
            self.started.set()
            yield b'{"type":"delta","message":{"content":"{"}}\n'
            self.closed.wait(3)
        def close(self):
            self.closed.set()
    stream = BlockingStream()
    queue = _InferenceQueue(1, 1)
    encoder = AdapterMemoryEncoder("http://adapter/api/chat", "model", queue,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))))
    result = []
    thread = threading.Thread(target=lambda: result.append(encoder("one", "{}", threading.Event())))
    thread.start()
    assert stream.started.wait(1)
    foreground = queue.acquire("person", 2)
    assert foreground is not None and stream.closed.is_set()
    thread.join(1)
    assert result == [None]
    queue.release(foreground)


def test_durable_inference_reserves_a_foreground_slot():
    queue = _InferenceQueue(2, 4)
    task = queue.acquire_durable_task("task", 0)
    assert task is not None
    assert queue.acquire_durable_task("second task", 0) is None
    voice = queue.acquire("voice", 0)
    assert voice is not None
    assert queue.snapshot()["active"] == 2
    queue.release(voice)
    queue.release(task)
    queue.release(task)
    assert queue.snapshot()["inflight"] == 0
    another = queue.acquire_durable_task("second task", 0)
    assert another is not None
    queue.release(another)


def test_capture_keeps_sensory_channels_and_authorities_separate(tmp_path):
    from portal.episodic_memory import MemoryCapture
    memory = service(tmp_path)
    turn = MemoryCapture(memory, "one", "senses", context_text="Attached media", live_turn=True,
        media={"has_audio_input": True}, media_receipts=["sha256:abc"])
    turn.observation({"transcript": "Bring that red thing here", "audio_observation": "Three short beeps",
        "visual_observation": "An invented current image", "evidence_provenance": {
            "current_media_modalities": ["audio"], "current_visual_input": False}, "speech_addressee": "OTHER"})
    turn.response({"message": {"content": "", "thinking": "hidden reasoning"}, "adapter": {"speech_addressee": "OTHER"}})
    view = memory.snapshot("one")["session"]
    assert view["modalities"] == {"speech": 1, "sound": 1}
    assert all(not item["provenance"]["current_visual_input"] for item in view["recent"])
    assert all(item["annotations"]["conversation_eligible"] is False for item in view["recent"])
    assert next(item for item in view["recent"] if item["modality"] == "speech")["authority"] == "unattributed"
    assert "hidden reasoning" not in json.dumps(view) and "Attached media" not in json.dumps(view)
    assert memory.search("one", "red", conversation_only=True)["results"] == []
    assert memory.search("one", "red", conversation_only=False)["results"]
    memory.close()


def test_tool_capture_keeps_ttl_payloads_and_uploaded_documents_out_of_durable_sources(tmp_path):
    from portal.episodic_memory import MemoryCapture
    memory = service(tmp_path)
    turn = MemoryCapture(memory, "one", "tools", user_text="Inspect this")
    turn.tool("get_user_location", {"city": "PRIVATELOCATION"}, ok=True, family="system", model_text="PRIVATELOCATION")
    turn.tool("web_fetch", {"text": "EPHEMERALWEB"}, ok=True, family="web", model_text="EPHEMERALWEB")
    turn.documents([{"id": "doc", "text": "EPHEMERALDOCUMENT"}])
    turn.tool("peripheral_inventory", {"interfaces": ["observed generic interface"]}, ok=True,
        family="peripherals", model_text='{"interfaces":["observed generic interface"]}')
    view = json.dumps(memory.snapshot("one"))
    assert not any(marker in view for marker in ("PRIVATELOCATION", "EPHEMERALWEB", "EPHEMERALDOCUMENT"))
    assert "observed generic interface" in view
    memory.close()


def test_restore_is_own_scope_historical_and_idempotent_and_excludes_current_episode(tmp_path):
    memory = service(tmp_path)
    turns = [{"turn_id": "old", "status": "complete", "started_at": 1000,
        "user": {"content": "Remember the amber box"}, "assistant": {"content": "I will"}},
        {"turn_id": "unfinished", "status": "running", "user": {"content": "never import partial"}}]
    memory.restore_conversation("one", turns)
    memory.restore_conversation("one", turns)
    assert memory.snapshot("one")["session"]["events"] == 2
    assert memory.search("one", "amber", exclude_episode="old")["results"] == []
    assert memory.search("two", "amber")["results"] == []
    assert "never import partial" not in json.dumps(memory.snapshot("one"))
    memory.close()


@pytest.mark.parametrize("detached", [False, True])
def test_portal_captures_actual_text_and_stream_observations_and_trash(tmp_path, monkeypatch, detached):
    from portal.app import PortalConfig, create_app
    memory = service(tmp_path / "graph")
    monkeypatch.setattr("portal.app.create_memory_service", lambda *_a, **_kw: memory)
    token = "test-token-with-more-than-24-characters"
    def handler(request):
        if request.url.path.endswith("/stream"):
            return httpx.Response(200, content=(
                '{"type":"observation","transcript":"I heard a bell","audio_observation":"A bell rang",'
                '"evidence_provenance":{"current_media_modalities":["audio"],"current_visual_input":false}}\n'
                '{"type":"stage","stage":"language"}\n'
                '{"type":"final","response":{"message":{"role":"assistant","content":"Understood"},"adapter":{}}}\n'))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "I remember", "thinking": "HIDDENTHOUGHT"}, "adapter": {}})
    app = create_app(PortalConfig(adapter_url="http://adapter/api/chat", adapter_health_url="http://adapter/healthz",
        comprehension_health_url="", tts_health_url="http://tts/healthz", ollama_health_url="http://ollama/api/tags",
        model="test-model", access_token=token, remote_camera_enabled=False, episodic_memory_enabled=True,
        background_task_path=tmp_path / "state" / "background.json", virtual_context_root=tmp_path / "virtual"),
        httpx.Client(transport=httpx.MockTransport(handler)))
    client = app.test_client()
    client.set_cookie("omni_portal_session", "browser-session-one")
    headers = {"Authorization": "Bearer " + token}
    body = {"model": "test-model", "messages": [{"role": "user", "content": "Transport placeholder"}],
        "portal_user_text": "The amber box is mine", "omni": {"schema": "robit.ollama.omni-adapter.v1", "task": "chat"},
        "stream": False, "response_modalities": ["text"], "speech_mode": "never"}
    assert client.post("/api/chat", json=body, headers=headers).status_code == 200
    body.update(stream=True, portal_user_text="")
    if detached:
        body.update(portal_detached_turn=True, portal_turn_id="retained-test-turn")
    result = client.post("/api/chat/stream", json=body, headers=headers)
    assert result.status_code == 200
    assert b'"type":"final"' in result.data
    runtime_id = result.headers["X-Omni-Request-ID"]
    dashboard = client.get("/api/observatory?activity=1", headers=headers).json
    assert any(event["event"] == "inference_stage" and event["request_id"] == runtime_id
               for event in dashboard["timeline"])
    assert {item["label"] for item in dashboard["activity_sources"]["sources"]
            if item["request_id"] == runtime_id} == {"Reply text", "Sound observation", "Speech transcript"}
    view = memory.snapshot("browser-session-one")["session"]
    assert view["modalities"]["speech"] == view["modalities"]["sound"] == 1
    assert "The amber box is mine" in json.dumps(view)
    assert "Transport placeholder" not in json.dumps(view) and "HIDDENTHOUGHT" not in json.dumps(view)
    event = client.post("/api/memory/events", json={"modality": "sensor", "text": "72 reported units", "authority": "user_statement"}, headers=headers)
    assert event.status_code == 201 and event.json["authority"] == "client_report"
    other = app.test_client()
    other.set_cookie("omni_portal_session", "browser-session-two")
    assert memory.snapshot("browser-session-two")["session"]["events"] == 0
    assert client.delete("/api/diagnostics", headers=headers).status_code == 204
    assert memory.snapshot("browser-session-one")["session"]["events"] == 0
    memory.close()


def test_local_voice_grant_exposes_only_owner_speech_and_typed_checkpoint_markers(tmp_path, monkeypatch):
    import hashlib

    from portal.app import PortalConfig, create_app
    from qwen_omni_adapters.voice_view import local_voice_view_key

    token = "test-token-with-more-than-24-characters"
    owner = hashlib.sha256(b"omni-voice-session\0" + token.encode()).hexdigest()
    memory = EpisodicMemoryService(tmp_path / "graph", owner_session=owner, generate=lambda *_: {},
        embed=lambda _: [1.0], admit=lambda: True, start_worker=False)
    monkeypatch.setattr("portal.app.create_memory_service", lambda *_args, **_kw: memory)
    state = tmp_path / "state"
    state.mkdir()
    proof = local_voice_view_key(state, create=True)

    def respond(request):
        if request.url.path.endswith("/stream"):
            assert "portal_voice_checkpoint" not in json.loads(request.content)
            return httpx.Response(200, content=(
                '{"type":"observation","transcript":"Private spoken words",'
                '"evidence_provenance":{"current_media_modalities":["audio"],"current_visual_input":false}}\n'
                '{"type":"final","response":{"message":{"role":"assistant","content":""},"adapter":{}}}\n'
            ))
        return httpx.Response(200, json={})

    app = create_app(PortalConfig(adapter_url="http://adapter/api/chat", adapter_health_url="http://adapter/healthz",
        comprehension_health_url="", tts_health_url="http://tts/healthz", ollama_health_url="http://ollama/api/tags", model="test-model",
        access_token=token, episodic_memory_enabled=True, remote_camera_enabled=False,
        background_task_path=state / "background.json", virtual_context_root=tmp_path / "virtual"),
        httpx.Client(transport=httpx.MockTransport(respond)))
    headers = {"Authorization": "Bearer " + token}
    origin = app.test_client()
    origin.set_cookie("omni_portal_session", owner)
    response = origin.post("/api/chat/stream", headers=headers, json={
        "model": "test-model", "stream": True, "portal_voice_checkpoint": True,
        "omni": {"schema": "robit.ollama.omni-adapter.v1", "task": "transcribe"},
        "messages": [{"role": "user", "content": "Interim speech transport"}],
    })
    assert b"Private spoken words" in response.data
    observer = app.test_client()
    observer.set_cookie("omni_portal_session", "ordinary-browser-session")
    endpoint = "/api/observatory?activity=1"
    public = observer.get(endpoint, headers=headers).json
    assert {"Speech checkpoint sent", "Interim transcript received"} <= {
        event["label"] for event in public["timeline"]}
    assert "Private spoken words" not in json.dumps(public)
    assert "voice_sources" not in observer.get(endpoint, headers={**headers, "X-Omni-Voice-View": "wrong"}).json
    local = observer.get(endpoint, headers={**headers, "X-Omni-Voice-View": proof}).json
    assert [source["text"] for source in local["voice_sources"]["sources"]] == ["Private spoken words"]
    assert local["activity_sources"]["sources"] == []
    public_with_proof = observer.get(endpoint, headers={**headers, "X-Omni-Voice-View": proof},
                                    base_url="https://shared.example").json
    assert "voice_sources" not in public_with_proof
    memory.close()


def test_legacy_import_verifies_backup_and_incrementally_retains_original_hashes(tmp_path):
    import hashlib
    import sqlite3

    from harness.history import ConversationHistory
    archive = tmp_path / "voice.sqlite3"
    history = ConversationHistory(archive)
    first = history.append("A bird called twice", kind="sound", occurred_at=100, source={"type": "sound"})
    memory = service(tmp_path / "graph", legacy_archive=archive)
    memory._import_legacy_step()
    view = memory.snapshot("owner")["session"]
    assert view["events"] == 1 and view["recent"][0]["occurred_at"] == 100
    assert view["recent"][0]["provenance"]["legacy_content_sha256"] == hashlib.sha256(b"A bird called twice").hexdigest()
    assert view["recent"][0]["provenance"]["legacy_event_id"] == first
    marker = next((tmp_path / "graph").glob("voice-import-*.json"))
    receipt = json.loads(marker.read_text())
    backup = tmp_path / "graph" / "backups" / receipt["backup"]
    with sqlite3.connect(backup) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    history.append("Where is that bird?", kind="heard", occurred_at=200, source={"speech_addressee": "OTHER"})
    memory._legacy_retry_after = 0
    memory._import_legacy_step()
    memory._legacy_retry_after = 0
    memory._import_legacy_step()
    assert memory.snapshot("owner")["session"]["events"] == 2
    assert memory.search("owner", "bird", conversation_only=True)["results"] == []
    assert memory.snapshot("stranger")["device"]["events"] == 2
    assert "bird" not in json.dumps(memory.snapshot("stranger"))
    memory.close()
    history.close()


def test_episodic_answer_plan_closes_dispatch_and_preserves_historical_sources(tmp_path):
    from portal.app import _tool_followup
    from portal.documents import SessionDocumentStore
    from portal.tools import PortalToolHarness, tool_schemas
    memory = service(tmp_path)
    capture(memory, "one", "The amber box belongs to Nia")
    harness = PortalToolHarness(SessionDocumentStore())
    harness.episodic_memory = memory
    payload = {"messages": [{"role": "user", "content": "Who owns the amber box?"}], "tools": tool_schemas(["episodic_search"])}
    response = {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": "episodic_search", "arguments": {"query": "amber box", "period": "all", "kind": "all", "after_retrieval": "answer"}}}]}}
    followup, executed, _ = _tool_followup(payload, response, harness, "one")
    assert executed[0]["ok"] is True
    assert followup["tool_choice"] == "none" and "tools" not in followup
    receipt = json.loads(executed[0]["result"])
    assert receipt["retrieval_complete"] is True and receipt["source_is_current_visual_evidence"] is False
    assert receipt["results"][0]["text"] == "The amber box belongs to Nia"
    memory.close()


def test_invalid_encoding_gets_one_source_bounded_repair_and_keeps_source_vectors(tmp_path):
    attempts = []
    def generate(_scope, packet, _cancel):
        value = json.loads(packet)
        attempts.append(value)
        current = value["current"]
        reference = {"event_id": current["event_id"] if len(attempts) > 1 else 9999, "quote": current["text"]}
        return {"entities": [{"name": "Aster", "aliases": [], "support": [reference]}], "facts": []}
    memory = EpisodicMemoryService(tmp_path, owner_session="owner", generate=generate,
        embed=lambda _: [1.0, 0.0], admit=lambda: True, start_worker=False)
    capture(memory, "one", "The recorder is Aster")
    key = memory._key("one")
    assert memory._encode_one(key, memory._scope(key)) is True
    assert len(attempts) == 2
    assert attempts[1]["encoding_validation"]["allowed_event_ids"] == [1]
    view = memory.snapshot("one")["session"]
    assert view["events"] == view["vectors"] == view["entities"] == 1
    assert view["encoding_pending"] == 0
    memory.close()
