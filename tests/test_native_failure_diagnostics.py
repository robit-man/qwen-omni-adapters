"""Native diagnostic codes survive uncertain execution without changing authority."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
from test_native_ui import Fixture as NativeFixture
from test_ui_task import Fixture, final, phase

from portal.native_ui import NativeUiUnavailable, _JsonBridge


def test_native_post_dispatch_failure_reaches_next_planner_packet_without_replay():
    fixture = Fixture([phase(), {"kind": "complete", "message": "Unverified claim"}])
    original_observe = fixture.adapter.observe
    original_perform = fixture.adapter.perform
    fail_next = [False]

    def perform(action):
        receipt = original_perform(action)
        fail_next[0] = True
        return receipt

    def observe():
        if fail_next[0]:
            fail_next[0] = False
            raise NativeUiUnavailable("native_observation_budget_exceeded")
        return original_observe()

    fixture.adapter.observe = observe
    fixture.adapter.perform = perform
    response = final(list(fixture.run()))
    raw = json.loads(response["portal"]["safe_tools_executed"][0]["result"])
    assert raw["status"] == "needs_replan"
    assert raw["outcomes"][0]["status"] == "unresolved"
    assert raw["outcomes"][0]["receipt"]["status"] == "dispatched"
    assert "native_observation_budget_exceeded" in raw["outcomes"][0]["reason"]
    assert "native_observation_budget_exceeded" in fixture.packets[1]["last_result"]["outcomes"][0]["reason"]
    assert len(fixture.adapter.calls) == 1
    assert response["portal"]["native_task"]["task_verified"] is False


def test_native_observe_failure_preserves_only_code_at_terminal_boundary():
    fixture = Fixture([])

    def fail():
        raise NativeUiUnavailable("native_provider_unavailable", "provider-private-detail")

    fixture.observe = fail
    response = final(list(fixture.run()))
    assert response["portal"]["native_task"]["status"] == "blocked"
    assert response["message"]["content"] == (
        "Native task callback failed: NativeUiUnavailable (native_provider_unavailable)"
    )
    assert "provider-private-detail" not in json.dumps(response)
    assert fixture.packets == fixture.executions == fixture.adapter.calls == []
    assert fixture.cancel_count == 1


@pytest.mark.parametrize("code", ["native_bridge_timeout", "a", "n" * 128])
def test_valid_bounded_code_keeps_original_code_and_custom_message(code):
    error = NativeUiUnavailable(code, "Original safe description")
    assert error.code == code
    assert str(error) == code + ": Original safe description"


@pytest.mark.parametrize("code", [
    "", "NATIVE_ERROR", "native-error", "native error=private", "native_error\nprivate",
    "native_error/secret", "n" * 129, "n" * 100000, "échec", 12, None, [], {}, b"native_error",
])
def test_malformed_code_is_preserved_internally_but_never_displayed(code):
    error = NativeUiUnavailable(code)
    assert error.code is code
    assert str(error) == "Native accessibility is unavailable"
    fixture = Fixture([])

    def fail():
        raise error

    fixture.observe = fail
    response = final(list(fixture.run()))
    assert response["message"]["content"] == "Native task callback failed: NativeUiUnavailable"
    assert response["portal"]["native_task"]["status"] == "blocked"
    assert fixture.packets == fixture.executions == fixture.adapter.calls == []


@pytest.mark.parametrize("boundary", ["cancelled", "deadline_exceeded"])
def test_native_code_never_overrides_latched_cancel_or_deadline(boundary):
    now = [0]
    fixture = Fixture([], clock=lambda: now[0])

    def fail():
        if boundary == "cancelled":
            fixture.cancelled.set()
        else:
            now[0] = 6
        raise NativeUiUnavailable("native_bridge_timeout", "private transport detail")

    fixture.observe = fail
    response = final(list(fixture.run(max_seconds=5)))
    assert response["portal"]["native_task"]["status"] == boundary
    assert response["message"]["content"] == "Native task stopped: " + boundary
    assert "native_bridge_timeout" not in json.dumps(response)
    assert fixture.cancel_count == 1


def test_optional_code_protocol_does_not_expose_arbitrary_exception_message():
    class CapabilityFailure(RuntimeError):
        code = "native_provider_unavailable"

    fixture = Fixture([])

    def fail():
        raise CapabilityFailure("private provider arguments and path")

    fixture.observe = fail
    response = final(list(fixture.run()))
    assert response["message"]["content"] == (
        "Native task callback failed: CapabilityFailure (native_provider_unavailable)"
    )
    assert "private provider" not in json.dumps(response)


def _transport_child(script, timeout_s):
    """Real fixed JSON pipe/timeout fixture; no native desktop/provider is opened."""
    bridge = object.__new__(_JsonBridge)
    bridge.closed = False
    bridge.session_id = "owned-diagnostic-fixture"
    bridge.process_id = None
    bridge.timeout_s = timeout_s
    bridge.process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-I", "-u", "-c", script],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return bridge


def test_real_bridge_error_reply_preserves_code_without_closing_synchronized_transport():
    bridge = _transport_child(
        "import json,sys\n"
        "for line in sys.stdin:\n"
        " request=json.loads(line)\n"
        " print(json.dumps({'schema':request['schema'],'id':request['id'],"
        "'ok':False,'error':'native_provider_unavailable'}),flush=True)\n", 2,
    )
    try:
        with pytest.raises(NativeUiUnavailable) as caught:
            bridge.request({"operation": "observe"})
        assert caught.value.code == "native_provider_unavailable"
        assert str(caught.value).startswith("native_provider_unavailable:")
        assert not bridge.closed and bridge.process.poll() is None
    finally:
        bridge.close()


def test_real_bridge_timeout_preserves_code_and_existing_owned_child_cleanup():
    bridge = _transport_child("import sys,time\nsys.stdin.readline()\ntime.sleep(30)\n", .05)
    try:
        with pytest.raises(NativeUiUnavailable) as caught:
            bridge.request({"operation": "observe"})
        assert caught.value.code == "native_bridge_timeout"
        assert str(caught.value).startswith("native_bridge_timeout:")
        assert bridge.closed and bridge.process.poll() is not None
    finally:
        bridge.close()


def test_actual_native_read_failure_retains_existing_provider_error_code():
    fixture = NativeFixture()

    def unreadable():
        raise RuntimeError("private native provider detail")

    fixture.button.get_name = unreadable
    with pytest.raises(NativeUiUnavailable) as caught:
        fixture.adapter.observe()
    assert caught.value.code == "native_observation_unavailable"
    assert str(caught.value) == "native_observation_unavailable: Native accessibility is unavailable"
    assert "private native provider detail" not in str(caught.value)
    assert fixture.button.events == []
