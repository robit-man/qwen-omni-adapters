from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.background_agent import BackgroundAgent
from portal.background_tasks import BackgroundTaskStore
from portal.documents import SessionDocumentStore
from portal.tools import PortalToolHarness, _run_shell
from qwen_omni_adapters.memory import MemoryGovernor, MemoryPolicy, MemoryPressure


def _policy() -> MemoryPolicy:
    return MemoryPolicy(
        enabled=True,
        soft_floor_gib=3.0,
        hard_floor_gib=2.0,
        operation_reserve_gib=1.0,
        poll_interval_s=0.01,
        wait_initial_s=0.01,
        wait_max_s=0.02,
    )


def test_one_governor_admits_every_tool_class_before_execution() -> None:
    governor = MemoryGovernor(_policy(), sampler=lambda: 2.5)

    class Browser:
        def act(self, *_args, **_kwargs):
            raise AssertionError("browser must not start below the shared reserve")

        def clear(self, _session_id: str) -> None:
            pass

    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300),
        browser_automation=Browser(),
        memory_governor=governor,
    )

    for name, arguments in (
        ("browser_interact", {"action": "snapshot"}),
        ("shell", {"command": "printf should-not-run"}),
        ("safe_math_eval", {"expression": "2 + 2"}),
    ):
        result = harness.execute("session", name, arguments)
        assert result["error"] == "resource_pressure"
        assert result["retryable"] is True


def test_normal_reserve_does_not_double_count_the_soft_floor() -> None:
    governor = MemoryGovernor(_policy(), sampler=lambda: 3.5)

    assert governor.required_gib() == 3.0
    governor.require("resident task")


def test_shell_is_killed_if_memory_collapses_after_admission(tmp_path: Path) -> None:
    samples = iter([10.0, 0.5])
    governor = MemoryGovernor(
        _policy(), sampler=lambda: next(samples, 0.5)
    )
    started = time.monotonic()

    with pytest.raises(MemoryPressure):
        _run_shell(
            "sleep 5",
            cwd=str(tmp_path),
            timeout_seconds=10,
            memory_governor=governor,
        )

    assert time.monotonic() - started < 1.0


def test_resource_pressure_never_becomes_task_evidence_or_a_blocker(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Create and verify the requested artifact.")
    governor = MemoryGovernor(_policy(), sampler=lambda: 0.5)

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("model and tools must not start below the shared reserve")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=threading.Event(),
        memory_governor=governor,
        client=client,
    )
    agent.start()
    time.sleep(0.08)
    agent.close()
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["status"] == "pending"
    assert current["progress"] == ["Accepted from the live conversation."]
    assert "error" not in current
    persisted = (tmp_path / "tasks.json").read_text(encoding="utf-8").lower()
    assert "memory headroom" not in persisted
    assert '"messages": []' in persisted
