from __future__ import annotations

import json
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
from qwen_omni_adapters.memory import (
    MemoryGovernor,
    MemoryPolicy,
    MemoryPressure,
    release_unused_process_memory,
)


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


def test_releasing_unused_memory_trims_glibc_without_changing_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    class Trim:
        argtypes = None
        restype = None

        def __call__(self, pad: int) -> int:
            calls.append(pad)
            return 1

    class LibC:
        malloc_trim = Trim()

    monkeypatch.setattr("qwen_omni_adapters.memory.gc.collect", lambda: 0)
    monkeypatch.setattr("qwen_omni_adapters.memory.platform.system", lambda: "Linux")
    monkeypatch.setattr("qwen_omni_adapters.memory.ctypes.CDLL", lambda _name: LibC())

    assert release_unused_process_memory() is True
    assert calls == [0]


def test_declarative_tool_admission_distinguishes_new_bounded_and_executor_work() -> None:
    governor = MemoryGovernor(_policy(), sampler=lambda: 2.5)

    class Browser:
        calls = 0

        def act(self, *_args, **_kwargs):
            self.calls += 1
            return {"rendered": True}

        def clear(self, _session_id: str) -> None:
            pass

    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300),
        browser_automation=Browser(),
        memory_governor=governor,
    )

    browser = harness.execute("session", "browser_interact", {"action": "snapshot"})
    shell = harness.execute("session", "shell", {"command": "printf admitted"})
    math_result = harness.execute("session", "safe_math_eval", {"expression": "2 + 2"})

    assert browser == {"rendered": True}
    assert harness.browser.calls == 1
    assert shell["exit_code"] == 0
    assert shell["stdout"] == "admitted"
    assert math_result["result"] == 4

    pressured = PortalToolHarness(
        SessionDocumentStore(ttl_s=300),
        memory_governor=MemoryGovernor(_policy(), sampler=lambda: 2.4),
    ).execute("session", "shell", {"command": "printf should-not-run"})
    assert pressured["error"] == "resource_pressure"
    assert pressured["retryable"] is True


def test_task_control_remains_available_at_the_memory_floor(tmp_path: Path) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Stop this task when asked.")
    governor = MemoryGovernor(_policy(), sampler=lambda: 0.5)
    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300),
        background_tasks=store,
        memory_governor=governor,
    )

    listed = harness.execute("voice", "background_task", {"action": "list"})
    cancelled = harness.execute(
        "voice",
        "background_task",
        {"action": "cancel", "task_id": task["task_id"]},
    )

    assert listed["tasks"][0]["task_id"] == task["task_id"]
    assert cancelled["found"] is True
    assert cancelled["task"]["status"] == "cancelled"


def test_normal_reserve_does_not_double_count_the_soft_floor() -> None:
    governor = MemoryGovernor(_policy(), sampler=lambda: 3.5)

    assert governor.required_gib() == 3.0
    governor.require("resident task")


def test_bounded_continuation_uses_hard_not_soft_floor() -> None:
    governor = MemoryGovernor(_policy(), sampler=lambda: 2.5)
    governor.require_hard_floor("resident continuation")

    governor = MemoryGovernor(_policy(), sampler=lambda: 1.5)
    with pytest.raises(MemoryPressure) as error:
        governor.require_hard_floor("resident continuation")
    assert error.value.required_gib == 2.0


def test_measured_incremental_capacity_does_not_use_unknown_work_floor() -> None:
    governor = MemoryGovernor(_policy(), sampler=lambda: 2.6)

    with pytest.raises(MemoryPressure):
        governor.require("unknown new work")
    governor.require_capacity("measured resident executor", 0.5)

    governor = MemoryGovernor(_policy(), sampler=lambda: 2.4)
    with pytest.raises(MemoryPressure) as error:
        governor.require_capacity("measured resident executor", 0.5)
    assert error.value.required_gib == 2.5


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


def test_background_compacts_before_soft_floor_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Build the exact requested app.", "A fresh HTTP probe passes.")
    claimed = store.claim_next("seed")
    assert claimed is not None
    messages = [
        {"role": "system", "content": "old policy"},
        {"role": "user", "content": "exact objective"},
    ]
    for index in range(24):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "function": {
                                "name": "shell",
                                "arguments": {"command": f"step-{index}"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_name": "shell",
                    "tool_call_id": f"call-{index}",
                    "content": '{"exit_code": 0}',
                },
            ]
        )
    store.checkpoint(
        task["task_id"],
        "seed",
        messages=messages,
        progress="The workspace exists.",
        status="pending",
    )

    stop = threading.Event()
    governor = MemoryGovernor(_policy(), sampler=lambda: 2.5)
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: (_ for _ in ()).throw(
                AssertionError("this test stops before network inference")
            )
        )
    )
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=stop,
        memory_governor=governor,
        client=client,
    )
    executed: list[dict[str, object]] = []

    def execute(compacted_task: dict[str, object]) -> None:
        executed.append(compacted_task)
        stop.set()

    monkeypatch.setattr(agent, "_execute", execute)
    agent.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not executed:
        time.sleep(0.01)
    agent.close()
    client.close()

    assert executed
    current = store.get(task["task_id"])
    assert current is not None
    assert current["compaction"]["reason"] == "memory_pressure_pre_admission"
    assert current["compaction"]["before"]["messages"] == len(messages)
    assert current["compaction"]["after"]["messages"] < 20
    assert current["round"] == 1
    persisted = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))
    retained = persisted["tasks"][0]["messages"]
    assert retained[1]["content"] == "exact objective"
    assert "The workspace exists" not in retained[2]["content"]
    assert "retained concrete state" in retained[2]["content"]


def test_idle_background_scheduler_does_not_poll_memory_or_log_pressure(
    tmp_path: Path,
) -> None:
    samples = 0

    def sample() -> float:
        nonlocal samples
        samples += 1
        return 0.5

    store = BackgroundTaskStore(tmp_path / "tasks.json")
    governor = MemoryGovernor(_policy(), sampler=sample)
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: (_ for _ in ()).throw(
                AssertionError("an idle worker must not call the portal")
            )
        )
    )
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

    assert samples == 0
