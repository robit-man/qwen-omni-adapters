"""The passive foreground guard tolerates only a bounded run of read timeouts."""
import threading
from types import SimpleNamespace

import pytest

from runtime import verify_long_horizon_native as runner
from runtime.passive_focus_guard import FocusUnavailable


class Deadline:
    def __init__(self):
        self.expired = False

    def _expire(self):
        self.expired = True


def guard(tmp_path, monkeypatch, outcomes):
    queue = list(outcomes)
    stop = threading.Event()

    def sample(_environment, _scope, stop_requested):
        item = queue.pop(0) if queue else "stop"
        if item == "stop":
            stop.set()
            return SimpleNamespace(identity=None, receipt=lambda: {"status": "stable"})
        if item == "ok":
            return SimpleNamespace(identity={"window_id": "1"}, receipt=lambda: {"status": "stable"})
        error_type, code = item
        raise FocusUnavailable(code, {"status": "unavailable", "code": code, "error_type": error_type})

    monkeypatch.setattr(runner, "sample_owned_focus", sample)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="X=1\nY=2\n"))
    desktop = SimpleNamespace(environment={}, windows=[SimpleNamespace(window_id="1", pid="2", process_start="3")])
    value = runner.Guard(desktop, Deadline(), tmp_path)
    value.stop = stop
    value.stop.wait = lambda _seconds: None
    value.run()
    return value


TIMEOUT = ("TimeoutExpired", "guard_focus_observation_unavailable")


def test_isolated_read_timeouts_are_recorded_and_tolerated(tmp_path, monkeypatch):
    value = guard(tmp_path, monkeypatch, ["ok", TIMEOUT, TIMEOUT, "ok", TIMEOUT, "ok"])
    assert value.failure is None and not value.deadline.expired
    assert value.read_timeouts == 3
    assert any(sample.get("error_type") == "TimeoutExpired" for sample in value.focus_samples)


def test_a_sustained_run_of_timeouts_still_stops_the_task(tmp_path, monkeypatch):
    value = guard(tmp_path, monkeypatch, ["ok"] + [TIMEOUT] * (runner.MAX_CONSECUTIVE_GUARD_READ_TIMEOUTS + 1))
    assert value.failure is not None and value.deadline.expired


@pytest.mark.parametrize("failure", [("NativeUiUnavailable", "guard_focus_owner_changed"),
                                     ("RuntimeError", "guard_focus_observation_unavailable")])
def test_observed_changes_and_non_timeout_failures_stop_immediately(tmp_path, monkeypatch, failure):
    value = guard(tmp_path, monkeypatch, ["ok", failure, "ok"])
    assert value.failure is not None and value.deadline.expired and value.read_timeouts == 0
