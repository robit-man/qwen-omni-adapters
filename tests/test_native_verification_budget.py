"""Desktop verification time for the retained Egg41 041 observation contract.

The fixture keeps the queried root and semantic revisions from the sealed
receipt; unrelated nodes are omitted. No X server, native input or model runs.
"""
from __future__ import annotations

import copy

import pytest

from portal.computer_use import ComputerUseStore
from qwen_omni_adapters import computer_use as core

EVENTS_SHA256 = "9bcdfb5d3f0f3084e17fec7083367ede7df573de9e752c7068f0bb13b21c6fb9"
ROOT = "atspi:1b93054e05c9fe0dc40f2db70f2bccf9"
BEFORE = "489dbd8dc3b88a2a9f5941d15198687838824cb5f70ebae398f36e6268fd9d08"
AFTER = "318c5607ee8a7b577e621ef138245f9527c120fbe2bf32f3e39198ab33576c56"
OBSERVATION_SECONDS = 4.406875471002422


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


class RetainedWindow:
    def __init__(self, clock, *, delay=OBSERVATION_SECONDS, unchanged=False):
        self.clock, self.delay, self.unchanged = clock, delay, unchanged
        self.dispatched = []
        self.reads = 0
        self.post_reads = 0
        self.visible = True
        self.after_read = None

    def observe(self):
        self.reads += 1
        if self.dispatched:
            self.post_reads += 1
            self.clock.value += self.delay
            if self.after_read:
                self.after_read()
        return {
            "session_id": "retained-041", "revision": AFTER if self.dispatched or self.unchanged else BEFORE,
            "scope": {"id": ROOT, "kind": "desktop", "provider": "atspi2", "window_id": "75497476"},
            "truncated": False, "nodes": [{
                "id": ROOT, "generation": "1", "scope_id": ROOT, "role": "window", "name": "Workbench",
                "visible": self.visible, "enabled": True, "sensitive": False, "focused": False,
                "operations": ["click", "double_click", "drag_drop", "right_click", "scroll", "key_chord", "switch_window"],
                "bounds": {"x": 165, "y": 131, "width": 975, "height": 834},
            }],
        }

    def perform(self, action):
        self.dispatched.append(copy.deepcopy(action))
        return {"status": "dispatched", "reason": "native_pointer_invoked", "backend": "x11_guarded_xtest"}


def request():
    return {"id": "open-brief", "verb": "click", "target": {"node_id": ROOT},
            "expect": [{"kind": "visible", "role": "window", "visible": True}]}


def fixture(monkeypatch, *, surface="desktop", **kwargs):
    clock = Clock()
    monkeypatch.setattr(core.time, "monotonic", clock)
    adapter = RetainedWindow(clock, **kwargs)
    store = ComputerUseStore(None, adapter_factory=lambda *_: adapter)
    owned = store._owned("retained-041", surface)
    owned.controller.observation_interval_s = .001
    return store, owned, adapter, clock


def test_store_desktop_accepts_retained_true_predicate_after_actual_observation_cost(monkeypatch):
    _, owned, adapter, _ = fixture(monkeypatch)
    result = owned.controller.run_step(request(), cancel=owned.cancel)
    assert result["status"] == "verified"
    assert result["verification"]["elapsed_s"] == OBSERVATION_SECONDS
    assert result["verification"]["timeout_s"] == 10
    assert result["before"]["revision"] == BEFORE
    assert result["after"]["revision"] == AFTER
    assert result["before"]["nodes"][0]["focused"] is False
    assert result["after"]["nodes"][0]["focused"] is False
    assert len(adapter.dispatched) == 1


def test_unchanged_visible_truth_verifies_without_mutation_or_task_progress(monkeypatch):
    store, _, adapter, _ = fixture(monkeypatch, unchanged=True)
    result = store.act("retained-041", {"operation": "workflow", "surface": "desktop",
                                        "steps": [request()], "final_predicates": request()["expect"]})
    assert result["status"] == "complete"
    assert result["outcomes"][0]["status"] == "verified"
    assert result["outcomes"][0]["before"]["revision"] == AFTER
    assert result["outcomes"][0]["after"]["revision"] == AFTER
    assert result["phase_handoff"]["new_effect_count"] == 0
    assert result["task_progress"] is False
    assert len(adapter.dispatched) == 1


def test_desktop_can_complete_two_bounded_observations_without_repeating_input(monkeypatch):
    _, owned, adapter, _ = fixture(monkeypatch)
    adapter.after_read = lambda: setattr(adapter, "visible", adapter.post_reads == 2)
    result = owned.controller.run_step(request(), cancel=owned.cancel)
    assert result["status"] == "verified"
    assert result["verification"]["observations"] == 2
    assert result["verification"]["elapsed_s"] == 2 * OBSERVATION_SECONDS
    assert len(adapter.dispatched) == 1


@pytest.mark.parametrize("delay", [10.001, 11])
def test_desktop_after_state_past_deadline_does_not_verify_true_predicate(monkeypatch, delay):
    _, owned, adapter, _ = fixture(monkeypatch, delay=delay)
    result = owned.controller.run_step(request(), cancel=owned.cancel)
    assert result["status"] == "unresolved"
    assert core.predicate_satisfied(request()["expect"][0], result["after"]) is True
    assert result["verification"]["observations"] == 1
    assert result["verification"]["timeout_s"] == 10
    assert len(adapter.dispatched) == 1


def test_exact_deadline_equality_keeps_existing_core_boundary(monkeypatch):
    _, owned, _, _ = fixture(monkeypatch, delay=10)
    assert owned.controller.run_step(request(), cancel=owned.cancel)["status"] == "verified"


def test_browser_and_generic_controller_keep_original_two_and_half_seconds(monkeypatch):
    _, owned, adapter, _ = fixture(monkeypatch, surface="browser")
    assert owned.controller.verification_timeout_s == 2.5
    assert core.RapidUiController(adapter).verification_timeout_s == 2.5
    result = owned.controller.run_step(request(), cancel=owned.cancel)
    assert result["status"] == "unresolved"
    assert result["verification"]["timeout_s"] == 2.5
    assert len(adapter.dispatched) == 1


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_cancellation_remains_authoritative_around_observation(monkeypatch, boundary):
    _, owned, adapter, _ = fixture(monkeypatch)
    if boundary == "before":
        owned.cancel.set()
    else:
        adapter.after_read = owned.cancel.set
    result = owned.controller.run_step(request(), cancel=owned.cancel)
    assert result["status"] == "cancelled"
    assert len(adapter.dispatched) == (0 if boundary == "before" else 1)
    if boundary == "before":
        assert adapter.reads == 0


def test_false_effect_stays_unresolved_and_cached_outcome_never_redispatches(monkeypatch):
    _, owned, adapter, _ = fixture(monkeypatch)
    step = request()
    step["expect"] = [{"kind": "focused", "node_id": ROOT}]
    result = owned.controller.run_step(step, cancel=owned.cancel)
    assert result["status"] == "unresolved"
    assert result["after"]["nodes"][0]["focused"] is False
    assert len(adapter.dispatched) == 1
    again = owned.controller.execute(result["action"]["id"], owned.cancel)
    assert again["status"] == "unresolved"
    assert len(adapter.dispatched) == 1


def test_existing_observation_count_and_constructor_maximum_unchanged(monkeypatch):
    _, owned, adapter, _ = fixture(monkeypatch)
    assert owned.controller.max_observations == 20
    assert core.RapidUiController(adapter, verification_timeout_s=30).verification_timeout_s == 30
    with pytest.raises(core.ComputerUseError):
        core.RapidUiController(adapter, verification_timeout_s=30.001)
