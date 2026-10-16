"""Private causal input boundaries: real readback never means verified progress."""

from __future__ import annotations

import copy
import hashlib
import json
import threading

import pytest
from jsonschema import Draft202012Validator

from portal.computer_use import ComputerUseStore, _boundary_changes
from portal.native_ui import NativeUiAdapter
from portal.ui_task import (
    _compact_result,
    _grounded_phase_schema,
    phase_response_schema,
    run_native_goal,
)
from qwen_omni_adapters.computer_use import (
    ComputerUseError,
    RapidUiController,
    parse_intent,
    parse_step,
)
from runtime.long_horizon_cases import load_case

CELL = "atspi:75b465cc5aca928406ab145db9785aea"
ROW = "atspi:977b8c5d8daf21ea3643becf65b542b3"
PARENT = "atspi:e18eaf1b95a7d7dff15579be9cd38434"


def node(identity, role, name, operations, **extra):
    return dict(
        id=identity,
        generation="1",
        scope_id="scope",
        role=role,
        name=name,
        operations=list(operations),
        visible=True,
        enabled=True,
        sensitive=False,
        **extra,
    )


def intent(verb="click", identity="read-brief"):
    return dict(id=identity, verb=verb, target={"node_id": CELL}, args={})


def ui_step(verb="click", identity="read-brief", expect=None):
    return {
        **intent(verb, identity),
        "expect": expect or [{"kind": "text", "text": "Not observed"}],
    }


def ui(*steps):
    return {
        "kind": "ui",
        "steps": list(steps),
        "final_predicates": [{"kind": "visible", "name": "Input", "visible": True}],
    }


class Adapter:
    def __init__(self):
        self.selected, self.focused, self.value = False, False, ""
        self.calls, self.prepared, self.semantic_reads = [], [], 0
        self.mutate = True
        self.after_error = False
        self.reply = {"status": "dispatched"}
        self.hook = None
        self.observe_hook = None
        self.finished = 0

    def observe(self):
        if self.observe_hook:
            self.observe_hook(self)
        nodes = [
            node(
                CELL,
                "table_cell",
                "Input",
                ("click", "double_click", "right_click", "invoke"),
                focused=False,
                bounds={"x": 248, "y": 124, "width": 470, "height": 44},
            ),
            node(ROW, "table_row", "Input", ("focus",), focused=self.focused),
            node(PARENT, "list", "", ("select",), selected=["Input"] if self.selected else []),
            node(
                "draft",
                "textbox",
                "Draft",
                ("set_value", "type_text", "focus"),
                value=self.value,
                focused=False,
            ),
        ]
        revision = hashlib.sha256(
            json.dumps([self.selected, self.focused, self.value]).encode()
        ).hexdigest()
        return dict(
            session_id="session",
            scope={"kind": "desktop", "id": "scope"},
            revision=revision,
            nodes=nodes,
            text="Workbench\nInput",
        )

    def prepare_step(self, step):
        self.prepared.append(parse_step(step))

    def prepare_intent(self, step):
        self.prepared.append(parse_intent(step))

    def finish_intent(self):
        self.finished += 1

    def boundary_observe(self):
        self.semantic_reads += 1
        if self.after_error:
            raise RuntimeError("controlled read failure")
        return self.observe()

    def perform(self, action):
        self.calls.append(copy.deepcopy(action))
        if self.hook:
            self.hook(self, action)
        if self.mutate:
            if action["verb"] == "set_value":
                self.value = action["args"]["value"]
                self.selected = self.focused = False
            elif action["verb"] == "focus":
                self.focused = True
            else:
                self.selected = self.focused = True
        return copy.deepcopy(self.reply)


def control(adapter=None, *, private=True, **kwargs):
    adapter = adapter or Adapter()
    controller = RapidUiController(
        adapter,
        authorize=lambda *_: True,
        max_observations=1,
        observation_interval_s=0.001,
        **kwargs,
    )
    if private:
        controller.enable_native_task_protocol()
    return controller


def write(value):
    return dict(
        id="write-" + value,
        verb="set_value",
        target={"node_id": "draft"},
        args={"value": value},
        expect=[{"kind": "value", "node_id": "$target", "value": value}],
    )


def store_fixture():
    adapter = Adapter()
    store = ComputerUseStore(None, adapter_factory=lambda *_: adapter)
    cancel = store.begin_task("session", private_boundaries=True)
    return adapter, store, cancel


def boundary(store, cancel, value=None):
    with store.task_scope("session", "desktop", cancel):
        return store.observation_boundary("session", value or intent())


def test_actual_042_goal_effect_is_rejected_but_no_expect_boundary_is_available():
    workbench = "/tmp/qwen-c6506a9-ornith-cpu-livelane-20261016T2005/runtime-data/gpu1-acceptance/long-horizon/ornith-LHN-042-20261016T201328Z/Workbench"
    goal = load_case("LHN-042")["goal_text"].replace("WORKBENCH", workbench)
    supplied = " Read Input/task-brief.txt for the supplied official source starting URLs, declared input inventory and exact delivery/archive requirements."
    expected = goal.removesuffix(supplied)
    assert len(expected) == 1473
    request = ui(ui_step(expect=[{"kind": "text", "text": expected}]))
    validator = Draft202012Validator(
        _grounded_phase_schema({"native_ui"}, {"state": Adapter().observe()})
    )
    assert Draft202012Validator(phase_response_schema({"native_ui"})).is_valid(request)
    assert not validator.is_valid(request)
    assert validator.is_valid({"kind": "ui_boundary", "step": intent()})


@pytest.mark.parametrize("verb", ["click", "double_click", "right_click", "invoke"])
def test_boundary_has_no_predicate_or_following_step_and_preserves_public_schema(verb):
    state = {"state": Adapter().observe()}
    schema = _grounded_phase_schema({"native_ui"}, state)
    validator = Draft202012Validator(schema)
    assert validator.is_valid({"kind": "ui_boundary", "step": intent(verb)})
    for extra in [
        {"expect": [{"kind": "visible", "name": "Input", "visible": True}]},
        {"steps": []},
        {"final_predicates": []},
    ]:
        assert not validator.is_valid({"kind": "ui_boundary", "step": {**intent(verb), **extra}})
    assert not validator.is_valid(
        {"kind": "ui_boundary", "step": intent(verb), "steps": [ui_step()]}
    )
    assert not validator.is_valid(ui(ui_step(verb)))
    assert Draft202012Validator(phase_response_schema({"native_ui"})).is_valid(ui(ui_step(verb)))
    with pytest.raises(ComputerUseError):
        parse_step(intent(verb))
    with pytest.raises(ComputerUseError):
        parse_intent(ui_step(verb))


def test_known_focus_effect_and_first_typing_precondition_keep_future_cross_control_actions():
    state = {"state": Adapter().observe()}
    validator = Draft202012Validator(_grounded_phase_schema({"native_ui"}, state))
    focus = dict(
        id="focus",
        verb="focus",
        target={"node_id": ROW},
        expect=[{"kind": "focused", "node_id": "$target"}],
    )
    future = dict(
        id="future",
        verb="type_text",
        target={"node_id": "future"},
        args={"text": "café\n雪"},
        expect=[{"kind": "value", "node_id": "cross-control", "value": "future"}],
    )
    assert validator.is_valid(ui(focus, future))
    assert not validator.is_valid(
        ui({**focus, "expect": [{"kind": "focused", "node_id": "draft"}]})
    )
    assert not validator.is_valid(ui({**future, "target": {"node_id": "draft"}}))
    state["state"]["nodes"][-1]["focused"] = True
    assert Draft202012Validator(_grounded_phase_schema({"native_ui"}, state)).is_valid(
        ui({**future, "target": {"node_id": "draft"}})
    )


def test_one_dispatch_one_semantic_read_no_predicate_verification_or_credit():
    adapter, store, cancel = store_fixture()
    result = boundary(store, cancel)
    assert len(adapter.calls) == adapter.semantic_reads == adapter.finished == 1
    assert "expect" not in adapter.prepared[0]
    assert result["status"] == "observed_unverified"
    assert result["ui_boundary_receipt"]["input_attempted"] is True
    assert result["effect_receipt"]["effect_state"] == "unknown"
    assert result["new_effect_count"] == 0 and result["task_progress"] is False
    owned = store._owned("session", "desktop")
    assert owned.controller._execution_epoch == 0
    assert not owned.settled_effects
    compact = _compact_result(result)
    changes = {item["id"]: item for item in compact["observed_changes"]["nodes"]}
    assert changes[ROW]["before"]["focused"] is False and changes[ROW]["after"]["focused"] is True
    assert changes[PARENT]["after"]["selected"] == ["Input"]
    assert result["ui_boundary_receipt"]["novelty_key"]


@pytest.mark.parametrize("first_boundary", [True, False])
def test_uncertain_same_semantic_intent_cannot_replay_by_step_effect_or_mode_after_verified_cycle(
    first_boundary,
):
    adapter = Adapter()
    adapter.mutate = False
    c = control(adapter)
    first = c.run_boundary(intent()) if first_boundary else c.run_step(ui_step())
    assert len(adapter.calls) == 1
    adapter.mutate = True
    assert c.run_step(write("B"))["status"] == "verified"
    assert c.run_step(write(""))["status"] == "verified"
    calls = len(adapter.calls)
    for changed in [intent(identity="different-id"), intent("click", "third-id")]:
        assert c.run_boundary(changed) == first
    assert c.run_step(ui_step(identity="false-predicate-alias")) == first
    assert len(adapter.calls) == calls


def test_verified_application_renews_only_after_a_separate_verified_completion():
    adapter = Adapter()
    c = control(adapter)
    request = ui_step(expect=[{"kind": "selected", "node_id": PARENT, "values": ["Input"]}])
    old = c.run_step(request)
    assert old["status"] == "verified"
    adapter.selected = adapter.focused = (
        False  # External-only reset is not a controller-verified transition.
    )
    assert c.run_boundary(intent(identity="alias")) == old
    assert len(adapter.calls) == 1
    assert c.run_step(write("B"))["status"] == "verified"
    assert c.run_step(write(""))["status"] == "verified"
    assert c.run_boundary(intent())["status"] == "observed_unverified"
    assert len(adapter.calls) == 4
    assert c.execute(old["action"]["id"]) == old
    assert len(adapter.calls) == 4


@pytest.mark.parametrize(
    "change", ["generation", "scope", "session", "sensitive", "disabled", "hidden", "revision"]
)
def test_shared_fresh_guard_rejects_without_calling_adapter(change):
    adapter = Adapter()
    c = control(adapter)
    calls = 0
    original = adapter.observe

    def changed():
        nonlocal calls
        calls += 1
        s = original()
        if calls == 2:
            if change in ("generation", "sensitive"):
                s["nodes"][0][change] = "2" if change == "generation" else True
            elif change in ("disabled", "hidden"):
                s["nodes"][0]["enabled" if change == "disabled" else "visible"] = False
            elif change == "scope":
                s["scope"]["id"] = "other"
            elif change == "session":
                s["session_id"] = "other"
            else:
                s["revision"] = "new"
        return s

    adapter.observe = changed
    result = c.run_boundary(intent())
    assert result["status"] == "rejected"
    assert adapter.calls == [] and adapter.semantic_reads == 0


@pytest.mark.parametrize(
    "reply,attempted,may_change",
    [
        ({"status": "rejected"}, None, True),
        ({"status": "rejected", "input_attempted": False}, False, False),
        ({"status": "dispatched"}, True, True),
    ],
)
def test_rejected_status_does_not_deny_possible_pointer_movement(reply, attempted, may_change):
    adapter, store, cancel = store_fixture()
    adapter.reply = reply
    result = boundary(store, cancel)
    assert result["ui_boundary_receipt"]["adapter_called"] is True
    assert result["ui_boundary_receipt"]["input_attempted"] is attempted
    assert result["effect_receipt"]["environment_may_have_changed"] is may_change
    again = boundary(store, cancel, intent(identity="new-id"))
    # The actual state changed in this controlled adapter even on rejection; restore its preimage.
    adapter.selected = adapter.focused = False
    before = len(adapter.calls)
    boundary(store, cancel, intent(identity="alias"))
    assert len(adapter.calls) == before
    assert again["new_effect_count"] == 0


def test_postinput_read_failure_and_cancel_remain_uncertain_and_locked():
    adapter, store, cancel = store_fixture()
    adapter.after_error = True
    first = boundary(store, cancel)
    assert first["ui_boundary_receipt"]["input_attempted"] is True
    assert first["ui_boundary_receipt"]["observed"] is False
    adapter.selected = adapter.focused = False
    adapter.after_error = False
    second = boundary(store, cancel, intent(identity="retry"))
    assert second["ui_boundary_receipt"]["cached"] is True
    assert second["ui_boundary_receipt"]["input_attempted"] is False
    assert len(adapter.calls) == 1
    other = Adapter()
    c = control(other)
    stopped = threading.Event()
    other.hook = lambda *_: stopped.set()
    assert c.run_boundary(intent(), cancel=stopped)["status"] == "cancelled"
    assert other.semantic_reads == 0 and len(other.calls) == 1
    stopped.clear()
    other.selected = other.focused = False
    assert c.run_boundary(intent(identity="later"))["status"] == "cancelled"
    assert len(other.calls) == 1


def test_boundary_requires_trusted_task_and_direct_client_behavior_is_unchanged():
    adapter = Adapter()
    c = control(adapter, private=False)
    with pytest.raises(ComputerUseError):
        c.run_boundary(intent())
    c.run_step(ui_step())
    c.run_step(ui_step(identity="new-id"))
    assert len(adapter.calls) == 2
    _, store, _ = store_fixture()
    with pytest.raises(ComputerUseError):
        store.observation_boundary("session", intent())
    with pytest.raises(ComputerUseError):
        store.act("session", {"operation": "observation_boundary", "surface": "desktop"})


def test_native_private_preparation_uses_shared_validation_and_never_points_on_postread(
    monkeypatch,
):
    adapter = NativeUiAdapter("session", bridge_runner=lambda *_: {})
    adapter.prepare_intent(intent())
    assert "expect" not in adapter._step
    monkeypatch.setattr(adapter, "_semantic_observe", lambda: Adapter().observe())
    monkeypatch.setattr(adapter, "observe", lambda: pytest.fail("prepared pointer observation"))
    assert adapter.boundary_observe()["session_id"] == "session"
    adapter.finish_intent()
    assert adapter._step is adapter._visual is None


def test_changes_redact_protected_clip_values_and_record_omissions_without_ancestry_guess():
    old = Adapter().observe()
    new = copy.deepcopy(old)
    new["nodes"][0]["sensitive"] = True
    new["nodes"][0]["name"] = "private"
    new["nodes"][-1]["value"] = "x" * 1000
    for i in range(30):
        new["nodes"].append(node("n" + str(i), "text", "Added", (), value="x"))
    changed = _boundary_changes(old, new)
    assert CELL not in [x["id"] for x in changed["nodes"]]
    draft = next(x for x in changed["nodes"] if x["id"] == "draft")["after"]
    assert len(draft["value"]) == 500 and draft["value_truncated"] is True
    assert changed["omitted"] > 0 and changed["task_progress"] is False
    assert all("parent_id" not in x for x in changed["nodes"])


def test_phase_loop_preserves_accumulated_failures_and_returns_actual_deltas_to_next_compiler():
    adapter, store, cancel = store_fixture()
    packets = []
    plans = iter(
        [
            ui(ui_step()),
            {"kind": "ui_boundary", "step": intent()},
            {"kind": "ui_boundary", "step": intent(identity="repeat")},
            {"kind": "ui_boundary", "step": intent(identity="repeat-again")},
        ]
    )

    def observe():
        with store.task_scope("session", "desktop", cancel):
            return store.planning_observation("session")

    def compile_phase(packet):
        packets.append(packet)
        return next(plans)

    def execute(_name, args):
        return boundary(store, cancel, args["intent"])

    events = list(
        run_native_goal(
            "Original whole task",
            observe=observe,
            compile_phase=compile_phase,
            execute=execute,
            cancel=lambda: None,
            allowed_tools={"native_ui"},
            max_seconds=1800,
        )
    )
    result = events[-1]["response"]["portal"]["native_task"]
    assert (
        result["status"] == "blocked" and result["phases"] == 4 and result["task_verified"] is False
    )
    assert (
        len(adapter.calls) == 2
    )  # Novel first selection, then one unchanged-state attempt; third is cached.
    assert packets[2]["last_result"]["observed_changes"]["nodes"]
    assert packets[2]["last_result"]["ui_boundary_receipt"]["new_effect_count"] == 0


def test_oversized_observation_or_schema_fallback_excludes_every_input_branch(monkeypatch):
    current = {"state": Adapter().observe()}
    monkeypatch.setattr("portal.ui_task._MAX_GROUNDED_SCHEMA_BYTES", 1)
    schema = _grounded_phase_schema({"native_ui"}, current)
    assert not {"ui", "ui_boundary"} & {x["properties"]["kind"]["const"] for x in schema["oneOf"]}


@pytest.mark.parametrize(
    "bad",
    [
        {"expect": []},
        {"expect": [{"kind": "text", "text": "arbitrary"}]},
        {"authorization": True},
        {"args": {"extra": True}},
        {"verb": "type_text"},
        {"verb": "switch_window"},
        {"target": {"node_id": "x", "coordinates": [1, 2]}},
        {"id": ""},
        {"id": True},
        {"target": {"name": "x" * 513}},
        {"verb": "unknown"},
    ],
)
def test_intent_is_closed_and_does_not_accept_predicate_or_authority_escape(bad):
    with pytest.raises(ComputerUseError):
        parse_intent({**intent(), **bad})


def test_remote_041_root_click_visible_already_true_does_not_imply_focus_or_applied_credit():
    adapter, store, cancel = store_fixture()
    root = node("root", "window", "Workbench", ("click",), focused=False)
    edit = node("edit", "textbox", "Draft", ("focus", "type_text"), focused=False, value="")
    original = adapter.observe

    def state():
        value = original()
        value["nodes"] += [root, edit]
        return value

    adapter.observe = state
    validator = Draft202012Validator(_grounded_phase_schema({"native_ui"}, {"state": state()}))
    wrong = {
        "id": "root-click",
        "verb": "click",
        "target": {"node_id": "root"},
        "expect": [{"kind": "visible", "role": "textbox", "name": "Draft", "visible": True}],
    }
    assert not validator.is_valid(ui(wrong))
    assert not validator.is_valid(
        ui({**wrong, "verb": "focus", "expect": [{"kind": "focused", "node_id": "$target"}]})
    )
    request = {key: value for key, value in wrong.items() if key != "expect"}
    assert validator.is_valid({"kind": "ui_boundary", "step": request})
    result = boundary(store, cancel, request)
    assert result["effect_receipt"]["effect_state"] == "unknown"
    assert result["ui_boundary_receipt"]["verified"] is False
    assert edit["focused"] is False
    assert store._owned("session", "desktop").controller._execution_epoch == 0


def test_unknown_lock_uses_semantic_revision_not_changed_visual_frame_or_knowledge_counts():
    adapter = Adapter()
    adapter.mutate = False
    frame = ["frame-A"]
    count = [0]
    original = adapter.observe

    def observed():
        state = original()
        state["base_semantic_revision"] = state["revision"]
        state["revision"] = frame[0]
        state["nodes"][0]["text_observation"] = {"selection_count": count[0]}
        return state

    adapter.observe = observed
    c = control(adapter)
    previous = c.run_boundary(intent())
    frame[0], count[0] = "frame-B", 1
    assert c.run_boundary(intent(identity="alias")) == previous
    assert c.run_step(ui_step(identity="canonical-alias")) == previous
    assert len(adapter.calls) == 1


def test_shared_authorization_cancellation_and_registry_budget_prevent_input():
    adapter = Adapter()
    c = control(adapter)
    c.authorize = lambda *_: False
    assert c.run_boundary(intent())["status"] == "rejected"
    assert not adapter.calls and not c._intent_keys
    c.authorize = lambda *_: True
    stopped = threading.Event()
    stopped.set()
    assert c.run_boundary(intent(), cancel=stopped)["status"] == "cancelled"
    assert not adapter.calls
    # The earlier exact rejected token stays cached; budget admission uses a
    # fresh controlled epoch rather than pretending authority edits clear it.
    c = control(adapter)
    c._intent_keys = {str(i): "missing" for i in range(1000)}
    assert c.run_boundary(intent())["reason"] == "Native intent budget exhausted"
    assert not adapter.calls


def test_postinput_absolute_deadline_is_unknown_with_no_second_observation(monkeypatch):
    adapter = Adapter()
    c = control(adapter, verification_timeout_s=1)
    ticks = iter([0, 2])
    monkeypatch.setattr("qwen_omni_adapters.computer_use.time.monotonic", lambda: next(ticks))
    result = c.run_boundary(intent())
    assert result["status"] == "unresolved" and len(adapter.calls) == adapter.semantic_reads == 1
    assert c._execution_epoch == 0
    assert result["dispatch_attempted"] is True


def test_pending_input_and_later_aliases_never_dispatch_twice():
    adapter = Adapter()
    c = control(adapter)
    entered, release = threading.Event(), threading.Event()
    adapter.hook = lambda *_: (entered.set(), release.wait(2))
    answers = []
    thread = threading.Thread(target=lambda: answers.append(c.run_boundary(intent())))
    thread.start()
    assert entered.wait(1)
    assert c.run_boundary(intent(identity="concurrent"))["status"] == "unresolved"
    release.set()
    thread.join(2)
    assert not thread.is_alive() and len(adapter.calls) == 1
    assert answers[0]["status"] == "observed_unverified"
    adapter.selected = adapter.focused = False
    assert c.run_boundary(intent(identity="later")) == answers[0]
    assert len(adapter.calls) == 1


def test_canonical_run_step_preserves_public_enumeration_override_and_callback_seam():
    adapter = Adapter()
    c = control(adapter)
    original = c.enumerate
    offered = []

    def observe_offers(step, state=None):
        offered.append(copy.deepcopy((step, state)))
        return original(step, state)

    c.enumerate = observe_offers
    assert c.run_step(write("Exact"))["status"] == "verified"
    assert len(offered) == 1 and offered[0][0] == write("Exact")
    assert offered[0][1]["nodes"][-1]["value"] == ""
    # The private no-expect path never routes through a public step parser.
    assert c.run_boundary(intent())["status"] == "observed_unverified"
    assert len(offered) == 1


@pytest.mark.parametrize("read_failure", [True, False])
def test_boundary_original_failure_reason_reaches_compaction_without_effect_credit(read_failure):
    adapter, store, cancel = store_fixture()
    if read_failure:
        adapter.after_error = True
    else:
        store._owned("session", "desktop").controller.authorize = lambda *_: False
    result = boundary(store, cancel)
    expected = result["dispatch_record"]["reason"]
    compact = _compact_result(result)
    assert compact["reason"] == expected and expected
    assert compact["ui_boundary_receipt"]["verified"] is False
    assert compact["ui_boundary_receipt"]["new_effect_count"] == 0
    assert result["effect_receipt"]["effect_state"] == ("unknown" if read_failure else "absent")


def test_boundary_passes_through_only_actual_finite_policy_receipt():
    adapter, store, cancel = store_fixture()
    original = adapter.observe

    def ambiguous():
        state = original()
        state["nodes"].append(node("other", "table_cell", "Input", ("click",)))
        return state

    adapter.observe = ambiguous
    receipt = {
        "id": "actual-policy",
        "model": "kev-pinned",
        "status": "accepted",
        "reply_sha256": "response",
    }
    calls = []

    class Policy:
        def decide(self, state, actions, *, context):
            calls.append(context)
            return {
                "choice": next(item["id"] for item in actions if item.get("node_id") == CELL),
                "receipt": receipt,
            }

    store.provider = Policy()
    request = {**intent(), "target": {"role": "table_cell", "name": "Input"}}
    result = boundary(store, cancel, request)
    assert result["policy_receipts"] == [receipt] and len(calls) == 1
    assert "expect" not in calls[0]["step"]
    assert _compact_result(result)["policy_receipts"] == [receipt]
