"""Execution-contract regressions; real-provider evidence is recorded separately."""
import copy
import json
import threading

import httpx
import pytest
from jsonschema import Draft202012Validator

from portal.app import _model_ui_observation
from portal.computer_use import ComputerUseStore, SystemOneUiProvider
from portal.ui_task import _compact_result, _grounded_phase_schema
from qwen_omni_adapters.computer_use import ComputerUseError, RapidUiController
from qwen_omni_adapters.text_entry import candidates, parse_text_entry, run_text_entry
from runtime.kev_server import validate_request


def intent(text="New value", **extra):
    return {"id": "replace-draft", "family": "text_entry", "mode": "replace",
            "objective": "Replace the draft with the requested text", "target": {"node_id": "draft"},
            "text": text, **extra}


class Editor:
    def __init__(self, value="old draft", *, setter=False):
        self.value, self.setter = value, setter
        self.focused, self.selection = False, None
        self.generation, self.revision = "1", 0
        self.calls = []
        self.hook = self.read_hook = None
        self.truncated = False

    def observe(self):
        if self.read_hook:
            self.read_hook(self)
        return {"session_id": "session", "scope": {"id": "editor", "kind": "native"},
            "revision": str(self.revision), "nodes": [{"id": "draft", "generation": self.generation,
                "scope_id": "editor", "role": "textbox", "name": "Draft", "visible": True, "enabled": True,
                "sensitive": False, "focused": self.focused, "value": self.value,
                "operations": ["set_value", "focus", "select_text", "type_text"],
                "text_observation": {"character_count": len(self.value) + self.truncated,
                    "captured_range": {"start": 0, "end": len(self.value)}, "truncated": self.truncated,
                    "selection_count": int(self.selection is not None)},
                **({"text_selection": self.selection} if self.selection else {})}]}

    def perform(self, action):
        self.calls.append(copy.deepcopy(action))
        verb = action["verb"]
        if verb == "set_value" and self.setter:
            self.value = action["args"]["value"]
        elif verb == "focus":
            self.focused = True
        elif verb == "select_text":
            self.selection = copy.deepcopy(action["args"])
        elif verb == "type_text":
            assert self.focused
            assert not self.value or self.selection == {"start": 0, "end": len(self.value)}
            self.value, self.selection = action["args"]["text"], None
        self.revision += 1
        if self.hook:
            self.hook(self, action)
        return {"status": "dispatched"}


def controller(editor):
    control = RapidUiController(editor, authorize=lambda *_: True, max_observations=2, observation_interval_s=.001)
    control.enable_native_task_protocol()
    return control


def pick(strategy, hook=None):
    def choose(state, bank, history):
        if hook:
            hook()
        return {"choice": next((row["id"] for row in bank if row.get("strategy") == strategy), "text-abstain")}
    return choose


def test_failed_setter_is_excluded_after_new_plan_then_physical_replaces_exactly_once():
    editor = Editor()
    control = controller(editor)
    first = control.run_step({"id": "ineffective-setter", "verb": "set_value", "target": {"node_id": "draft"},
        "args": {"value": "New value"}, "expect": [{"kind": "value", "node_id": "$target", "value": "New value"}]})
    assert first["status"] == "unresolved" and editor.value == "old draft"
    assert [row["verb"] for row in editor.calls] == ["set_value"]
    current = control.observe()
    bank, history = candidates(control, current, intent(id="fresh-narration"))
    assert {row.get("strategy") for row in bank} == {"physical_replace", None}
    assert history[0]["effect_verified"] is False
    second = run_text_entry(control, intent(id="fresh-narration"), pick("physical_replace"))
    assert second["status"] == "complete" and editor.value == "New value"
    assert [row["verb"] for row in editor.calls] == ["set_value", "focus", "select_text", "type_text"]
    assert len(second["outcomes"]) == 3
    assert second["text_entry_receipt"]["goal_verified"] is False
    third = run_text_entry(control, intent(), pick("verify_existing"))
    assert third["status"] == "complete" and not third["outcomes"] and len(editor.calls) == 4


@pytest.mark.parametrize("old,text", [("", "URL"), ("draft", "café\nnaïve"), ("a\nb", "literal\\ntext"),
                                     ("旧内容", "日本語 😀"), ("draft", "x" * 2048)])
def test_physical_replacement_derives_exact_effect_and_preserves_content(old, text):
    editor = Editor(old)
    result = run_text_entry(controller(editor), intent(text), pick("physical_replace"))
    assert result["status"] == "complete" and editor.value == text
    assert result["outcomes"][-1]["action"]["expect"] == [{"kind": "value", "node_id": "draft", "value": text}]
    assert sum(row["verb"] == "type_text" for row in editor.calls) == 1


@pytest.mark.parametrize("fault", ["partial", "transport", "no_effect", "readback"])
def test_uncertain_physical_write_never_retries(fault):
    editor = Editor("")
    control = controller(editor)

    def fail(editor, action):
        if action["verb"] != "type_text":
            return
        if fault == "partial":
            editor.value = "New"
        elif fault == "no_effect":
            editor.value = ""
        elif fault == "transport":
            editor.value = ""
            raise RuntimeError("Delivery unknown")
        else:
            editor.read_hook = lambda _: (_ for _ in ()).throw(RuntimeError("readback disconnected"))
    editor.hook = fail
    result = run_text_entry(control, intent(), pick("physical_replace"))
    assert result["status"] == "needs_replan"
    editor.read_hook = None
    # This new full read reconciles current value; it never causes another write.
    again = run_text_entry(control, intent(), pick("physical_replace"))
    assert again["status"] != "complete"
    assert sum(row["verb"] == "type_text" for row in editor.calls) == 1


@pytest.mark.parametrize("change", ["value", "generation", "focus", "scope"])
def test_selection_cannot_authorize_input_after_operand_change(change):
    editor = Editor()
    def mutate():
        if change == "value":
            editor.value = "someone else's edit"
        elif change == "generation":
            editor.generation = "2"
        elif change == "focus":
            editor.focused = True
        else:
            original = editor.observe
            editor.observe = lambda: {**original(), "scope": {"id": "another-window", "kind": "native"}}
        editor.revision += 1
    result = run_text_entry(controller(editor), intent(), pick("physical_replace", mutate))
    assert result["status"] == "needs_replan" and not editor.calls


def test_drift_after_focus_prevents_selection_and_write():
    editor = Editor()
    editor.hook = lambda editor, action: setattr(editor, "value", "changed") if action["verb"] == "focus" else None
    result = run_text_entry(controller(editor), intent(), pick("physical_replace"))
    assert result["status"] == "needs_replan"
    assert [row["verb"] for row in editor.calls] == ["focus"]


@pytest.mark.parametrize("when", ["before_input", "after_write"])
def test_geometry_guards_input_but_does_not_invalidate_read_only_verification(when):
    editor = Editor()
    original_observe = editor.observe
    changed = False
    def observe():
        state = original_observe()
        state["nodes"][0]["bounds"] = {"x": 28 if changed else 0, "y": 0, "width": 300, "height": 32}
        return state
    def change():
        nonlocal changed
        changed = True
    editor.observe = observe
    if when == "after_write":
        editor.hook = lambda _, action: change() if action["verb"] == "type_text" else None
    result = run_text_entry(controller(editor), intent(), pick("physical_replace", change if when == "before_input" else None))
    assert result["status"] == ("complete" if when == "after_write" else "needs_replan")
    assert len(editor.calls) == (3 if when == "after_write" else 0)


def test_partial_read_cannot_offer_or_verify_replacement():
    editor = Editor("New value")
    editor.truncated = True
    result = run_text_entry(controller(editor), intent(), pick("verify_existing"))
    assert result["status"] == "abstained" and not editor.calls


def test_new_generation_invalidates_failed_strategy_history():
    editor = Editor()
    control = controller(editor)
    control.run_step({"id": "ineffective", "verb": "set_value", "target": {"node_id": "draft"},
        "args": {"value": "New value"}, "expect": [{"kind": "value", "node_id": "$target", "value": "New value"}]})
    editor.generation = "2"
    bank, history = candidates(control, control.observe(), intent())
    assert any(row.get("strategy") == "physical_replace" for row in bank) and history == []


def test_cancel_after_selection_prevents_input():
    editor, cancel = Editor(), threading.Event()
    result = run_text_entry(controller(editor), intent(), pick("physical_replace", cancel.set), cancel)
    assert result["status"] == "cancelled" and not editor.calls


def test_generic_setter_receipt_also_suppresses_the_strategy():
    editor = Editor()
    control = controller(editor)
    control.run_step({"id": "old-plan", "verb": "set_value", "target": {"node_id": "draft"},
                      "args": {"value": "New value"}, "expect": [{"kind": "focused", "node_id": "$target"}]})
    bank, _ = candidates(control, control.observe(), intent())
    assert not any(row.get("strategy") == "native_set" for row in bank)


@pytest.mark.parametrize("extra", [{"mode": "insert"}, {"text": ""}, {"text": "\0"}, {"strategy": "magic"},
                                  {"text": "x" * 2049}, {"objective": " "}])
def test_invalid_or_not_yet_supported_text_modes_are_rejected(extra):
    with pytest.raises(ComputerUseError):
        parse_text_entry(intent(**extra))


def test_planner_schema_and_active_policy_share_the_effect_oriented_family():
    editor, requests = Editor(), []
    original_observe = editor.observe
    def observe():
        state = original_observe()
        state["nodes"].append({**copy.deepcopy(state["nodes"][0]), "id": "title", "name": "Title"})
        return state
    editor.observe = observe
    def respond(request):
        payload = json.loads(request.content)
        validate_request(payload)
        requests.append(payload)
        criteria = payload["questions"]["action"]["criteria"]
        selected = next(key for key, value in criteria.items() if json.loads(value).get("node_id") == "draft")
        return httpx.Response(200, json={"model": provider.model, "answers": {"action": {
            "choice": selected, "probabilities": {key: float(key == selected) for key in criteria}}}})
    provider = SystemOneUiProvider("http://127.0.0.1:1", mode="active", client=httpx.Client(transport=httpx.MockTransport(respond)))
    store = ComputerUseStore(None, provider=provider, adapter_factory=lambda *_: editor)
    cancel = store.begin_task("session", private_boundaries=True)
    schema = _grounded_phase_schema(["native_ui"], {"state": editor.observe()}, intent_choice=True)
    Draft202012Validator(schema).validate({"kind": "ui_subgoal", "subgoal": intent()})
    with store.task_scope("session", "desktop", cancel):
        result = store.intent_boundary("session", intent(target={"role": "textbox"}), goal="Replace my draft with New value")
    assert result["status"] == "complete" and len(requests) == 1
    assert result["policy_receipts"][0]["model_confirmed"]
    assert _compact_result(result)["text_entry_receipt"]["exact_value_verified"]
    with pytest.raises(ComputerUseError, match="binding"):
        store.text_entry("session", intent(), goal="Replace my draft")


def test_text_family_survives_actual_budgeted_projection_without_relaxing_executor_coverage():
    editor = Editor()
    state = editor.observe()
    node = state["nodes"][0]
    node["description"] = ""
    compact = {key: value for key, value in node.items() if key != "text_observation"}
    node["description"] = "d" * (12000 - len(json.dumps(compact, ensure_ascii=False)))
    projected = _model_ui_observation(state)
    assert "text_observation" not in projected["nodes"][0]
    assert sum(len(json.dumps(row, ensure_ascii=False)) for row in projected["nodes"]) == 12000
    schema = _grounded_phase_schema(["native_ui"], {"state": projected}, intent_choice=True)
    Draft202012Validator(schema).validate({"kind": "ui_subgoal", "subgoal": intent()})
    editor.truncated = True
    result = run_text_entry(controller(editor), intent(), pick("physical_replace"))
    assert result["status"] == "abstained" and not editor.calls


@pytest.mark.parametrize("change", ["hidden", "disabled", "sensitive", "no_write_capability"])
def test_projection_discovery_does_not_invent_editability(change):
    state = Editor().observe()
    node = state["nodes"][0]
    del node["text_observation"]
    if change == "hidden":
        node["visible"] = False
    elif change == "disabled":
        node["enabled"] = False
    elif change == "sensitive":
        node["sensitive"] = True
    else:
        node["operations"] = ["focus"]
    schema = _grounded_phase_schema(["native_ui"], {"state": state}, intent_choice=True)
    assert not Draft202012Validator(schema).is_valid({"kind": "ui_subgoal", "subgoal": intent()})
