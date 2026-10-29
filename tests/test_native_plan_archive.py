"""Entire unexecuted plans remain task-local, immutable and expandable."""
import copy
import json

import pytest
from jsonschema import Draft202012Validator
from test_native_working_state import envelope
from test_ui_task import phase

from portal.ui_plans import NativePlanArchive
from portal.ui_task import _NativeWorkingState, _working_phase_schema, phase_response_schema
from qwen_omni_adapters.computer_use import ComputerUseError


def long_plan():
    plan = phase()
    for index in range(2, 25):
        step = copy.deepcopy(plan["steps"][0])
        step.update(id=f"retained-step-{index}", args={"value": f"exact-unexecuted-{index}:" + "界" * 100})
        plan["steps"].append(step)
    return plan


def test_every_byte_of_all25_instructions_and_terminal_claim_is_retrievable():
    archive = NativePlanArchive("e" * 32, "a" * 64)
    plan = long_plan()
    expected = copy.deepcopy(plan)
    reference = archive.remember(1, plan)["reference"]
    plan["steps"][23]["args"]["value"] = "mutated outside archive"
    text = ""
    fragment = reference + "@0"
    while fragment is not None:
        result = archive.recall_fragment(fragment)
        assert result["input_attempted"] is False and result["task_progress"] is False
        assert result["plan_recall_receipt"]["current_ui_verified"] is False
        body = result["recalled_plan"]
        assert len(body["serialized_fragment"]) <= archive.FRAGMENT_CHARACTERS
        text += body["serialized_fragment"]
        fragment = body["next_fragment"]
    assert json.loads(text) == expected
    assert archive.recall_fragment(reference + "@0")["plan_recall_receipt"]["first_expansion_in_task"] is False


def test_reference_is_bound_to_epoch_and_goal_with_no_cross_task_read():
    first = NativePlanArchive("a" * 32, "b" * 64)
    ref = first.remember(1, phase())["reference"]
    for epoch, goal in [("c" * 32, "b" * 64), ("a" * 32, "d" * 64)]:
        other = NativePlanArchive(epoch, goal)
        assert other.remember(1, phase())["reference"] != ref
        with pytest.raises(ComputerUseError):
            other.recall_fragment(ref + "@0")


def test_capacity_failure_never_evicts_old_plan_or_overwrites_it():
    archive = NativePlanArchive("e" * 32, "a" * 64)
    ref = archive.remember(1, phase())["reference"]
    archive.MAX_BYTES = archive.bytes
    with pytest.raises(ComputerUseError, match="capacity"):
        archive.remember(2, long_plan())
    assert archive.recall_fragment(ref + "@0")["recalled_plan"]["source_phase"] == 1
    assert len(archive.records) == 1


def test_unavailable_cross_length_offset_returns_bounded_replan_without_input():
    working = _NativeWorkingState("Exact original user goal")
    short = working.plans.remember(1, phase())
    working.plans.remember(2, long_plan())
    state = working.view(3, 0)
    schema = _working_phase_schema({"native_ui"}, phase_response_schema({"native_ui"}), state)
    validator = Draft202012Validator(schema)
    branch = next(b for b in schema["$defs"]["native_working_phase"]["oneOf"]
                  if b["properties"]["kind"]["const"] == "recall_plan")
    offsets = branch["properties"]["offset"]["enum"]
    assert len(offsets) > 2
    unavailable = {"kind": "recall_plan", "reference": short["reference"], "offset": offsets[-1]}
    assert validator.is_valid(envelope({"working_state": state}, unavailable))
    result = working.plans.safe_recall(short["reference"], offsets[-1])
    assert result["error"] == "native_plan_fragment_unavailable"
    assert result["input_attempted"] is False and result["task_progress"] is False
    assert result["plan_recall_receipt"]["available_offsets"] == [0]
    assert working.plans.safe_recall(short["reference"], 0)["status"] == "historical_plan_claim"


def test_25_large_legal_plans_do_not_expand_grammar_past_its_original_budget():
    working = _NativeWorkingState("Exact original user goal")
    plan = long_plan()
    for step in plan["steps"]:
        if step["verb"] == "set_value":
            step["args"]["value"] = "x" * 4000
    for phase_index in range(1, 26):
        working.plans.remember(phase_index, plan)
        state = working.view(phase_index, 0)
        schema = _working_phase_schema({"native_ui"}, phase_response_schema({"native_ui"}), state)
        assert len(json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode()) <= 56 * 1024


@pytest.mark.parametrize("offset", [-1, True, 1, 10**8])
def test_invalid_fragment_boundaries_fail_without_input(offset):
    archive = NativePlanArchive("e" * 32, "a" * 64)
    ref = archive.remember(1, phase())["reference"]
    with pytest.raises(ComputerUseError):
        archive.recall(ref, offset)
