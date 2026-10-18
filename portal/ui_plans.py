"""Immutable task-local planner claims, retrievable without input authority."""
from __future__ import annotations

import copy
import hashlib
import json

from qwen_omni_adapters.computer_use import ComputerUseError, parse_predicates, parse_step


class NativePlanArchive:
    MAX_BYTES = 4 * 1024 * 1024
    FRAGMENT_CHARACTERS = 2048

    def __init__(self, epoch: str, goal_sha256: str):
        self.epoch, self.goal_sha256 = epoch, goal_sha256
        self.records, self.expanded, self.bytes = {}, set(), 0

    def remember(self, phase: int, plan: dict) -> dict:
        if type(phase) is not int or not 1 <= phase <= 40 or not isinstance(plan, dict):
            raise ComputerUseError("Plan archive requires a current bounded source phase")
        steps = plan.get("steps")
        if not isinstance(steps, list) or not 1 <= len(steps) <= 25:
            raise ComputerUseError("Plan archive requires bounded typed instructions")
        for step in steps:
            parse_step(step)
        parse_predicates(plan.get("final_predicates"))
        serialized = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        encoded = serialized.encode()
        content_hash = hashlib.sha256(encoded).hexdigest()
        reference = "plan-" + hashlib.sha256(json.dumps([self.epoch, self.goal_sha256, phase, content_hash]).encode()).hexdigest()[:24]
        if reference not in self.records:
            if len(self.records) >= 40 or self.bytes + len(encoded) > self.MAX_BYTES:
                raise ComputerUseError("Plan archive capacity exhausted; instructions were not discarded")
            self.records[reference] = {"source_phase": phase, "serialized": serialized,
                "content_sha256": content_hash, "step_count": len(steps)}
            self.bytes += len(encoded)
        return self.describe(reference)

    def describe(self, reference: str) -> dict:
        row = self.records[reference]
        return {"reference": reference, "source_phase": row["source_phase"],
            "content_sha256": row["content_sha256"], "step_count": row["step_count"],
            "serialized_characters": len(row["serialized"])}

    def view(self) -> dict:
        return {"schema": "robit.omni.native-plan-index.v1", "authority": "unverified_planner_claim",
            "input_authority": False, "task_lifetime_only": True,
            "fragment_characters": self.FRAGMENT_CHARACTERS,
            "entries": [{key: value for key, value in self.describe(ref).items() if key != "content_sha256"}
                        for ref in self.records]}

    def fragments(self) -> dict[str, tuple[str, int]]:
        return {f"{reference}@{offset}": (reference, offset)
                for reference, row in self.records.items()
                for offset in range(0, len(row["serialized"]), self.FRAGMENT_CHARACTERS)}

    def recall_fragment(self, fragment: str) -> dict:
        choices = self.fragments()
        if not isinstance(fragment, str) or fragment not in choices:
            raise ComputerUseError("Plan fragment is not retained in this native task")
        return self.recall(*choices[fragment])

    def safe_recall(self, reference: str, offset: int) -> dict:
        """Invalid page combinations consume no input and return replanning facts."""
        try:
            return self.recall(reference, offset)
        except ComputerUseError:
            row = self.records.get(reference) if isinstance(reference, str) else None
            return {"status": "unresolved", "error": "native_plan_fragment_unavailable",
                "reason": "Choose a retained plan and one of its available fragment offsets.",
                "dispatch_status": "not_dispatched", "input_attempted": False,
                "task_progress": False, "new_effect_count": 0,
                "evidence_authority": "unverified_historical_planner_claim",
                "plan_recall_receipt": {"schema": "robit.omni.native-plan-recall.v1",
                    "reference": reference if row is not None else None,
                    "available_offsets": list(range(0, len(row["serialized"]), self.FRAGMENT_CHARACTERS)) if row else [],
                    "first_expansion_in_task": False, "input_authority": False,
                    "task_progress": False, "current_ui_verified": False}}

    def recall(self, reference: str, offset: int) -> dict:
        if not isinstance(reference, str) or reference not in self.records or type(offset) is not int:
            raise ComputerUseError("Plan claim is not retained in this native task")
        row = self.records[reference]
        text = row["serialized"]
        if offset < 0 or offset >= len(text) or offset % self.FRAGMENT_CHARACTERS:
            raise ComputerUseError("Plan fragment offset is not a retained boundary")
        end = min(len(text), offset + self.FRAGMENT_CHARACTERS)
        first = (reference, offset) not in self.expanded
        self.expanded.add((reference, offset))
        return {"status": "historical_plan_claim", "dispatch_status": "not_dispatched",
            "input_attempted": False, "task_progress": False, "new_effect_count": 0,
            "evidence_authority": "unverified_historical_planner_claim", "recalled_plan": {
                **copy.deepcopy(self.describe(reference)), "source_epoch": self.epoch,
                "goal_sha256": self.goal_sha256, "offset": offset, "end": end,
                "next_offset": end if end < len(text) else None, "serialized_fragment": text[offset:end],
                "next_fragment": f"{reference}@{end}" if end < len(text) else None,
                "fragment_may_be_incomplete_json": True, "input_authority": False},
            "plan_recall_receipt": {"schema": "robit.omni.native-plan-recall.v1",
                "reference": reference, "offset": offset, "first_expansion_in_task": first,
                "input_authority": False, "task_progress": False, "current_ui_verified": False}}
