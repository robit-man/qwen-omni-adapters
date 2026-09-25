"""Deterministic adversarial evaluation for bounded virtual context.

The benchmark deliberately separates *memory-subsystem preparation* from
language-model answer quality.  Expected answers are used only to score the
resulting working set (and to locate source chunks for the labelled oracle);
they are never appended to production retrieval queries.
"""

from __future__ import annotations

import random
import resource
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qwen_omni_adapters.virtual_memory.controller import (
    ControllerConfig,
    RecursiveMemoryController,
)
from qwen_omni_adapters.virtual_memory.embedding import HashingEmbedder
from qwen_omni_adapters.virtual_memory.engine import select_relevant_memories
from qwen_omni_adapters.virtual_memory.extractor import StructuredMemoryExtractor
from qwen_omni_adapters.virtual_memory.models import RetrievalHit, WorkingContext
from qwen_omni_adapters.virtual_memory.packer import (
    ContextBudget,
    WorkingContextPacker,
    conservative_token_estimate,
)
from qwen_omni_adapters.virtual_memory.retrieval import HybridRetriever
from qwen_omni_adapters.virtual_memory.store import ImmutableEvidenceStore

MATRIX_SCHEMA = "robit.virtual-context-benchmark-matrix.v1"
DEFAULT_SOURCE_LENGTHS = (
    16_000,
    32_000,
    64_000,
    128_000,
    256_000,
    512_000,
    1_000_000,
)
BASELINES = (
    "fifo",
    "dense_rag",
    "lexical_rag",
    "hybrid_rag",
    "recurrent_textual",
    "recursive_replay",
    "structured_replay",
    "oracle",
)
PRODUCTION_BASELINE = "structured_replay"


@dataclass(frozen=True)
class BenchmarkDocument:
    source: str
    text: str
    position: float
    kind: str = "document"
    media_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    entities: tuple[str, ...] = ()
    relationships: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    family: str
    query: str
    required_terms: tuple[str, ...]
    oracle_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...] = ()
    active_memory_subject: str | None = None
    active_memory_term: str | None = None


@dataclass(frozen=True)
class PreparedCorpus:
    documents: tuple[BenchmarkDocument, ...]
    scenarios: tuple[BenchmarkScenario, ...]
    source_text: str
    source_tokens: int
    seed: int | None = None


@dataclass(frozen=True)
class BaselineMeasurement:
    source_tokens: int
    scenario: str
    family: str
    baseline: str
    required_terms: int
    evidence_terms_found: int
    replay_terms_found: int
    evidence_recall: float
    replay_recall: float
    provenance_accuracy: float
    distractor_terms_replayed: int
    resident_input_tokens: int
    retrieved_evidence_tokens: int
    compression_ratio: float
    retrieval_rounds: int
    sufficient: bool
    active_memory_correct: bool | None
    latency_seconds: float
    passed: bool


def _word_tokens(value: str) -> int:
    return len(value.split())


def _documents() -> list[BenchmarkDocument]:
    """High-signal sources distributed through otherwise irrelevant history."""

    return [
        BenchmarkDocument(
            "conversation://constraint/1",
            "MUST NOT replace the golden controller configuration.",
            0.01,
            kind="conversation",
        ),
        BenchmarkDocument(
            "notes://single-needle",
            "Calibration ledger: NEEDLE_ID_7F3A has exact value Quartz-41927.",
            0.06,
        ),
        BenchmarkDocument(
            "conversation://preference/1",
            "I chose to use telemetry_format JSONL for durable event exports.",
            0.10,
            kind="conversation",
        ),
        BenchmarkDocument(
            "config://motor-controller/v1",
            "motor_controller is configured as amber-17.",
            0.15,
        ),
        BenchmarkDocument(
            "notes://multiple-needle/alpha",
            "Survey key MULTI_ALPHA records Aster-104.",
            0.20,
        ),
        BenchmarkDocument(
            "src://drive/config.py",
            """class DriveConfig:\n    CAN_TIMEOUT_MS = 275\n""",
            0.24,
            kind="code",
            media_type="text/x-python",
        ),
        BenchmarkDocument(
            "graph://dropbear/1",
            "Dropbear uses left_leg.",
            0.28,
            entities=("Dropbear", "left_leg"),
            relationships=(("Dropbear", "uses", "left_leg"),),
        ),
        BenchmarkDocument(
            "src://drive/validator.py",
            """class ChecksumMismatch(Exception):\n    pass\n\ndef validate_frame(frame):\n    if not frame.checksum_ok:\n        raise ChecksumMismatch(\"E_CHECKSUM_4D2A\")\n""",
            0.31,
            kind="code",
            media_type="text/x-python",
        ),
        BenchmarkDocument(
            "notes://multiple-needle/beta",
            "Survey key MULTI_BETA records Birch-208.",
            0.34,
        ),
        BenchmarkDocument(
            "graph://dropbear/2",
            "left_leg uses AS5600.",
            0.37,
            entities=("left_leg", "AS5600"),
            relationships=(("left_leg", "uses", "AS5600"),),
        ),
        BenchmarkDocument(
            "src://drive/bus.py",
            """from .config import DriveConfig\nfrom .validator import validate_frame\n\ndef send_frame(frame):\n    validate_frame(frame)\n    return DriveConfig.CAN_TIMEOUT_MS\n""",
            0.40,
            kind="code",
            media_type="text/x-python",
        ),
        BenchmarkDocument(
            "notes://multiple-needle/gamma",
            "Survey key MULTI_GAMMA records Cedar-312.",
            0.45,
        ),
        BenchmarkDocument(
            "graph://dropbear/3",
            "AS5600 -> CAN_42; CAN42_BITRATE_BPS=1000000.",
            0.48,
            entities=("AS5600", "CAN_42"),
            relationships=(("AS5600", "communicates_through", "CAN_42"),),
        ),
        BenchmarkDocument(
            "tests://drive/test_bus.py",
            """def test_send_frame_rejects_bad_checksum():\n    # send_frame must surface ChecksumMismatch for E_CHECKSUM_4D2A.\n    pass\n""",
            0.51,
            kind="code",
            media_type="text/x-python",
        ),
        BenchmarkDocument(
            "graph://dropbear/4",
            "CAN_42 has bitrate CAN42_BITRATE_BPS=1000000.",
            0.54,
            entities=("CAN_42",),
        ),
        BenchmarkDocument(
            "power://atlas",
            "Power ledger states NUM_ATLAS_W=17.",
            0.57,
        ),
        BenchmarkDocument(
            "power://boreal",
            "Power ledger states NUM_BOREAL_W=23.",
            0.59,
        ),
        BenchmarkDocument(
            "power://cygnus",
            "Power ledger states NUM_CYGNUS_W=31.",
            0.61,
        ),
        BenchmarkDocument(
            "config://motor-controller/v2",
            "motor_controller changed from amber-17 to violet-29.",
            0.64,
        ),
        BenchmarkDocument(
            "config://dropbear-left/v1",
            "dropbear_left_controller is configured as DBL-OLD-118.",
            0.66,
        ),
        BenchmarkDocument(
            "config://dropbear-right",
            "dropbear_right_controller is configured as DBR-742.",
            0.67,
        ),
        BenchmarkDocument(
            "config://dropbearr-left-decoy",
            "dropbearr_left_controller is configured as DBL-DECOY-991.",
            0.68,
        ),
        BenchmarkDocument(
            "config://dropbear-left/v2",
            "dropbear_left_controller changed from DBL-OLD-118 to DBL-NEW-552.",
            0.70,
        ),
        BenchmarkDocument(
            "logs://thermal/request-88",
            "2037-04-03T12:44:09Z request=req-88 fault=E_THERMAL_LOCK_8A77 zone=gantry.",
            0.73,
            kind="log",
        ),
        BenchmarkDocument(
            "conversation://open-question/1",
            "Open question: verify whether the auxiliary encoder is shielded.",
            0.75,
            kind="conversation",
        ),
    ]


def _scenarios() -> tuple[BenchmarkScenario, ...]:
    return (
        BenchmarkScenario(
            "single_needle",
            "sparse_retrieval",
            "What exact value is recorded for NEEDLE_ID_7F3A?",
            ("Quartz-41927",),
            ("NEEDLE_ID_7F3A",),
        ),
        BenchmarkScenario(
            "multiple_needles",
            "multi_source_retrieval",
            "Return the recorded values for MULTI_ALPHA, MULTI_BETA, and MULTI_GAMMA.",
            ("Aster-104", "Birch-208", "Cedar-312"),
            ("MULTI_ALPHA", "MULTI_BETA", "MULTI_GAMMA"),
        ),
        BenchmarkScenario(
            "chronology_and_supersession",
            "temporal_contradiction",
            "What is the current motor_controller value and what value did it replace?",
            ("amber-17", "violet-29"),
            ("motor_controller is configured", "motor_controller changed"),
            active_memory_subject="motor_controller",
            active_memory_term="violet-29",
        ),
        BenchmarkScenario(
            "numerical_aggregation_evidence",
            "numerical_aggregation",
            "Find NUM_ATLAS_W, NUM_BOREAL_W, and NUM_CYGNUS_W so their total can be computed.",
            ("NUM_ATLAS_W=17", "NUM_BOREAL_W=23", "NUM_CYGNUS_W=31"),
            ("NUM_ATLAS_W", "NUM_BOREAL_W", "NUM_CYGNUS_W"),
        ),
        BenchmarkScenario(
            "multi_hop_entity_graph",
            "multi_hop",
            "What exact numeric setting does Dropbear ultimately use through its left_leg sensor path?",
            ("left_leg", "AS5600", "CAN_42", "CAN42_BITRATE_BPS=1000000"),
            ("Dropbear uses", "left_leg uses", "AS5600 -> CAN_42", "CAN42_BITRATE"),
        ),
        BenchmarkScenario(
            "cross_file_symbol_trace",
            "code_dependency",
            "Trace function `send_frame` through `validate_frame` to error `ChecksumMismatch`, including the relevant test and timeout constant.",
            (
                "send_frame",
                "validate_frame",
                "ChecksumMismatch",
                "test_send_frame_rejects_bad_checksum",
                "CAN_TIMEOUT_MS = 275",
            ),
            (
                "def send_frame",
                "def validate_frame",
                "test_send_frame_rejects_bad_checksum",
                "CAN_TIMEOUT_MS = 275",
            ),
        ),
        BenchmarkScenario(
            "buried_constraint",
            "constraint_persistence",
            "Replace the controller configuration with a convenient local default.",
            ("MUST NOT replace the golden controller configuration",),
            ("MUST NOT replace the golden controller configuration",),
        ),
        BenchmarkScenario(
            "user_decision",
            "preference_and_decision",
            "Which telemetry_format did the user choose for durable event exports?",
            ("telemetry_format", "JSONL"),
            ("telemetry_format JSONL",),
        ),
        BenchmarkScenario(
            "exact_log_string",
            "exact_string_and_citation",
            "Which request emitted exact fault string `E_THERMAL_LOCK_8A77`?",
            ("req-88", "E_THERMAL_LOCK_8A77"),
            ("E_THERMAL_LOCK_8A77",),
        ),
        BenchmarkScenario(
            "near_duplicate_conflict",
            "adversarial_conflict",
            "What is the current dropbear_left_controller and what did it supersede?",
            ("DBL-OLD-118", "DBL-NEW-552"),
            ("dropbear_left_controller is configured", "dropbear_left_controller changed"),
            forbidden_terms=("DBR-742", "DBL-DECOY-991"),
            active_memory_subject="dropbear_left_controller",
            active_memory_term="DBL-NEW-552",
        ),
    )


def _randomized_fixture(
    seed: int,
) -> tuple[list[BenchmarkDocument], tuple[BenchmarkScenario, ...]]:
    """Create one deterministic, previously unseen domain-evaluation variant.

    Seeded variants change both the address vocabulary and the payload values.
    Only the task shapes remain stable.  Expected values are returned solely in
    ``BenchmarkScenario`` evaluator fields and follow the same no-leak boundary
    as the original development fixture.
    """

    rng = random.Random(int(seed))

    def token(prefix: str, bits: int = 24) -> str:
        return f"{prefix}_{rng.getrandbits(bits):0{bits // 4}X}"

    def value(prefix: str) -> str:
        return f"{prefix}-{rng.randrange(10_000, 99_999)}"

    suffix = f"{rng.getrandbits(16):04x}"
    needle_key = token("NEEDLE")
    needle_value = value("Mineral")
    needle_decoy_key = f"{needle_key[:-1]}{rng.choice('ABCDEF')}"
    if needle_decoy_key == needle_key:
        needle_decoy_key = f"{needle_key[:-1]}0"
    needle_decoy_value = value("Decoy")

    multi_keys = tuple(token(f"SURVEY_{name}") for name in ("A", "B", "C"))
    multi_values = tuple(value(name) for name in ("Lumen", "Mica", "Nimbus"))
    multi_decoy_key = f"{multi_keys[0]}_ARCHIVE"
    multi_decoy_value = value("Stale")

    controller_subject = f"motor_controller_{suffix}"
    controller_old = value("MC-OLD")
    controller_new = value("MC-NEW")

    telemetry_key = f"telemetry_format_{suffix}"
    telemetry_value = rng.choice(("CBOR", "JSONL", "MessagePack", "Parquet"))

    emit_fn = f"emit_packet_{suffix}"
    verify_fn = f"verify_packet_{suffix}"
    error_type = f"IntegrityFault{rng.getrandbits(16):04X}"
    error_code = token("E_FRAME")
    timeout_name = f"PACKET_TIMEOUT_MS_{suffix.upper()}"
    timeout_value = rng.randrange(180, 980)
    test_name = f"test_{emit_fn}_surfaces_{error_type.lower()}"

    root_entity = f"Rover{rng.getrandbits(12):03X}"
    limb_entity = f"left_arm_{suffix}"
    sensor_entity = f"Encoder{rng.getrandbits(12):03X}"
    bus_entity = f"BUS_{rng.randrange(20, 90)}_{suffix.upper()}"
    bitrate_key = f"{bus_entity}_RATE_BPS"
    bitrate_value = rng.choice((250_000, 500_000, 800_000, 1_000_000, 2_000_000))
    graph_decoy_value = rng.choice(
        tuple(
            value
            for value in (125_000, 333_000, 666_000, 1_500_000)
            if value != bitrate_value
        )
    )

    numeric_keys = tuple(token(f"POWER_{name}", bits=16) for name in ("A", "B", "C"))
    numeric_values = tuple(rng.sample(range(11, 89), 3))

    conflict_subject = f"{root_entity.casefold()}_left_controller_{suffix}"
    conflict_right = f"{root_entity.casefold()}_right_controller_{suffix}"
    conflict_typo = f"{root_entity.casefold()}x_left_controller_{suffix}"
    conflict_old = value("LEFT-OLD")
    conflict_new = value("LEFT-NEW")
    conflict_right_value = value("RIGHT")
    conflict_decoy_value = value("LEFT-DECOY")

    request_id = f"req-{rng.randrange(100, 999)}-{suffix}"
    fault_code = token("E_THERMAL")
    decoy_request_id = f"req-{rng.randrange(100, 999)}-{suffix}"
    while decoy_request_id == request_id:
        decoy_request_id = f"req-{rng.randrange(100, 999)}-{suffix}"
    decoy_fault_code = f"{fault_code}_ARCHIVE"

    labels = (
        "single",
        "single_decoy",
        "decision",
        "motor_v1",
        "motor_v2",
        "multi_a",
        "multi_b",
        "multi_c",
        "multi_decoy",
        "code_config",
        "code_validator",
        "code_bus",
        "code_test",
        "graph_1",
        "graph_2",
        "graph_3",
        "graph_4",
        "graph_decoy",
        "power_a",
        "power_b",
        "power_c",
        "conflict_v1",
        "conflict_right",
        "conflict_typo",
        "conflict_v2",
        "log",
        "log_decoy",
        "open",
    )
    slots = rng.sample(range(25, 850), len(labels))
    positions = {label: slot / 1_000 for label, slot in zip(labels, slots, strict=True)}
    for older, newer in (("motor_v1", "motor_v2"), ("conflict_v1", "conflict_v2")):
        if positions[older] > positions[newer]:
            positions[older], positions[newer] = positions[newer], positions[older]

    metadata = {"held_out_seed": int(seed)}
    documents = [
        BenchmarkDocument(
            f"conversation://constraint/{suffix}",
            f"MUST NOT replace the golden controller configuration {suffix}.",
            0.005,
            kind="conversation",
            metadata=metadata,
        ),
        BenchmarkDocument(
            f"notes://needle/{suffix}",
            f"Calibration ledger: {needle_key} has exact value {needle_value}.",
            positions["single"],
            metadata=metadata,
        ),
        BenchmarkDocument(
            f"notes://needle-decoy/{suffix}",
            f"Archived calibration ledger: {needle_decoy_key} has obsolete value {needle_decoy_value}.",
            positions["single_decoy"],
            metadata={**metadata, "distractor": True},
        ),
        BenchmarkDocument(
            f"conversation://preference/{suffix}",
            f"I chose to use {telemetry_key} {telemetry_value} for durable event exports.",
            positions["decision"],
            kind="conversation",
            metadata=metadata,
        ),
        BenchmarkDocument(
            f"config://motor/{suffix}/v1",
            f"{controller_subject} is configured as {controller_old}.",
            positions["motor_v1"],
            metadata=metadata,
        ),
        BenchmarkDocument(
            f"config://motor/{suffix}/v2",
            f"{controller_subject} changed from {controller_old} to {controller_new}.",
            positions["motor_v2"],
            metadata=metadata,
        ),
        *[
            BenchmarkDocument(
                f"notes://survey/{suffix}/{index}",
                f"Survey key {key} records {recorded}.",
                positions[f"multi_{'abc'[index]}"],
                metadata=metadata,
            )
            for index, (key, recorded) in enumerate(zip(multi_keys, multi_values, strict=True))
        ],
        BenchmarkDocument(
            f"notes://survey-decoy/{suffix}",
            f"Archived survey key {multi_decoy_key} records {multi_decoy_value}.",
            positions["multi_decoy"],
            metadata={**metadata, "distractor": True},
        ),
        BenchmarkDocument(
            f"src://packet_{suffix}/config.py",
            f"class PacketConfig:\n    {timeout_name} = {timeout_value}\n",
            positions["code_config"],
            kind="code",
            media_type="text/x-python",
            metadata=metadata,
        ),
        BenchmarkDocument(
            f"src://packet_{suffix}/validator.py",
            (
                f"class {error_type}(Exception):\n    pass\n\n"
                f"def {verify_fn}(packet):\n"
                "    if not packet.integrity_ok:\n"
                f"        raise {error_type}(\"{error_code}\")\n"
            ),
            positions["code_validator"],
            kind="code",
            media_type="text/x-python",
            metadata=metadata,
        ),
        BenchmarkDocument(
            f"src://packet_{suffix}/bus.py",
            (
                "from .config import PacketConfig\n"
                f"from .validator import {verify_fn}\n\n"
                f"def {emit_fn}(packet):\n"
                f"    {verify_fn}(packet)\n"
                f"    return PacketConfig.{timeout_name}\n"
            ),
            positions["code_bus"],
            kind="code",
            media_type="text/x-python",
            metadata=metadata,
        ),
        BenchmarkDocument(
            f"tests://packet_{suffix}/test_bus.py",
            (
                f"def {test_name}():\n"
                f"    # {emit_fn} must surface {error_type} for {error_code}.\n"
                "    pass\n"
            ),
            positions["code_test"],
            kind="code",
            media_type="text/x-python",
            metadata=metadata,
        ),
        BenchmarkDocument(
            f"graph://{suffix}/1",
            f"{root_entity} uses {limb_entity}.",
            positions["graph_1"],
            metadata=metadata,
            entities=(root_entity, limb_entity),
            relationships=((root_entity, "uses", limb_entity),),
        ),
        BenchmarkDocument(
            f"graph://{suffix}/2",
            f"{limb_entity} uses {sensor_entity}.",
            positions["graph_2"],
            metadata=metadata,
            entities=(limb_entity, sensor_entity),
            relationships=((limb_entity, "uses", sensor_entity),),
        ),
        BenchmarkDocument(
            f"graph://{suffix}/3",
            f"{sensor_entity} communicates through {bus_entity}.",
            positions["graph_3"],
            metadata=metadata,
            entities=(sensor_entity, bus_entity),
            relationships=((sensor_entity, "communicates_through", bus_entity),),
        ),
        BenchmarkDocument(
            f"graph://{suffix}/4",
            f"{bus_entity} has exact setting {bitrate_key}={bitrate_value}.",
            positions["graph_4"],
            metadata=metadata,
            entities=(bus_entity, bitrate_key),
            relationships=((bus_entity, "has_setting", bitrate_key),),
        ),
        BenchmarkDocument(
            f"graph://{suffix}/decoy",
            (
                f"{root_entity}_archive used retired bus {bus_entity}_OLD with "
                f"{bitrate_key}_OLD={graph_decoy_value}."
            ),
            positions["graph_decoy"],
            metadata={**metadata, "distractor": True},
            entities=(f"{root_entity}_archive", f"{bus_entity}_OLD"),
        ),
        *[
            BenchmarkDocument(
                f"power://{suffix}/{index}",
                f"Power ledger states {key}={number}.",
                positions[f"power_{'abc'[index]}"],
                metadata=metadata,
            )
            for index, (key, number) in enumerate(zip(numeric_keys, numeric_values, strict=True))
        ],
        BenchmarkDocument(
            f"config://conflict/{suffix}/v1",
            f"{conflict_subject} is configured as {conflict_old}.",
            positions["conflict_v1"],
            metadata=metadata,
        ),
        BenchmarkDocument(
            f"config://conflict/{suffix}/right",
            f"{conflict_right} is configured as {conflict_right_value}.",
            positions["conflict_right"],
            metadata={**metadata, "distractor": True},
        ),
        BenchmarkDocument(
            f"config://conflict/{suffix}/typo",
            f"{conflict_typo} is configured as {conflict_decoy_value}.",
            positions["conflict_typo"],
            metadata={**metadata, "distractor": True},
        ),
        BenchmarkDocument(
            f"config://conflict/{suffix}/v2",
            f"{conflict_subject} changed from {conflict_old} to {conflict_new}.",
            positions["conflict_v2"],
            metadata=metadata,
        ),
        BenchmarkDocument(
            f"logs://thermal/{request_id}",
            f"2038-06-11T09:14:27Z request={request_id} fault={fault_code} zone=gantry.",
            positions["log"],
            kind="log",
            metadata=metadata,
        ),
        BenchmarkDocument(
            f"logs://thermal/{decoy_request_id}",
            (
                f"2037-02-04T08:01:02Z request={decoy_request_id} "
                f"fault={decoy_fault_code} status=retired."
            ),
            positions["log_decoy"],
            kind="log",
            metadata={**metadata, "distractor": True},
        ),
        BenchmarkDocument(
            f"conversation://open-question/{suffix}",
            "Open question: verify whether the auxiliary encoder is shielded.",
            positions["open"],
            kind="conversation",
            metadata=metadata,
        ),
    ]

    scenarios = (
        BenchmarkScenario(
            "single_needle",
            "sparse_retrieval",
            f"What exact value is recorded for {needle_key}?",
            (needle_value,),
            (needle_key,),
            forbidden_terms=(needle_decoy_value,),
        ),
        BenchmarkScenario(
            "multiple_needles",
            "multi_source_retrieval",
            f"Return the recorded values for {multi_keys[0]}, {multi_keys[1]}, and {multi_keys[2]}.",
            multi_values,
            multi_keys,
            forbidden_terms=(multi_decoy_value,),
        ),
        BenchmarkScenario(
            "chronology_and_supersession",
            "temporal_contradiction",
            f"What is the current {controller_subject} value and what value did it replace?",
            (controller_old, controller_new),
            (f"{controller_subject} is configured", f"{controller_subject} changed"),
            active_memory_subject=controller_subject,
            active_memory_term=controller_new,
        ),
        BenchmarkScenario(
            "numerical_aggregation_evidence",
            "numerical_aggregation",
            f"Find {numeric_keys[0]}, {numeric_keys[1]}, and {numeric_keys[2]} so their total can be computed.",
            tuple(f"{key}={number}" for key, number in zip(numeric_keys, numeric_values, strict=True)),
            numeric_keys,
        ),
        BenchmarkScenario(
            "multi_hop_entity_graph",
            "multi_hop",
            f"What exact numeric setting does {root_entity} ultimately use through its {limb_entity} sensor path?",
            (limb_entity, sensor_entity, bus_entity, f"{bitrate_key}={bitrate_value}"),
            (
                f"{root_entity} uses",
                f"{limb_entity} uses",
                f"{sensor_entity} communicates through",
                bitrate_key,
            ),
            forbidden_terms=(str(graph_decoy_value),),
        ),
        BenchmarkScenario(
            "cross_file_symbol_trace",
            "code_dependency",
            (
                f"Trace function `{emit_fn}` through `{verify_fn}` to error `{error_type}`, "
                "including the relevant test and timeout constant."
            ),
            (
                emit_fn,
                verify_fn,
                error_type,
                test_name,
                f"{timeout_name} = {timeout_value}",
                error_code,
            ),
            (
                f"def {emit_fn}",
                f"def {verify_fn}",
                f"def {test_name}",
                f"{timeout_name} = {timeout_value}",
            ),
        ),
        BenchmarkScenario(
            "buried_constraint",
            "constraint_persistence",
            f"Replace the golden controller configuration {suffix} with a convenient local default.",
            (f"MUST NOT replace the golden controller configuration {suffix}",),
            (f"MUST NOT replace the golden controller configuration {suffix}",),
        ),
        BenchmarkScenario(
            "user_decision",
            "preference_and_decision",
            f"Which {telemetry_key} did the user choose for durable event exports?",
            (telemetry_key, telemetry_value),
            (f"{telemetry_key} {telemetry_value}",),
        ),
        BenchmarkScenario(
            "exact_log_string",
            "exact_string_and_citation",
            f"Which request emitted exact fault string `{fault_code}`?",
            (request_id, fault_code),
            (fault_code,),
            forbidden_terms=(decoy_request_id,),
        ),
        BenchmarkScenario(
            "near_duplicate_conflict",
            "adversarial_conflict",
            f"What is the current {conflict_subject} and what did it supersede?",
            (conflict_old, conflict_new),
            (f"{conflict_subject} is configured", f"{conflict_subject} changed"),
            forbidden_terms=(conflict_right_value, conflict_decoy_value),
            active_memory_subject=conflict_subject,
            active_memory_term=conflict_new,
        ),
    )
    return documents, scenarios


def build_adversarial_corpus(
    source_tokens: int,
    *,
    seed: int | None = None,
) -> PreparedCorpus:
    """Build fixed or seeded adversarial facts at a requested source length.

    ``seed=None`` preserves the original development fixture byte-for-byte.
    Any integer seed creates a deterministic held-out variant with randomized
    identifiers, values, source positions, and semantically adjacent decoys.
    """

    if source_tokens < 4_000:
        raise ValueError("source_tokens must be at least 4000")
    if seed is None:
        evidence = _documents()
        scenarios = _scenarios()
    else:
        evidence, scenarios = _randomized_fixture(int(seed))
    evidence = sorted(evidence, key=lambda item: item.position)
    fixed_tokens = sum(_word_tokens(item.text) for item in evidence)
    if source_tokens <= fixed_tokens:
        raise ValueError("source token target is too small for benchmark evidence")
    noise_total = source_tokens - fixed_tokens

    def assemble(noise_count: int) -> tuple[list[BenchmarkDocument], str]:
        documents: list[BenchmarkDocument] = []
        emitted_noise = 0
        noise_ordinal = 0

        def append_noise(count: int, position: float) -> None:
            nonlocal noise_ordinal
            remaining = count
            while remaining:
                size = min(2_048, remaining)
                # ``x0`` plus its separator is three bytes and one whitespace
                # token, keeping the conservative resident estimator stable.
                words = " ".join(f"x{index % 10}" for index in range(size))
                documents.append(
                    BenchmarkDocument(
                        source=f"noise://{source_tokens}/{noise_ordinal}",
                        text=words,
                        position=position,
                        metadata={"distractor": True},
                    )
                )
                noise_ordinal += 1
                remaining -= size

        for item in evidence:
            target_noise = int(noise_count * item.position)
            append_noise(max(0, target_noise - emitted_noise), item.position - 0.0001)
            emitted_noise = target_noise
            documents.append(item)
        append_noise(noise_count - emitted_noise, 1.0)
        text = "\n\n".join(item.text for item in documents)
        return documents, text

    # The fallback estimator is max(whitespace words, UTF-8 bytes / 3).  The
    # initial word-based estimate is close; each correction changes the noise
    # by the measured delta and converges because one xN token costs ~3 bytes.
    for _attempt in range(5):
        documents, text = assemble(noise_total)
        measured = conservative_token_estimate(text)
        if measured == source_tokens:
            break
        noise_total += source_tokens - measured
        if noise_total < 0:
            raise ValueError("source token target is too small for benchmark evidence")
    else:  # pragma: no cover - construction invariant
        raise AssertionError(f"constructed {measured} tokens, expected {source_tokens}")
    return PreparedCorpus(tuple(documents), scenarios, text, measured, seed=seed)


def _dedupe_hits(hits: Iterable[RetrievalHit]) -> list[RetrievalHit]:
    found: dict[str, RetrievalHit] = {}
    for hit in hits:
        previous = found.get(hit.chunk.chunk_id)
        if previous is None or hit.score > previous.score:
            found[hit.chunk.chunk_id] = hit
    return sorted(found.values(), key=lambda item: item.score, reverse=True)


def _oracle_hits(
    store: ImmutableEvidenceStore, terms: Sequence[str]
) -> list[RetrievalHit]:
    return _dedupe_hits(
        RetrievalHit(
            chunk=chunk,
            score=1.0,
            channels=("oracle",),
            channel_scores={"oracle": 1.0},
        )
        for term in terms
        for chunk in store.exact_search(term, limit=100)
        if term.casefold() in chunk.original_text.casefold()
    )


def _direct_hits(
    store: ImmutableEvidenceStore,
    embedder: HashingEmbedder,
    query: str,
    baseline: str,
) -> list[RetrievalHit]:
    if baseline == "dense_rag":
        return [
            RetrievalHit(
                chunk=chunk,
                score=max(0.0, score),
                channels=("dense",),
                channel_scores={"dense": max(0.0, score)},
            )
            for chunk, score in store.dense_search(embedder(query), limit=12)
        ]
    if baseline == "lexical_rag":
        return [
            RetrievalHit(
                chunk=chunk,
                score=score,
                channels=("bm25",),
                channel_scores={"bm25": score},
            )
            for chunk, score in store.lexical_search(query, limit=12)
        ]
    raise ValueError(f"unsupported direct baseline: {baseline}")


def _recurrent_text(
    query: str,
    chunks: Sequence[Any],
    *,
    memory_tokens: int = 2_048,
) -> tuple[str, tuple[str, ...]]:
    """Training-free MemAgent-shaped baseline: memory_n + chunk_n -> memory_n+1.

    This intentionally remains a derived textual memory, not exact evidence.
    It retains query-overlapping lines and durable high-authority statements
    within a fixed budget; every retained line still records its raw chunk ID.
    """

    query_terms = {
        term.casefold().strip("`'\".,:;()[]")
        for term in query.split()
        if len(term.strip("`'\".,:;()[]")) >= 3
    }
    retained: list[tuple[float, int, str, str]] = []
    sequence = 0
    for chunk in chunks:
        for line in chunk.original_text.splitlines() or [chunk.original_text]:
            normalized = line.strip()
            if not normalized:
                continue
            line_terms = {
                term.casefold().strip("`'\".,:;()[]")
                for term in normalized.split()
            }
            overlap = len(query_terms & line_terms)
            durable = any(
                marker in normalized
                for marker in (
                    "MUST ",
                    "changed from",
                    "is configured as",
                    "I chose",
                    "def ",
                    "class ",
                    " uses ",
                    "communicates through",
                    "fault=",
                )
            )
            if not overlap and not durable:
                continue
            retained.append((float(overlap) + (0.25 if durable else 0.0), sequence, normalized, chunk.chunk_id))
            sequence += 1
        # WRITE: update the bounded derived state after every segment.
        ranked = sorted(retained, key=lambda item: (item[0], item[1]), reverse=True)
        selected: list[tuple[float, int, str, str]] = []
        used = 0
        seen: set[str] = set()
        for item in ranked:
            if item[2] in seen:
                continue
            tokens = conservative_token_estimate(item[2])
            if used + tokens > memory_tokens:
                continue
            selected.append(item)
            seen.add(item[2])
            used += tokens
        retained = sorted(selected, key=lambda item: item[1])
    return "\n".join(item[2] for item in retained), tuple(item[3] for item in retained)


def _term_count(text: str, terms: Sequence[str]) -> int:
    folded = text.casefold()
    return sum(term.casefold() in folded for term in terms)


def _provenance_accuracy(
    store: ImmutableEvidenceStore,
    context: WorkingContext | None,
    required_terms: Sequence[str],
) -> float:
    if context is None or not required_terms:
        return 0.0
    supported: set[str] = set()
    for item in context.items:
        for pointer in item.provenance:
            chunk = store.get_chunk(pointer.chunk_id)
            if chunk is None or not pointer.exact:
                continue
            if not 0 <= pointer.char_start <= pointer.char_end <= len(chunk.original_text):
                continue
            source = chunk.original_text[pointer.char_start : pointer.char_end].casefold()
            for term in required_terms:
                if term.casefold() in source and term.casefold() in item.text.casefold():
                    supported.add(term)
    return len(supported) / len(required_terms)


def _fifo_prompt(corpus: PreparedCorpus, query: str, budget: ContextBudget) -> str:
    contract = "Answer only from resident source evidence."
    fixed = conservative_token_estimate(f"{contract}\n{query}")
    allowance = max(0, budget.input_ceiling - fixed)
    words = corpus.source_text.split()
    tail = " ".join(words[-allowance:])
    return f"{contract}\n<resident_fifo>{tail}</resident_fifo>\n<current_query>{query}</current_query>"


class VirtualContextBenchmarkMatrix:
    """Run retrieval/packing baselines over one immutable adversarial corpus."""

    def __init__(self, *, physical_context_tokens: int = 16_384) -> None:
        self.budget = ContextBudget(max_tokens=physical_context_tokens)
        self.packer = WorkingContextPacker(budget=self.budget)

    def run_length(self, source_tokens: int, database: Path) -> dict[str, Any]:
        corpus = build_adversarial_corpus(source_tokens)
        embedder = HashingEmbedder()
        store = ImmutableEvidenceStore(database, embedder=embedder)
        extractor = StructuredMemoryExtractor(store)
        ingest_started = time.perf_counter()
        all_chunks = []
        try:
            for ordinal, document in enumerate(corpus.documents):
                chunks = store.ingest(
                    document.text,
                    source=document.source,
                    document_id=f"matrix-{source_tokens}-{ordinal}",
                    captured_at=1_700_000_000.0 + ordinal,
                    media_type=document.media_type,
                    kind=document.kind,
                    metadata=document.metadata,
                    entities=document.entities,
                )
                all_chunks.extend(chunks)
                extractor.extract(chunks, authority="benchmark_source")
                if document.relationships:
                    for subject, predicate, object_ in document.relationships:
                        store.add_relationship(
                            subject,
                            predicate,
                            object_,
                            chunk_id=chunks[0].chunk_id,
                            valid_from=1_700_000_000.0 + ordinal,
                        )
            ingest_seconds = time.perf_counter() - ingest_started
            retriever = HybridRetriever(store, query_embedder=embedder)
            memories = store.active_memories()
            measurements = [
                self._measure(
                    corpus,
                    scenario,
                    baseline,
                    store=store,
                    embedder=embedder,
                    retriever=retriever,
                    memories=memories,
                    chunks=all_chunks,
                )
                for scenario in corpus.scenarios
                for baseline in BASELINES
            ]
            reconstruction = self._reconstruction_check(store, memories)
            stats = store.stats()
        finally:
            store.close()
        return {
            "source_tokens": corpus.source_tokens,
            "physical_context_tokens": self.budget.max_tokens,
            "resident_input_ceiling": self.budget.input_ceiling,
            "ingest_seconds": round(ingest_seconds, 6),
            "index_bytes": database.stat().st_size if database.exists() else 0,
            "process_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3),
            "peak_vram_mib": 0,
            "store_stats": stats,
            "reconstruction": reconstruction,
            "measurements": [asdict(item) for item in measurements],
        }

    def _measure(
        self,
        corpus: PreparedCorpus,
        scenario: BenchmarkScenario,
        baseline: str,
        *,
        store: ImmutableEvidenceStore,
        embedder: HashingEmbedder,
        retriever: HybridRetriever,
        memories: Sequence[Any],
        chunks: Sequence[Any],
    ) -> BaselineMeasurement:
        started = time.perf_counter()
        hits: list[RetrievalHit] = []
        context: WorkingContext | None = None
        retrieval_rounds = 0
        sufficient = True
        prompt = ""
        recurrent_ids: tuple[str, ...] = ()
        if baseline == "fifo":
            prompt = _fifo_prompt(corpus, scenario.query, self.budget)
        elif baseline == "recurrent_textual":
            recurrent, recurrent_ids = _recurrent_text(scenario.query, chunks)
            context = self.packer.pack(
                scenario.query,
                system_contract="Treat recurrent memory as derived and recover raw evidence before authoritative use.",
                evidence=[],
                recurrent_memory=recurrent,
            )
            prompt = context.text
        else:
            retrieval_queries: Sequence[str] = (scenario.query,)
            selected_memories: Sequence[Any] = ()
            if baseline in {"dense_rag", "lexical_rag"}:
                hits = _direct_hits(store, embedder, scenario.query, baseline)
            elif baseline == "hybrid_rag":
                hits = retriever.retrieve(scenario.query)
            elif baseline in {"recursive_replay", "structured_replay"}:
                result = RecursiveMemoryController(
                    retriever, config=ControllerConfig(max_rounds=6)
                ).gather(scenario.query)
                hits = list(result.evidence)
                retrieval_queries = result.queries
                retrieval_rounds = sum(
                    event.get("operation") == "RETRIEVE" for event in result.trace
                )
                sufficient = result.sufficient
                if baseline == "structured_replay":
                    selected_memories = select_relevant_memories(
                        memories, scenario.query
                    )
            elif baseline == "oracle":
                hits = _oracle_hits(store, scenario.oracle_terms)
            else:  # pragma: no cover - BASELINES is internal and fixed
                raise ValueError(f"unknown baseline: {baseline}")
            context = self.packer.pack(
                scenario.query,
                system_contract="Answer from replayed exact evidence; preserve conflicts and chronology.",
                evidence=hits,
                retrieval_queries=retrieval_queries,
                memories=selected_memories,
            )
            prompt = context.text
        evidence_text = "\n".join(hit.chunk.original_text for hit in hits)
        if baseline == "recurrent_textual":
            evidence_text = "\n".join(
                chunk.original_text for chunk in chunks if chunk.chunk_id in recurrent_ids
            )
        evidence_found = _term_count(evidence_text, scenario.required_terms)
        replay_found = _term_count(prompt, scenario.required_terms)
        distractors = _term_count(prompt, scenario.forbidden_terms)
        provenance = _provenance_accuracy(store, context, scenario.required_terms)
        active_correct: bool | None = None
        if scenario.active_memory_subject is not None:
            active = store.active_memories(subject=scenario.active_memory_subject)
            active_correct = bool(
                active
                and scenario.active_memory_term
                and scenario.active_memory_term.casefold() in active[0].content.casefold()
            )
        required = max(1, len(scenario.required_terms))
        replay_recall = replay_found / required
        evidence_recall = evidence_found / required
        if context is not None:
            resident = context.total_tokens - self.budget.output_headroom
            evidence_tokens = context.token_usage.get("exact_evidence", 0)
        else:
            resident = min(self.budget.input_ceiling, conservative_token_estimate(prompt))
            evidence_tokens = 0
        exact_replay_baseline = baseline not in {"fifo", "recurrent_textual"}
        passed = replay_recall == 1.0
        if exact_replay_baseline:
            passed = passed and provenance == 1.0
        if active_correct is not None and baseline == "structured_replay":
            passed = passed and active_correct
        if baseline == "structured_replay" and scenario.forbidden_terms:
            passed = passed and distractors == 0
        if baseline in {"recursive_replay", "structured_replay"}:
            passed = passed and sufficient
        return BaselineMeasurement(
            source_tokens=corpus.source_tokens,
            scenario=scenario.name,
            family=scenario.family,
            baseline=baseline,
            required_terms=len(scenario.required_terms),
            evidence_terms_found=evidence_found,
            replay_terms_found=replay_found,
            evidence_recall=round(evidence_recall, 6),
            replay_recall=round(replay_recall, 6),
            provenance_accuracy=round(provenance, 6),
            distractor_terms_replayed=distractors,
            resident_input_tokens=resident,
            retrieved_evidence_tokens=evidence_tokens,
            compression_ratio=round(corpus.source_tokens / max(1, resident), 3),
            retrieval_rounds=retrieval_rounds,
            sufficient=sufficient,
            active_memory_correct=active_correct,
            latency_seconds=round(time.perf_counter() - started, 6),
            passed=passed,
        )

    @staticmethod
    def _reconstruction_check(store: ImmutableEvidenceStore, memories: Sequence[Any]) -> dict[str, Any]:
        checked = 0
        exact = 0
        for memory in memories:
            for pointer, recovered in store.reconstruct(memory.memory_id):
                checked += 1
                chunk = store.get_chunk(pointer.chunk_id)
                if (
                    chunk is not None
                    and recovered
                    == chunk.original_text[pointer.char_start : pointer.char_end]
                ):
                    exact += 1
        return {
            "spans_checked": checked,
            "exact_spans": exact,
            "fidelity": exact / max(1, checked),
        }


def summarize_matrix(length_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    measurements = [
        measurement
        for result in length_results
        for measurement in result["measurements"]
    ]
    baseline_summary = {}
    for baseline in BASELINES:
        selected = [item for item in measurements if item["baseline"] == baseline]
        baseline_summary[baseline] = {
            "cases": len(selected),
            "pass_rate": sum(bool(item["passed"]) for item in selected) / max(1, len(selected)),
            "mean_evidence_recall": sum(float(item["evidence_recall"]) for item in selected) / max(1, len(selected)),
            "mean_replay_recall": sum(float(item["replay_recall"]) for item in selected) / max(1, len(selected)),
            "mean_provenance_accuracy": sum(float(item["provenance_accuracy"]) for item in selected) / max(1, len(selected)),
            "mean_latency_seconds": sum(float(item["latency_seconds"]) for item in selected) / max(1, len(selected)),
        }
    production = baseline_summary[PRODUCTION_BASELINE]
    oracle = baseline_summary["oracle"]
    all_reconstruction_exact = all(
        result["reconstruction"]["fidelity"] == 1.0 for result in length_results
    )
    gate_passed = (
        production["pass_rate"] == 1.0
        and oracle["pass_rate"] == 1.0
        and all_reconstruction_exact
        and all(
            item["resident_input_tokens"] <= result["resident_input_ceiling"]
            for result in length_results
            for item in result["measurements"]
            if item["baseline"] == PRODUCTION_BASELINE
        )
    )
    return {
        "production_baseline": PRODUCTION_BASELINE,
        "gate_passed": gate_passed,
        "all_reconstruction_exact": all_reconstruction_exact,
        "baseline_summary": baseline_summary,
        "learned_branch_status": {
            "latent_context_compilation": "research_only_not_a_source_of_truth",
            "lychee_memory_gate": "research_only_not_a_source_of_truth",
            "reversible_virtual_tokens": "research_only_not_a_source_of_truth",
            "kivi_pyramidkv_quest": "separate_model_level_experiments_not_semantic_memory",
        },
    }


def benchmark_payload(length_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    physical_contexts = {
        int(result["physical_context_tokens"]) for result in length_results
    }
    if len(physical_contexts) != 1:
        raise ValueError("matrix results must use one physical context budget")
    return {
        "schema": MATRIX_SCHEMA,
        "physical_context_tokens": physical_contexts.pop(),
        "source_token_measurement": "conservative_whitespace_or_utf8_bytes_div_3",
        "source_lengths": [result["source_tokens"] for result in length_results],
        "baselines": list(BASELINES),
        "results": list(length_results),
        "summary": summarize_matrix(length_results),
    }
