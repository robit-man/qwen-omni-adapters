"""Lossless context virtualization and bounded working-set invariants."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from qwen_omni_adapters.virtual_memory import (
    ContextBudget,
    ControllerConfig,
    HashingEmbedder,
    HybridRetriever,
    ImmutableEvidenceStore,
    MemoryClass,
    MemoryHierarchy,
    QueryEvidenceCompiler,
    RecurrentConfig,
    RecurrentMemoryBuilder,
    RecursiveMemoryController,
    RetrievalHit,
    StructureAwareChunker,
    StructuredMemoryExtractor,
    TraceCollector,
    VirtualContextEngine,
    WorkingContextPacker,
)
from qwen_omni_adapters.virtual_memory.models import EvidenceChunk, ProvenancePointer
from qwen_omni_adapters.virtual_memory.packer import ContextOverflow


def word_tokens(value: str) -> int:
    return len(value.split())


def test_hashing_embedder_is_stable_normalized_and_fuzzy() -> None:
    embed = HashingEmbedder(dimensions=128)
    original = embed("MotorController rotor mismatch E42")
    repeated = embed("MotorController rotor mismatch E42")
    related = embed("motor controller reports rotor mismatch E42")
    unrelated = embed("cloud cover and afternoon rain")

    dot = lambda left, right: sum(  # noqa: E731 - compact test helper
        a * b for a, b in zip(left, right, strict=True)
    )
    assert original == repeated
    assert sum(value * value for value in original) == pytest.approx(1.0)
    assert dot(original, related) > dot(original, unrelated)


def test_python_chunks_preserve_exact_offsets_and_symbols() -> None:
    source = (
        '"""module"""\n\n'
        "class MotorController:\n"
        "    def engage(self, can_id: int) -> None:\n"
        "        raise RuntimeError('CAN bus offline')\n\n"
        "def shutdown() -> None:\n"
        "    pass\n"
    )
    chunks = StructureAwareChunker(target_tokens=64, max_tokens=128, overlap_tokens=8).chunk(
        source, source="motor.py"
    )

    assert chunks
    assert any(chunk.parent_name == "MotorController" for chunk in chunks)
    assert any(("engage", "function") in chunk.symbols for chunk in chunks)
    for chunk in chunks:
        assert source[chunk.char_start : chunk.char_end] == chunk.text
        assert len(source[: chunk.char_start].encode()) == chunk.byte_start
        assert len(source[: chunk.char_end].encode()) == chunk.byte_end


def test_generic_code_chunks_follow_declarations_and_index_symbols(tmp_path: Path) -> None:
    source = (
        "import { bus } from './bus';\n\n"
        "class MotorController { engage() { return bus.open(); } }\n\n"
        "function shutdown() { return true; }\n"
    )
    store = ImmutableEvidenceStore(tmp_path / "virtual.sqlite3")
    chunks = store.ingest(source, source="controller.ts")

    assert any(chunk.parent_name == "MotorController" for chunk in chunks)
    assert any(chunk.parent_name == "shutdown" for chunk in chunks)
    assert store.symbol_search("MotorController")[0].parent_name == "MotorController"
    assert store.symbol_search("shutdown")[0].parent_name == "shutdown"
    store.close()


def test_python_code_topology_finds_callers_callees_imports_and_inheritance(
    tmp_path: Path,
) -> None:
    source = (
        "from drivers import Bus\n\n"
        "class MotorController(Bus):\n"
        "    def engage(self):\n"
        "        return calibrate_motor()\n\n"
        "def calibrate_motor():\n"
        "    return 17\n"
    )
    store = ImmutableEvidenceStore(tmp_path / "virtual.sqlite3")
    chunks = store.ingest(source, source="controller.py")

    topology = store.code_search("calibrate_motor", max_hops=2)
    topology_names = {chunk.parent_name for chunk, _distance, _edges in topology}
    predicates = {
        predicate for _chunk, _distance, edge_types in topology for predicate in edge_types
    }

    assert {"MotorController", "calibrate_motor"} <= topology_names
    assert "calls" in predicates
    assert any(
        "inherits" in edge_types
        for chunk, _distance, edge_types in store.code_search("Bus", max_hops=2)
        if chunk.parent_name == "MotorController"
    )
    assert len(chunks) >= 3
    store.close()


def test_generic_code_hint_preserves_more_specific_python_topology(
    tmp_path: Path,
) -> None:
    source = (
        "from .validator import validate_frame\n\n"
        "def send_frame(frame):\n"
        "    return validate_frame(frame)\n"
    )
    store = ImmutableEvidenceStore(tmp_path / "generic-python.sqlite3")
    store.ingest(
        source,
        source="src/bus.py",
        media_type="text/x-python",
        kind="code",
    )

    topology = store.code_search("validate_frame", max_hops=2)

    assert any(chunk.parent_name == "send_frame" for chunk, _distance, _edges in topology)
    assert any("calls" in edges for _chunk, _distance, edges in topology)
    assert store.stats()["code_edges"] >= 2
    store.close()


def test_evidence_is_database_immutable_and_idempotent(tmp_path: Path) -> None:
    store = ImmutableEvidenceStore(tmp_path / "virtual.sqlite3")
    chunks = store.ingest(
        "alpha beta gamma",
        source="notes.md",
        document_id="notes",
        version="v1",
        message_id="m-1",
    )
    again = store.ingest(
        "alpha beta gamma",
        source="notes.md",
        document_id="notes",
        version="v1",
        message_id="m-1",
    )

    assert [chunk.chunk_id for chunk in again] == [chunk.chunk_id for chunk in chunks]
    with pytest.raises(sqlite3.IntegrityError, match="immutable evidence"):
        store._db.execute(  # noqa: SLF001 - verifies a database-level invariant
            "UPDATE chunks SET original_text = 'changed' WHERE chunk_id = ?",
            (chunks[0].chunk_id,),
        )
    store.close()


def test_hybrid_retrieval_combines_exact_symbol_dense_and_metadata(tmp_path: Path) -> None:
    def embed(value: str) -> list[float]:
        lowered = value.casefold()
        return [
            float("motor" in lowered or "actuator" in lowered),
            float("weather" in lowered),
            0.1,
        ]

    store = ImmutableEvidenceStore(tmp_path / "virtual.sqlite3", embedder=embed)
    wanted = store.ingest(
        "def calibrate_motor():\n    raise RuntimeError('E42 rotor mismatch')\n",
        source="drive.py",
        metadata={"branch": "main"},
        entities=["left_leg"],
    )[0]
    store.ingest(
        "Cloud cover will increase this afternoon.",
        source="weather.md",
        metadata={"branch": "archive"},
    )
    retriever = HybridRetriever(store, query_embedder=embed)
    plan = retriever.plan(
        'Find function `calibrate_motor` causing "E42 rotor mismatch"',
        metadata_filters={"branch": "main"},
    )

    hits = retriever.retrieve(plan.original, plan=plan)

    assert hits[0].chunk.chunk_id == wanted.chunk_id
    assert {"exact", "symbol", "bm25", "dense", "metadata"} <= set(hits[0].channels)
    store.close()


def test_query_plan_promotes_bare_identifiers_and_cleans_question_entities(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "query-plan.sqlite3")
    retriever = HybridRetriever(store)

    # Preserve the high-recall stage at the top of the V1 target range. Large
    # document corpora can contain well over 120 mentions of one query entity;
    # query-aware reranking cannot recover a support chunk excluded here.
    assert retriever.candidate_limit == 200

    plan = retriever.plan("Were Scott Derrickson and Ed Wood using controller_id-42?")

    assert "controller_id-42" in plan.exact_strings
    assert "Scott Derrickson" in plan.entities
    assert "Ed Wood" in plan.entities
    assert all(not entity.startswith("Were ") for entity in plan.entities)
    store.close()


def test_retrieval_channels_can_be_ablated_independently(tmp_path: Path) -> None:
    embed = HashingEmbedder()
    store = ImmutableEvidenceStore(tmp_path / "retrieval-ablation.sqlite3", embedder=embed)
    store.ingest(
        "The zirconium actuator calibration value is cobalt-7319.",
        source="calibration.txt",
    )
    query = "zirconium actuator calibration"

    lexical = HybridRetriever(
        store,
        query_embedder=embed,
        enabled_channels=("bm25",),
    ).retrieve(query)
    dense = HybridRetriever(
        store,
        query_embedder=embed,
        enabled_channels=("dense",),
    ).retrieve(query)

    assert lexical and all(hit.channels == ("bm25",) for hit in lexical)
    assert dense and all(hit.channels == ("dense",) for hit in dense)
    with pytest.raises(ValueError, match="unknown retrieval channels"):
        HybridRetriever(store, enabled_channels=("imaginary",))
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever(store, enabled_channels=())
    store.close()


def test_retrieval_pins_every_named_address_before_source_diversity(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "anchor-coverage.sqlite3")
    anchors = ("alpha-key", "beta-key", "gamma-key", "delta-key")
    for index, anchor in enumerate(anchors):
        store.ingest(
            f"The immutable value for {anchor} is value-{index}.",
            source="one-large-corpus.txt",
            document_id=f"section-{index}",
        )

    hits = HybridRetriever(store, source_cap=1, final_limit=4).retrieve(
        "Return alpha-key, beta-key, gamma-key, and delta-key."
    )
    replay = "\n".join(hit.chunk.original_text for hit in hits).casefold()

    assert len(hits) == 4
    assert all(anchor in replay for anchor in anchors)
    store.close()


def test_named_location_uses_literal_entity_page_and_is_sufficient(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "location.sqlite3")
    wanted = store.ingest(
        "Document 17:\nNormandy is a region in France.",
        source="encyclopedia.txt",
        document_id="17",
        entities=["Normandy"],
    )[0]
    for index in range(20):
        store.ingest(
            f"Document {index + 100}:\nCountry policy archive item {index}.",
            source="encyclopedia.txt",
            document_id=str(index + 100),
        )
    controller = RecursiveMemoryController(HybridRetriever(store))

    result = controller.gather("In what country is Normandy located?")

    assert result.sufficient is True
    assert wanted.chunk_id in {hit.chunk.chunk_id for hit in result.evidence}
    assert "entity_exact" in next(
        hit.channels for hit in result.evidence if hit.chunk.chunk_id == wanted.chunk_id
    )
    store.close()


def test_controller_requires_all_explicit_addresses_before_answer(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "missing-anchor.sqlite3")
    store.ingest(
        "alpha-key is 11 and beta-key is 22; no other key is recorded.",
        source="partial.txt",
    )
    controller = RecursiveMemoryController(HybridRetriever(store))

    result = controller.gather("Return alpha-key, beta-key, and gamma-key.")

    assert result.evidence
    assert result.sufficient is False
    assert result.trace[-1]["detail"]["allowed"] is False
    store.close()


def test_query_compiler_builds_complete_multi_key_values_with_exact_provenance(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "compiled-values.sqlite3")
    for anchor, value in (
        ("alpha-key", "1234567"),
        ("beta-key", "7654321"),
        ("gamma-key", "2468135"),
    ):
        store.ingest(
            f"The immutable number for {anchor} is: {value}.",
            source="records.txt",
            document_id=anchor,
        )
    query = "What are the numbers for alpha-key, beta-key, and gamma-key?"
    evidence = HybridRetriever(store, source_cap=1).retrieve(query)

    compiled = QueryEvidenceCompiler(store).compile(query, evidence)

    assert compiled.complete is True
    assert compiled.consume_evidence is True
    assert compiled.operators == ("exact_value_lookup",)
    content = compiled.memories[0].content
    assert "key=alpha-key values=[1234567]" in content
    assert "key=beta-key values=[7654321]" in content
    assert "key=gamma-key values=[2468135]" in content
    recovered = store.reconstruct(compiled.memories[0].memory_id)
    assert len(recovered) == 3
    assert all(pointer.exact and text for pointer, text in recovered)
    store.close()


def test_query_compiler_exhausts_one_explicit_key_across_raw_corpus(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "compiled-multi-value.sqlite3")
    for index, value in enumerate(("5491659", "4689178", "5944302", "4647549")):
        store.ingest(
            f"One of the immutable numbers for learned-boom is: {value}.",
            source=f"record-{index}.txt",
            document_id=f"record-{index}",
        )
    query = "What are all the numbers for learned-boom?"
    # Deliberately provide only one retrieved page.  Exhaustive compilation
    # must recover every exact occurrence from append-only raw evidence.
    evidence = HybridRetriever(store, final_limit=1).retrieve(query)[:1]

    compiled = QueryEvidenceCompiler(store).compile(query, evidence)

    assert compiled.complete is True
    assert compiled.consume_evidence is True
    content = compiled.memories[0].content
    for value in ("5491659", "4689178", "5944302", "4647549"):
        assert value in content
    assert len(store.reconstruct(compiled.memories[0].memory_id)) == 4
    relation_event = next(
        event for event in compiled.trace if event["operation"] == "COMPILE_RELATIONS"
    )
    assert relation_event["detail"]["exhaustive_single_anchor"] is True
    assert relation_event["detail"]["exact_source_preserved"] is True
    store.close()


def test_query_compiler_resolves_assignment_graph_and_ignores_other_target(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "compiled-chain.sqlite3")
    chunks = []
    for index, line in enumerate(
        (
            "VAR LEBYM = 77969",
            "VAR VKBQB = VAR LEBYM",
            "VAR DOHZD = VAR VKBQB",
            "VAR TPUFN = VAR DOHZD",
            "VAR LLWZH = VAR TPUFN",
            "VAR DISTRACTOR = 11223",
        )
    ):
        chunks.extend(store.ingest(line, source=f"event-{index}.log", kind="log"))
    evidence = tuple(
        RetrievalHit(chunk, 1.0, ("exact",), {"exact": 1.0})
        for chunk in chunks
    )

    compiled = QueryEvidenceCompiler(store).compile(
        "Find all variables that are assigned the value 77969.",
        evidence,
    )

    assert compiled.complete is True
    assert compiled.operators == ("assignment_resolution",)
    content = compiled.memories[0].content
    for variable in ("LEBYM", "VKBQB", "DOHZD", "TPUFN", "LLWZH"):
        assert f"variable={variable} " in content
    assert "DISTRACTOR" not in content
    assert len(store.reconstruct(compiled.memories[0].memory_id)) == 5
    store.close()


def test_entity_graph_traverses_dependencies_with_provenance(tmp_path: Path) -> None:
    store = ImmutableEvidenceStore(tmp_path / "virtual.sqlite3")
    chunk = store.ingest(
        "Dropbear sends left_leg angle data through CAN ID 0x21.",
        source="robot.md",
        entities=["Dropbear", "left_leg", "CAN ID 0x21"],
    )[0]
    store.add_relationship("Dropbear", "controls", "left_leg", chunk_id=chunk.chunk_id)
    store.add_relationship("left_leg", "uses", "CAN ID 0x21", chunk_id=chunk.chunk_id)

    found = store.graph_search("How does Dropbear reach CAN ID 0x21?", max_hops=3)

    assert found
    assert found[0][0].chunk_id == chunk.chunk_id
    assert found[0][1] in {1, 2}
    store.close()


def test_supersession_preserves_old_and_current_values(tmp_path: Path) -> None:
    store = ImmutableEvidenceStore(tmp_path / "virtual.sqlite3")
    old_source = store.ingest("motor_controller = X", source="config-v1.txt")[0]
    old = store.write_memory(
        MemoryClass.DECISION,
        "motor_controller",
        "Use controller X.",
        provenance=[ProvenancePointer(old_source.chunk_id, 0, len(old_source.original_text))],
    )
    new_source = store.ingest("motor_controller = Y", source="config-v2.txt")[0]
    new = store.write_memory(
        MemoryClass.DECISION,
        "motor_controller",
        "Use controller Y.",
        provenance=[ProvenancePointer(new_source.chunk_id, 0, len(new_source.original_text))],
        supersedes=old.memory_id,
    )

    active = store.active_memories(classes=[MemoryClass.DECISION])

    assert [memory.memory_id for memory in active] == [new.memory_id]
    assert store.get_memory(old.memory_id).valid_to is not None
    assert store.reconstruct(old.memory_id)[0][1] == "motor_controller = X"
    store.close()


def test_structured_extractor_promotes_constraints_and_explicit_supersession(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "virtual.sqlite3")
    extractor = StructuredMemoryExtractor(store)
    first = store.ingest(
        "MUST NOT replace the golden controller.\nI decided to use motor_controller.\n",
        source="conversation:user",
    )
    created = extractor.extract(first, authority="user")
    second = store.ingest(
        "motor_controller changed from amber-17 to violet-29.",
        source="conversation:user",
        document_id="update",
    )
    updated = extractor.extract(second, authority="user")

    constraint = next(memory for memory in created if memory.memory_class is MemoryClass.CONSTRAINT)
    old_decision = next(memory for memory in created if memory.memory_class is MemoryClass.DECISION)
    new_decision = updated[0]
    assert constraint.importance == 1.0
    assert constraint.subject == "golden controller"
    assert constraint.ttl_seconds is None
    assert store.reconstruct(constraint.memory_id)[0][1] == constraint.content
    assert new_decision.supersedes == old_decision.memory_id
    assert store.get_memory(old_decision.memory_id).valid_to is not None
    assert [memory.memory_id for memory in store.active_memories(subject="motor_controller")] == [
        new_decision.memory_id
    ]
    store.close()


def test_structured_extractor_does_not_promote_casual_personal_must(tmp_path: Path) -> None:
    store = ImmutableEvidenceStore(tmp_path / "personal.sqlite3")
    extractor = StructuredMemoryExtractor(store)
    chunk = store.ingest("I must leave for lunch now.", source="conversation:user")

    assert extractor.extract(chunk, authority="user") == []
    assert store.active_memories(classes=[MemoryClass.CONSTRAINT]) == []
    store.close()


def test_memory_classes_have_independent_default_lifetimes(tmp_path: Path) -> None:
    store = ImmutableEvidenceStore(tmp_path / "virtual.sqlite3")
    chunk = store.ingest("Execute the migration plan.", source="plan.md")[0]
    plan = store.write_memory(
        MemoryClass.CURRENT_PLAN,
        "migration",
        "Execute the migration plan.",
        provenance=[ProvenancePointer(chunk.chunk_id, 0, len(chunk.original_text))],
    )
    fact = store.write_memory(
        MemoryClass.FACT,
        "migration owner",
        "The migration owner is Rhea.",
        provenance=[ProvenancePointer(chunk.chunk_id, 0, len(chunk.original_text))],
    )

    assert plan.importance == 1.0
    assert plan.ttl_seconds == 7 * 24 * 60 * 60
    assert fact.importance == 0.72
    assert fact.ttl_seconds is None
    store.close()


def test_conflicting_active_memories_page_in_both_exact_sources_and_block_authority(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "virtual.sqlite3")
    first = store.ingest("CAN bitrate is 500000.", source="claim-a.txt")[0]
    second = store.ingest("CAN bitrate is 1000000.", source="claim-b.txt")[0]
    for chunk, value in ((first, "500000"), (second, "1000000")):
        store.write_memory(
            MemoryClass.FACT,
            "CAN bitrate",
            f"CAN bitrate is {value}.",
            provenance=[ProvenancePointer(chunk.chunk_id, 0, len(chunk.original_text))],
        )
    retriever = HybridRetriever(store)
    controller = RecursiveMemoryController(
        retriever,
        sufficiency_judge=lambda _query, evidence: 1.0 if evidence else 0.0,
    )
    engine = VirtualContextEngine(
        store,
        controller,
        WorkingContextPacker(token_counter=word_tokens),
    )

    prepared = engine.prepare_turn(
        "What is the CAN bitrate?",
        system_contract="Resolve conflicts from exact sources.",
    )

    assert prepared.answer_allowed is False
    assert "conflicting active memory" in prepared.unresolved_reason
    assert "500000" in prepared.context.text
    assert "1000000" in prepared.context.text
    assert any(event["operation"] == "memory_conflict" for event in prepared.context.trace)
    store.close()


def test_recursive_controller_retrieves_a_second_hop_and_stops(tmp_path: Path) -> None:
    store = ImmutableEvidenceStore(tmp_path / "virtual.sqlite3")
    store.ingest("The actuator project is called Dropbear.", source="one.md")
    target = store.ingest("Dropbear's motor bus bitrate is exactly 1000000 baud.", source="two.md")[
        0
    ]
    hybrid = HybridRetriever(store, final_limit=8)

    class TwoHopRetriever:
        def retrieve(self, query, *, trace=None):
            if "Dropbear" not in query:
                return hybrid.retrieve("actuator project", trace=trace)[:1]
            return hybrid.retrieve("Dropbear motor bus bitrate", trace=trace)

    retriever = TwoHopRetriever()

    def planner(_query, evidence, history):
        if len(history) == 1 and evidence:
            return ["Dropbear motor bus bitrate"]
        return []

    controller = RecursiveMemoryController(
        retriever,  # type: ignore[arg-type]
        config=ControllerConfig(max_rounds=3, sufficiency_threshold=0.5),
        dependency_planner=planner,
        sufficiency_judge=lambda _query, evidence: (
            1.0 if any("1000000" in hit.chunk.original_text for hit in evidence) else 0.0
        ),
    )

    result = controller.gather("What is the actuator project's motor bus bitrate?")

    assert result.sufficient
    assert target.chunk_id in {hit.chunk.chunk_id for hit in result.evidence}
    assert len(result.queries) == 2
    operations = [event["operation"] for event in result.trace]
    assert operations.count("PRETHINK") == 2
    assert operations[-2:] == ["STOP", "ANSWER"]
    store.close()


def test_recursive_controller_follows_natural_language_bridge_entity(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "natural-bridge.sqlite3")
    first = store.ingest(
        "Kiss and Tell stars Shirley Temple as Corliss Archer.",
        source="film.txt",
    )[0]
    target = store.ingest(
        "Shirley Temple served as Chief of Protocol of the United States.",
        source="biography.txt",
    )[0]

    class TwoHopRetriever:
        def retrieve(self, query, *, trace=None):
            chunk = target if '"Shirley Temple"' in query else first
            return [RetrievalHit(chunk, 1.0, ("exact",), {"exact": 1.0})]

    controller = RecursiveMemoryController(
        TwoHopRetriever(),  # type: ignore[arg-type]
        config=ControllerConfig(max_rounds=3, sufficiency_threshold=0.5),
        sufficiency_judge=lambda _query, evidence: 1.0 if evidence else 0.0,
    )

    result = controller.gather(
        "What government position was held by the woman who portrayed "
        "Corliss Archer in Kiss and Tell?"
    )

    assert result.sufficient is True
    assert target.chunk_id in {hit.chunk.chunk_id for hit in result.evidence}
    assert any('"Shirley Temple"' in query for query in result.queries)
    assert [event["operation"] for event in result.trace][-2:] == ["STOP", "ANSWER"]
    store.close()


def test_controller_treats_output_directives_as_control_not_missing_evidence(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "direct-fact.sqlite3")
    target = store.ingest(
        "For the actuator calibration record, the exact immutable nonce is heliotrope-7319.",
        source="conversation:user",
    )[0]
    controller = RecursiveMemoryController(HybridRetriever(store))

    result = controller.gather(
        "What is the exact immutable actuator calibration nonce? Reply with the nonce only."
    )

    assert result.sufficient is True
    assert target.chunk_id in {hit.chunk.chunk_id for hit in result.evidence}
    assert result.queries == (
        "What is the exact immutable actuator calibration nonce? Reply with the nonce only.",
    )
    sufficiency = next(
        event for event in result.trace if event["operation"] == "evidence_sufficiency"
    )
    assert sufficiency["detail"]["score"] >= 0.78
    store.close()


def test_controller_does_not_treat_an_old_unanswered_question_as_evidence(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "question-only.sqlite3")
    store.ingest(
        "What is the exact immutable actuator calibration nonce?",
        source="conversation:user",
    )
    controller = RecursiveMemoryController(HybridRetriever(store))

    result = controller.gather("What is the exact immutable actuator calibration nonce?")

    assert result.evidence
    assert result.sufficient is False
    assert result.trace[-1]["operation"] == "ANSWER"
    assert result.trace[-1]["detail"]["allowed"] is False
    store.close()


def test_recursive_controller_closes_four_hop_assignment_chain_before_stop(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "assignment-chain.sqlite3")
    assignments = (
        "VAR IWSHA = 72955",
        "VAR YCSMT = VAR IWSHA",
        "VAR RQMUC = VAR YCSMT",
        "VAR FRHPM = VAR RQMUC",
        "VAR NLTIS = VAR FRHPM",
    )
    expected = {
        store.ingest(
            f"The grass is green. {assignment}. Here we go.",
            source=f"chain-{index}.txt",
        )[0].chunk_id
        for index, assignment in enumerate(assignments)
    }
    controller = RecursiveMemoryController(
        HybridRetriever(store, final_limit=12),
        config=ControllerConfig(max_rounds=6, sufficiency_threshold=0.5),
        # Deliberately overconfident: dependency closure must still prevent an
        # early stop after the first matching value.
        sufficiency_judge=lambda _query, evidence: 1.0 if evidence else 0.0,
    )

    result = controller.gather("Find all variables that are assigned the value 72955.")

    assert result.sufficient
    assert expected <= {hit.chunk.chunk_id for hit in result.evidence}
    assert len(result.queries) == 6
    assert not any('"The"' in query or '"Here"' in query for query in result.queries)
    sufficiency_events = [
        event for event in result.trace if event["operation"] == "evidence_sufficiency"
    ]
    assert sufficiency_events[0]["detail"]["unresolved_dependencies"]
    assert sufficiency_events[-1]["detail"]["unresolved_dependencies"] == []
    assert [event["operation"] for event in result.trace][-2:] == ["STOP", "ANSWER"]
    store.close()


def test_recursive_controller_fails_closed_when_dependency_budget_expires(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "bounded-chain.sqlite3")
    store.ingest("VAR ROOT = 72955", source="root.txt")
    store.ingest("VAR NEXT = VAR ROOT", source="next.txt")
    controller = RecursiveMemoryController(
        HybridRetriever(store),
        config=ControllerConfig(max_rounds=1, sufficiency_threshold=0.5),
        sufficiency_judge=lambda _query, evidence: 1.0 if evidence else 0.0,
    )

    result = controller.gather("Find all variables assigned the value 72955.")

    assert result.sufficient is False
    assert result.trace[-2]["operation"] == "STOP"
    assert result.trace[-2]["detail"]["reason"] == "retrieval_budget_exhausted"
    assert result.trace[-1]["detail"]["allowed"] is False
    store.close()


def test_packer_keeps_constraints_and_replays_exact_evidence_next_to_query(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "virtual.sqlite3")
    chunk = store.ingest(
        "The launch code is cobalt-771.\n\nUnrelated paragraph " + "noise " * 200,
        source="runbook.md",
    )[0]
    constraint = store.write_memory(
        MemoryClass.CONSTRAINT,
        "asset edits",
        "MUST update the existing asset; NEVER create a new asset for each edit.",
        provenance=[ProvenancePointer(chunk.chunk_id, 0, 30)],
        importance=1.0,
    )
    hit = HybridRetriever(store).retrieve('What is the "cobalt-771" launch code?')[0]
    packer = WorkingContextPacker(
        budget=ContextBudget(
            max_tokens=4096,
            output_headroom=512,
            system_target=40,
            pinned_target=60,
            structured_target=20,
            recent_target=60,
            evidence_target=100,
        ),
        token_counter=word_tokens,
    )
    trace = TraceCollector()

    packed = packer.pack(
        "What is the launch code?",
        system_contract="Use evidence and obey active constraints.",
        evidence=[hit],
        memories=[constraint],
        recent_context=["old " * 100, "most recent exchange"],
        trace=trace,
    )

    assert packed.total_tokens <= packed.max_tokens
    assert "NEVER create a new asset" in packed.text
    assert "cobalt-771" in packed.text
    assert packed.text.rfind("<exact_evidence") < packed.text.rfind("<current_query>")
    assert packed.items[-2].category == "exact_evidence"
    assert any(event["operation"] == "EVICT" for event in packed.trace)
    store.close()


def test_packer_reserves_structured_memory_before_large_evidence(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "structured-reserve.sqlite3")
    source = store.ingest(
        "needle-17 " + "distractor " * 3_000,
        source="archive.txt",
    )[0]
    fact = store.write_memory(
        MemoryClass.FACT,
        "verified aggregate",
        "The verified whole-corpus result is orchid-17.",
        provenance=[ProvenancePointer(source.chunk_id, 0, len(source.original_text))],
        verified=True,
    )
    hit = RetrievalHit(source, 1.0, ("exact",), {"exact": 1.0})
    packer = WorkingContextPacker(
        budget=ContextBudget(
            max_tokens=4_096,
            output_headroom=512,
            structured_target=200,
            evidence_target=4_000,
        ),
        token_counter=word_tokens,
    )

    packed = packer.pack(
        "What is the result?",
        system_contract="Use the verified memory and exact evidence.",
        evidence=[hit],
        memories=[fact],
    )

    assert "orchid-17" in packed.text
    assert packed.token_usage["fact"] > 0
    assert packed.total_tokens <= 4_096
    store.close()


def test_packer_evicts_overlapping_page_without_new_query_evidence() -> None:
    def chunk(chunk_id: str, start: int, end: int, text: str) -> EvidenceChunk:
        return EvidenceChunk(
            chunk_id=chunk_id,
            document_id="chain",
            message_id=None,
            source="chain.log",
            version="v1",
            captured_at=1.0,
            ordinal=start,
            token_start=start,
            token_end=end,
            char_start=start,
            char_end=end,
            byte_start=start,
            byte_end=end,
            parent_kind="log_event",
            parent_name=None,
            previous_chunk_id=None,
            next_chunk_id=None,
            content_hash=chunk_id,
            original_text=text,
        )

    first = RetrievalHit(
        chunk("first", 0, 100, "VAR RQMUC = VAR YCSMT"),
        2.0,
        ("exact",),
    )
    duplicate = RetrievalHit(
        chunk("duplicate", 80, 180, "VAR RQMUC = VAR YCSMT plus archive noise"),
        1.9,
        ("exact",),
    )
    dependency = RetrievalHit(
        chunk("dependency", 200, 300, "VAR NLTIS = VAR FRHPM"),
        1.8,
        ("exact",),
    )
    packer = WorkingContextPacker(
        budget=ContextBudget(max_tokens=4_096, output_headroom=512),
        token_counter=word_tokens,
    )

    packed = packer.pack(
        "Trace RQMUC, YCSMT, NLTIS, and FRHPM.",
        system_contract="Use exact evidence.",
        evidence=[first, duplicate, dependency],
    )

    assert "first" in packed.evidence_chunk_ids
    assert "dependency" in packed.evidence_chunk_ids
    assert "duplicate" not in packed.evidence_chunk_ids
    assert any(
        event["detail"].get("reason") == "overlapping_evidence_duplicate"
        for event in packed.trace
        if event["operation"] == "EVICT"
    )


def test_packer_uses_dependency_query_for_exact_line_replay(tmp_path: Path) -> None:
    store = ImmutableEvidenceStore(tmp_path / "line-replay.sqlite3")
    source = (
        "routine archive noise\n" * 300
        + "VAR FRHPM = VAR RQMUC\n"
        + "routine archive noise\n" * 300
    )
    store.ingest(source, source="events.log", kind="log")
    hit = HybridRetriever(store).retrieve('Trace dependency "RQMUC"')[0]
    store.ingest(
        "VAR SIT = 74925\nVAR YRE = VAR SIT\n",
        source="unrelated.log",
        kind="log",
    )
    distractor = HybridRetriever(store).retrieve('Trace dependency "SIT"')[0]
    packer = WorkingContextPacker(
        budget=ContextBudget(
            max_tokens=4096,
            output_headroom=512,
            evidence_target=40,
        ),
        token_counter=word_tokens,
    )

    packed = packer.pack(
        "Find all variables assigned the value 72955.",
        system_contract="Use exact evidence.",
        evidence=[distractor, hit],
        retrieval_queries=['Trace dependency "RQMUC"'],
    )

    assert "VAR FRHPM = VAR RQMUC" in packed.text
    assert "VAR SIT = 74925" not in packed.text
    assert any(
        event["operation"] == "EVICT" and event["detail"].get("reason") == "dependency_focus"
        for event in packed.trace
    )
    evidence_item = next(item for item in packed.items if item.category == "exact_evidence")
    pointer = evidence_item.provenance[0]
    chunk = store.get_chunk(pointer.chunk_id)
    assert chunk is not None
    replayed = chunk.original_text[pointer.char_start : pointer.char_end]
    assert replayed in evidence_item.text
    assert len(replayed) < len(chunk.original_text)
    store.close()


def test_packer_replays_multiple_disjoint_exact_spans_from_one_chunk(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(
        tmp_path / "multi-span.sqlite3",
        chunker=StructureAwareChunker(target_tokens=2048, max_tokens=4096, overlap_tokens=0),
    )
    chunk = store.ingest(
        "needle_key = value-111\n" + "noise " * 900 + "needle_key = value-222\n" + "noise " * 900,
        source="single-large-chunk.txt",
    )[0]
    hit = RetrievalHit(
        chunk=chunk,
        score=1.0,
        channels=("oracle",),
        channel_scores={"oracle": 1.0},
    )
    packer = WorkingContextPacker(
        budget=ContextBudget(
            max_tokens=4096,
            output_headroom=512,
            evidence_target=500,
        ),
        token_counter=word_tokens,
    )

    packed = packer.pack(
        "Return both values for needle_key.",
        system_contract="Use exact evidence.",
        evidence=[hit],
        retrieval_queries=("value-111", "value-222"),
    )

    assert "value-111" in packed.text
    assert "value-222" in packed.text
    item = next(item for item in packed.items if item.category == "exact_evidence")
    assert len(item.provenance) >= 2
    assert all(
        chunk.original_text[pointer.char_start : pointer.char_end] in item.text
        for pointer in item.provenance
    )
    store.close()


def test_document_number_headings_create_independent_retrieval_sections() -> None:
    text = (
        "Task preamble.\n\n"
        "Document 1:\nNormandy is a region in France.\n\n"
        "Document 2:\nScott Derrickson is an American director.\n"
    )

    chunks = StructureAwareChunker().chunk(text, kind="document")

    assert [chunk.parent_kind for chunk in chunks] == [
        "document_section",
        "document_section",
        "document_section",
    ]
    assert chunks[1].parent_name == "1"
    assert chunks[2].parent_name == "2"
    assert "Normandy" in chunks[1].text
    assert "Scott Derrickson" in chunks[2].text
    assert all(text[chunk.char_start : chunk.char_end] == chunk.text for chunk in chunks)


def test_packer_refuses_to_silently_drop_active_constraints(tmp_path: Path) -> None:
    store = ImmutableEvidenceStore(tmp_path / "virtual.sqlite3")
    chunk = store.ingest("source", source="source.txt")[0]
    constraint = store.write_memory(
        MemoryClass.CONSTRAINT,
        "oversized",
        "MUST " + "retain " * 5000,
        provenance=[ProvenancePointer(chunk.chunk_id, 0, len(chunk.original_text))],
    )
    packer = WorkingContextPacker(
        budget=ContextBudget(max_tokens=4096, output_headroom=512),
        token_counter=word_tokens,
    )

    with pytest.raises(ContextOverflow, match="constraints/current plan"):
        packer.pack(
            "do it",
            system_contract="system",
            evidence=[],
            memories=[constraint],
        )
    store.close()


def test_packer_reserves_live_tool_and_transport_envelope(tmp_path: Path) -> None:
    store = ImmutableEvidenceStore(tmp_path / "envelope.sqlite3")
    store.ingest("answer-17 " + "evidence " * 2000, source="large.txt")
    hit = HybridRetriever(store).retrieve("answer-17")[0]
    packer = WorkingContextPacker(
        budget=ContextBudget(max_tokens=4096, output_headroom=512),
        token_counter=word_tokens,
    )

    packed = packer.pack(
        "What is the answer?",
        system_contract="Use evidence.",
        evidence=[hit],
        reserved_tokens=900,
    )

    assert packed.total_tokens <= 4096
    assert packed.token_usage["request_envelope"] == 900
    assert sum(item.tokens for item in packed.items) <= 4096 - 512 - 900
    store.close()


def test_constraint_survives_over_100k_intervening_tokens(tmp_path: Path) -> None:
    store = ImmutableEvidenceStore(
        tmp_path / "virtual.sqlite3",
        chunker=StructureAwareChunker(target_tokens=1024, max_tokens=2048),
    )
    invariant = store.ingest(
        "MUST NOT replace the golden controller configuration.",
        source="conversation:turn-1",
        kind="conversation",
    )[0]
    store.ingest(
        " ".join(f"distractor{i}" for i in range(100_500)),
        source="conversation:turn-2",
        kind="conversation",
    )
    constraint = store.write_memory(
        MemoryClass.CONSTRAINT,
        "controller configuration",
        "MUST NOT replace the golden controller configuration.",
        provenance=[ProvenancePointer(invariant.chunk_id, 0, len(invariant.original_text))],
        importance=1.0,
    )
    packed = WorkingContextPacker(token_counter=word_tokens).pack(
        "Replace the controller configuration with a locally convenient default.",
        system_contract="Follow active constraints.",
        evidence=[],
        memories=[constraint],
        recent_context=["distractor100499"],
    )

    assert "MUST NOT replace" in packed.text
    assert any(item.item_id == constraint.memory_id and item.pinned for item in packed.items)
    store.close()


@pytest.mark.parametrize("memory_tokens", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("chunk_tokens", [2048, 4096, 8192])
def test_recurrent_memory_budget_frontier_keeps_raw_sources_recoverable(
    tmp_path: Path, memory_tokens: int, chunk_tokens: int
) -> None:
    store = ImmutableEvidenceStore(tmp_path / f"memory-{memory_tokens}-{chunk_tokens}.sqlite3")
    chunks = [
        store.ingest("Project Zephyr uses the Copperfinch bus.", source="segment-1.txt")[0],
        store.ingest("Copperfinch operates at exactly 833333 baud.", source="segment-2.txt")[0],
    ]

    def writer(request):
        return " ".join(
            part for part in (request.previous_memory, request.chunk.original_text) if part
        )

    result = RecurrentMemoryBuilder(
        store,
        writer,
        config=RecurrentConfig(
            memory_tokens=memory_tokens,
            chunk_tokens=chunk_tokens,
        ),
        token_counter=word_tokens,
    ).process("What bitrate does Project Zephyr use?", [chunk.chunk_id for chunk in chunks])

    assert result.memory is not None
    assert "833333" in result.memory.content
    assert result.memory.compression_generation == 2
    assert result.memory.verified is False
    assert [text for _pointer, text in store.reconstruct(result.memory.memory_id)] == [
        chunk.original_text for chunk in chunks
    ]
    assert len(store.active_memories(classes=[MemoryClass.EPISODE])) == 1
    store.close()


def test_recurrent_writer_cannot_silently_overrun_its_budget(tmp_path: Path) -> None:
    store = ImmutableEvidenceStore(tmp_path / "memory.sqlite3")
    chunk = store.ingest("small exact source", source="segment.txt")[0]
    builder = RecurrentMemoryBuilder(
        store,
        lambda _request: "overflow " * 513,
        config=RecurrentConfig(memory_tokens=512, chunk_tokens=2048),
        token_counter=word_tokens,
    )

    with pytest.raises(ValueError, match="exceeded memory budget"):
        builder.process("query", [chunk.chunk_id])
    assert store.active_memories(classes=[MemoryClass.EPISODE]) == []
    assert store.get_chunk(chunk.chunk_id).original_text == "small exact source"
    store.close()


def test_recurrent_memory_periodically_rebuilds_from_raw_evidence(tmp_path: Path) -> None:
    store = ImmutableEvidenceStore(tmp_path / "regenerate.sqlite3")
    chunks = [
        store.ingest(
            f"segment-{index} exact-value-{index}",
            source=f"segment-{index}.txt",
            document_id=f"segment-{index}",
        )[0]
        for index in range(4)
    ]

    def writer(request):
        return " ".join(
            part for part in (request.previous_memory, request.chunk.original_text) if part
        )

    result = RecurrentMemoryBuilder(
        store,
        writer,
        config=RecurrentConfig(
            memory_tokens=512,
            chunk_tokens=2048,
            regenerate_every=4,
        ),
        token_counter=word_tokens,
    ).process("List the exact segment values.", [chunk.chunk_id for chunk in chunks])

    assert result.memory is not None
    assert result.memory.compression_generation == 1
    assert result.memory.metadata["regenerated_from_raw"] is True
    assert all(f"exact-value-{index}" in result.memory.content for index in range(4))
    assert any(event["operation"] == "RECONSTRUCT" for event in result.trace)
    store.close()


def test_explicit_hierarchy_operations_are_observable_and_pins_are_protected(
    tmp_path: Path,
) -> None:
    store = ImmutableEvidenceStore(tmp_path / "virtual.sqlite3")
    first_chunk = store.ingest("controller = amber", source="one.txt")[0]
    second_chunk = store.ingest("controller = violet", source="two.txt")[0]
    first = store.write_memory(
        MemoryClass.DECISION,
        "controller",
        "Use amber.",
        provenance=[ProvenancePointer(first_chunk.chunk_id, 0, len(first_chunk.original_text))],
    )
    hierarchy = MemoryHierarchy(store)
    hierarchy.page_in(first_chunk.chunk_id, level="L3", tokens=4, pinned=True)

    with pytest.raises(ValueError, match="pinned"):
        hierarchy.evict(first_chunk.chunk_id, reason="pressure")
    hierarchy.unpin(first_chunk.chunk_id)
    assert hierarchy.page_out(first_chunk.chunk_id, reason="answer complete")
    replacement = hierarchy.supersede(
        first.memory_id,
        content="Use violet.",
        provenance=[ProvenancePointer(second_chunk.chunk_id, 0, len(second_chunk.original_text))],
    )
    assert hierarchy.reconstruct(replacement.memory_id)[0][1] == "controller = violet"
    operations = [event["operation"] for event in hierarchy.trace.export()]
    assert operations == [
        "PAGE_IN",
        "PIN",
        "UNPIN",
        "PAGE_OUT",
        "SUPERSEDE",
        "RECONSTRUCT",
    ]
    store.close()
