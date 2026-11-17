from __future__ import annotations

import hashlib
import json

import pytest

from qwen_omni_adapters.episodic_memory import MODALITIES, EpisodicMemory


def record(store, key, text, *, modality="text", authority="user_statement", at=1000):
    return store.append(event_key=key, episode=key, modality=modality, authority=authority,
        actor="user" if authority == "user_statement" else "observer", text=text, occurred_at=at)


def fact(event_id, quote, subject, predicate, object_, **extra):
    return {"subject": subject, "predicate": predicate, "object": object_, "relation_type": "entity",
        "epistemic_status": "explicit", "support": [{"event_id": event_id, "quote": quote}], **extra}


@pytest.mark.parametrize("modality", sorted(MODALITIES))
def test_each_modality_commits_before_encoding_and_reopens_as_historical(tmp_path, modality):
    path = tmp_path / "events.sqlite3"
    store = EpisodicMemory(path)
    source = record(store, modality, "Original observed event", modality=modality)
    assert store.stats()["pending"] == 1
    store.close()
    restored = EpisodicMemory(path)
    row = restored.search("", end=2000)["results"][0]
    assert row["event_id"] == source and row["modality"] == modality
    assert row["content_sha256"] == hashlib.sha256(b"Original observed event").hexdigest()
    assert row["provenance"]["current_visual_input"] is False
    assert path.stat().st_mode & 0o777 == 0o600
    restored.close()


def test_source_keys_cannot_rewrite_content_modality_authority_or_actor(tmp_path):
    store = EpisodicMemory(tmp_path / "events.sqlite3")
    arguments = dict(event_key="one", episode="episode", modality="speech", authority="unattributed", actor="unknown", text="Speech without a verified identity")
    source = store.append(**arguments)
    assert store.append(**arguments) == source
    for changes in [{"text": "replacement"}, {"modality": "image"}, {"authority": "user_statement"}, {"actor": "device owner"}]:
        with pytest.raises(ValueError):
            store.append(**(arguments | changes))
    assert store.stats()["events"] == 1
    store.close()


def test_fabricated_encoding_rolls_back_the_entire_derived_graph(tmp_path):
    store = EpisodicMemory(tmp_path / "events.sqlite3")
    source = record(store, "one", "A green crate is on the platform.")
    with pytest.raises(ValueError, match="quote"):
        store.apply_encoding([source], {"entities": [{"name": "crate", "support": [{"event_id": source, "quote": "green crate"}]}],
            "facts": [fact(source, "red crate", "crate", "color", "red")]})
    assert store.stats()["entities"] == store.stats()["facts"] == 0
    assert store.stats()["events"] == store.stats()["pending"] == 1
    assert store.search("green", end=2000)["results"][0]["text"] == "A green crate is on the platform."
    store.close()


def test_corrections_close_fact_validity_without_changing_original_sources(tmp_path):
    store = EpisodicMemory(tmp_path / "events.sqlite3")
    old = record(store, "old", "My delivery address is Alder Street.", at=1000)
    store.apply_encoding([old], {"facts": [fact(old, "delivery address is Alder Street", "User", "delivery_address", "Alder Street")]}, {old: [1.0, 0.0]})
    original = store.facts()[0]
    newer = record(store, "new", "Use Birch Avenue as my delivery address now.", at=2000)
    store.apply_encoding([newer], {"facts": [fact(newer, "Use Birch Avenue as my delivery address now", "User", "delivery_address", "Birch Avenue", supersedes=[original["fact_id"]])]}, {newer: [1.0, 0.0]})
    assert store.facts()[0]["object"] == "Birch Avenue"
    historical = store.facts(active_only=False)
    assert next(item for item in historical if item["object"] == "Alder Street")["valid_to"] == 2000
    assert store.stats()["events"] == 2
    assert any("Alder Street" in item["text"] for item in store.search("", end=3000)["results"])
    store.close()


def test_unattributed_speech_cannot_supersede_an_owner_statement(tmp_path):
    store = EpisodicMemory(tmp_path / "events.sqlite3")
    first = record(store, "first", "The access panel is green.")
    store.apply_encoding([first], {"facts": [fact(first, "access panel is green", "panel", "color", "green")]})
    target = store.facts()[0]["fact_id"]
    ambiguous = record(store, "ambient", "The panel is red.", modality="speech", authority="unattributed", at=2000)
    with pytest.raises(ValueError):
        store.apply_encoding([ambiguous], {"facts": [fact(ambiguous, "panel is red", "panel", "color", "red", supersedes=[target])]})
    assert store.facts()[0]["object"] == "green"
    store.close()


def test_generated_narration_is_retained_but_cannot_ground_an_external_fact(tmp_path):
    store = EpisodicMemory(tmp_path / "events.sqlite3")
    source = record(store, "reply", "The valve is closed.", authority="assistant_generated")
    with pytest.raises(ValueError, match="narration"):
        store.apply_encoding([source], {"facts": [fact(source, "valve is closed", "valve", "state", "closed")]})
    result = store.search("valve", end=2000)
    assert result["results"][0]["authority"] == "assistant_generated"
    store.close()


def test_graph_recall_bridges_modalities_and_applies_time_filter_before_expansion(tmp_path):
    store = EpisodicMemory(tmp_path / "events.sqlite3")
    image = record(store, "image", "A blue enclosure carries the label Quartz.", modality="image", authority="perceptual_observation", at=1000)
    audio = record(store, "sound", "The Quartz unit emits three high chirps.", modality="sound", authority="perceptual_observation", at=1100)
    text = record(store, "text", "Quartz is the monitor next to the workbench.", at=1200)
    for event_id, quote, subject, predicate, object_ in [
        (image, "blue enclosure carries the label Quartz", "Quartz", "appearance", "blue enclosure"),
        (audio, "Quartz unit emits three high chirps", "Quartz", "sound", "three high chirps"),
        (text, "Quartz is the monitor next to the workbench", "Quartz", "location", "workbench"),
    ]:
        store.apply_encoding([event_id], {"facts": [fact(event_id, quote, subject, predicate, object_)]}, {event_id: [1.0, 0.0]})
    retrieved = store.search("the thing that warbled", vector=[1.0, 0.0], end=2000)
    assert {item["modality"] for item in retrieved["results"]} == {"image", "sound", "text"}
    assert all("graph" in item["channels"] for item in retrieved["results"])
    limited = store.search("Quartz", vector=[1.0, 0.0], start=1050, end=1150)
    assert [item["event_id"] for item in limited["results"]] == [audio]
    assert image not in {source["event_id"] for item in limited["facts"] for source in item["support"]}
    assert "blue enclosure" not in json.dumps(limited)
    store.close()


def test_large_episodes_drain_in_bounded_batches_and_embeddings_can_retry(tmp_path):
    store = EpisodicMemory(tmp_path / "events.sqlite3")
    for index in range(20):
        store.append(event_key=str(index), episode="long", modality="sensor", authority="perceptual_observation", actor="sensor", text=f"Observed reading {index}")
    sizes = []
    while store.pending():
        batch = store.pending()
        sizes.append(len(batch))
        store.apply_encoding([item["event_id"] for item in batch], {}, {item["event_id"]: [1.0, 0.0] for item in batch})
    assert sizes == [8, 8, 4]
    fresh = record(store, "later", "A later observed event", at=2000)
    store.apply_encoding([fresh], {})
    assert store.pending()[0]["encoded"] is True
    assert store.pending()[0]["vector_available"] is False
    store.close()


def test_old_sources_remain_lexically_and_semantically_retrievable_after_newer_activity(tmp_path):
    store = EpisodicMemory(tmp_path / "events.sqlite3")
    original = record(store, "original", "The glider is stored at Heliodrome.", at=1000)
    store.apply_encoding([original], {"facts": [fact(original, "glider is stored at Heliodrome", "glider", "stored_at", "Heliodrome")]}, {original: [1.0, 0.0]})
    text = "Unrelated later activity."
    digest = hashlib.sha256(text.encode()).hexdigest()
    with store._db:
        store._db.executemany(
            "INSERT INTO event_sources(event_key,episode,modality,authority,actor,text,occurred_at,recorded_at,content_sha256,provenance_json,vector_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            [(f"noise-{index}", "noise", "text", "user_statement", "user", text, 2000 + index, 2000 + index, digest, '{"historical":true}', "[0,1]") for index in range(8300)],
        )
    lexical = store.search("Heliodrome", end=20000)
    assert lexical["results"][0]["event_id"] == original
    assert lexical["facts"][0]["object"] == "Heliodrome"
    semantic = store.search("the aircraft's home", vector=[1.0, 0.0], end=20000)
    assert semantic["results"][0]["event_id"] == original
    assert semantic["events_in_window"] == 8301
    store.close()
