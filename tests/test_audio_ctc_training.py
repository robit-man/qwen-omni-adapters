import hashlib
import json

import pytest

from runtime.train_audio_ctc_head import (
    load_records,
    minimum_ctc_frames,
    prevent_split_overlap,
    validate_feature_manifest,
    word_error_rate,
)


def test_content_aliases_cannot_leak_between_training_or_evaluation_splits():
    rows = [{"audio": "one.wav", "split": "train"}, {"audio": "renamed.wav", "split": "validation"}]
    metadata = {path: {"audio_sha256": "same-content"} for path in ("one.wav", "renamed.wav")}
    with pytest.raises(ValueError, match="different dataset splits"):
        prevent_split_overlap(rows, metadata)
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="different dataset splits"):
        prevent_split_overlap(rows, metadata)
    rows[0]["split"] = "validation"
    prevent_split_overlap(rows, metadata)  # Train-only repetitions also remain valid.


def test_changed_dataset_or_non_native_features_cannot_reuse_a_cache(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_bytes(b"original dataset")
    value = {
        "schema": "robit.omni.native-audio-feature-cache.v1",
        "dtype": "float32",
        "width": 1280,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "language_weights_loaded": False,
        "language_forward_passes": 0,
    }
    validate_feature_manifest(value, manifest)
    manifest.write_bytes(b"different split assignments")
    with pytest.raises(ValueError, match="different dataset"):
        validate_feature_manifest(value, manifest)
    value["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    value["language_weights_loaded"] = True
    with pytest.raises(ValueError, match="second language model"):
        validate_feature_manifest(value, manifest)


def test_evaluation_cannot_omit_empty_audio_and_still_report_a_false_transcript_rate(tmp_path):
    path = tmp_path / "manifest.jsonl"
    rows = [
        {
            "id": split,
            "audio": split + ".wav",
            "split": split,
            "transcript": "Speech.",
            "schema": "robit.qwen3a-audio-bridge-dataset.v1",
            "current_visual_input": False,
        }
        for split in ("train", "validation", "test")
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="speech and empty-reference"):
        load_records(path)
    rows += [
        {**r, "id": r["id"] + "-noise", "transcript": ""} for r in rows if r["split"] != "train"
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert len(load_records(path)) == 5


def test_ctc_alignment_accounts_for_repeated_symbols_and_metric_preserves_empty_audio_errors():
    assert minimum_ctc_frames([]) == 0
    assert minimum_ctc_frames([2, 2, 3, 3]) == 6
    assert word_error_rate("", "invented words") == 1
    assert word_error_rate("The blue bicycle.", "the blue bicycle") == 0
