from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qwen_omni_adapters.release_notices import stage_training_notices


def _inputs(tmp_path: Path):
    rows = [{
        "id": "recording-1", "split": "train", "license": "CC-BY-3.0",
        "uploader": "Original creator", "source_title": 'Original "recording" — 鳥',
        "source_url": "https://example.org/recording/1",
        "license_url": "https://creativecommons.org/licenses/by/3.0/",
        "normalization": "resample-pcm16-v1",
    }, {"id": "validation-1", "split": "validation", "license": "CC-BY-3.0"},
        {"id": "public-domain-1", "split": "train", "license": "CC0-1.0"}]
    credits = [{
        "id": "recording-1", "creator": rows[0]["uploader"], "title": rows[0]["source_title"],
        "source_url": rows[0]["source_url"], "license": rows[0]["license"],
        "license_url": rows[0]["license_url"], "changes": "Resampled to 16 kHz mono PCM16.",
    }]
    return rows, credits


def _stage(tmp_path: Path, rows, credits, *, pin: str | None = None):
    manifest, ledger = tmp_path / "manifest.jsonl", tmp_path / "original.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    # Intentionally retain a blank line and omit the final newline. The output
    # must preserve the submitted ledger rather than silently reserializing it.
    raw = ("\n" + "\n".join(json.dumps(row, ensure_ascii=False) for row in credits)).encode()
    ledger.write_bytes(raw)
    report = stage_training_notices(
        manifest=manifest, attribution_ledger=ledger, output=tmp_path / "out",
        expected_attribution_sha256=pin or hashlib.sha256(raw).hexdigest(),
    )
    return report, raw


def test_notice_retains_exact_credit_bytes_and_separate_qualification(tmp_path):
    rows, credits = _inputs(tmp_path)
    report, raw = _stage(tmp_path, rows, credits)
    assert (tmp_path / "out/ATTRIBUTION.jsonl").read_bytes() == raw
    notice = (tmp_path / "out/TRAINING-NOTICE.txt").read_bytes()
    assert b"BEGIN EXACT ATTRIBUTION JSONL\n" + raw + b"\nEND EXACT ATTRIBUTION JSONL\n" in notice
    assert report["attributed_TRAIN_records"] == 1
    assert report["publication"] is False and report["release_qualified"] is False
    assert json.loads((tmp_path / "out/training-notices.json").read_text()) == report
    for file in report["files"]:
        content = (tmp_path / "out" / file["path"]).read_bytes()
        assert len(content) == file["bytes"]
        assert hashlib.sha256(content).hexdigest() == file["sha256"]


@pytest.mark.parametrize("field", ["creator", "title", "source_url", "license", "license_url"])
def test_misattributed_or_relicensed_recording_is_rejected_before_output(tmp_path, field):
    rows, credits = _inputs(tmp_path)
    credits[0][field] = "Changed provenance"
    with pytest.raises(ValueError, match="differs from training provenance"):
        _stage(tmp_path, rows, credits)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("defect", ["missing", "extra", "duplicate", "validation"])
def test_incomplete_duplicate_or_held_out_credits_cannot_ship(tmp_path, defect):
    rows, credits = _inputs(tmp_path)
    if defect == "missing":
        credits.clear()
    elif defect == "extra":
        credits.append({**credits[0], "id": "absent-recording"})
    elif defect == "duplicate":
        credits.append(credits[0].copy())
    else:
        credits.append({**credits[0], "id": "validation-1"})
    with pytest.raises(ValueError):
        _stage(tmp_path, rows, credits)
    assert not (tmp_path / "out").exists()


def test_stale_pinned_ledger_is_rejected(tmp_path):
    rows, credits = _inputs(tmp_path)
    with pytest.raises(ValueError, match="pinned SHA-256"):
        _stage(tmp_path, rows, credits, pin="0" * 64)
    assert not (tmp_path / "out").exists()


def test_ambiguous_manifest_identity_is_rejected(tmp_path):
    rows, credits = _inputs(tmp_path)
    rows.append({**rows[0], "source_title": "Different source"})
    with pytest.raises(ValueError, match="duplicate record"):
        _stage(tmp_path, rows, credits)
    assert not (tmp_path / "out").exists()


def test_missing_transformation_notice_is_rejected(tmp_path):
    rows, credits = _inputs(tmp_path)
    credits[0]["changes"] = ""
    with pytest.raises(ValueError, match="changes notice"):
        _stage(tmp_path, rows, credits)
    assert not (tmp_path / "out").exists()


def test_existing_notice_packet_cannot_be_overwritten(tmp_path):
    rows, credits = _inputs(tmp_path)
    _stage(tmp_path, rows, credits)
    before = (tmp_path / "out/TRAINING-NOTICE.txt").read_bytes()
    with pytest.raises(FileExistsError):
        _stage(tmp_path, rows, credits)
    assert (tmp_path / "out/TRAINING-NOTICE.txt").read_bytes() == before
