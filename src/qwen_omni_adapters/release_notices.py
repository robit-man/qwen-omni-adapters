"""Build recording credits from pinned training provenance; never publish weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _records(raw: bytes, name: str) -> dict[str, dict[str, Any]]:
    records = {}
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"{name} line {number}: missing record identity")
        if row["id"] in records:
            raise ValueError(f"{name}: duplicate record {row['id']}")
        records[row["id"]] = row
    return records


def stage_training_notices(
    *, manifest: Path, attribution_ledger: Path, expected_attribution_sha256: str,
    output: Path, required_license: str = "CC-BY-3.0",
) -> dict[str, Any]:
    """Preserve the exact ledger only after checking its complete TRAIN cohort.

    Other recording/model licenses remain the publisher's separate inputs. A
    completed notice packet is provenance evidence, not model qualification.
    """
    ledger_raw = attribution_ledger.read_bytes()
    ledger_sha = hashlib.sha256(ledger_raw).hexdigest()
    if ledger_sha != expected_attribution_sha256:
        raise ValueError("attribution ledger does not match the pinned SHA-256")
    manifest_raw = manifest.read_bytes()
    rows = _records(manifest_raw, "training manifest")
    credits = _records(ledger_raw, "attribution ledger")
    required = {
        identity: row for identity, row in rows.items()
        if row.get("split") == "train" and row.get("license") == required_license
    }
    if not required or set(required) != set(credits):
        raise ValueError("ledger must cover exactly the required-license TRAIN records")
    fields = {
        "creator": "uploader", "title": "source_title", "source_url": "source_url",
        "license": "license", "license_url": "license_url",
    }
    for identity, row in required.items():
        credit = credits[identity]
        for field, source_field in fields.items():
            value = credit.get(field)
            if not isinstance(value, str) or not value.strip() or value != row.get(source_field):
                raise ValueError(f"{identity}: credit {field} differs from training provenance")
        if not isinstance(credit.get("changes"), str) or not credit["changes"].strip():
            raise ValueError(f"{identity}: missing recording changes notice")
        if not isinstance(row.get("normalization"), str) or not row["normalization"].strip():
            raise ValueError(f"{identity}: missing recording normalization provenance")
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    prefix = (
        "Training corpus recording credits\n\n"
        "These credits preserve recording provenance for the supplied training corpus.\n"
        "Backbone and other component licenses are supplied separately and are unchanged.\n"
        "\n"
        f"Recording license: {required_license}\n"
        f"Training manifest SHA-256: {manifest_sha}\n"
        f"Exact attribution ledger SHA-256: {ledger_sha}\n"
        f"Attributed TRAIN recordings: {len(credits)}\n\n"
        "BEGIN EXACT ATTRIBUTION JSONL\n"
    ).encode()
    separator = b"" if ledger_raw.endswith(b"\n") else b"\n"
    notice = prefix + ledger_raw + separator + b"END EXACT ATTRIBUTION JSONL\n"
    payloads = {"ATTRIBUTION.jsonl": ledger_raw, "TRAINING-NOTICE.txt": notice}
    report = {
        "schema": "robit.omni.training-notices.v1", "complete": True,
        "manifest_sha256": manifest_sha, "attribution_sha256": ledger_sha,
        "required_recording_license": required_license, "attributed_TRAIN_records": len(credits),
        "exact_cohort_and_credit_fields_verified": True,
        "original_ledger_bytes_preserved": True,
        "scope": "Recording provenance; other component licenses and model gates remain separate.",
        "files": [{"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                  for name, raw in payloads.items()],
        "release_qualified": False, "publication": False,
    }
    # Validate completely before creating the destination, and never replace a
    # prior packet. The report is written last as its completion marker.
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        (output / name).write_bytes(raw)
    (output / "training-notices.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attribution-ledger", type=Path, required=True)
    parser.add_argument("--expected-attribution-sha256", required=True)
    parser.add_argument("--required-license", default="CC-BY-3.0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = stage_training_notices(
        manifest=args.manifest, attribution_ledger=args.attribution_ledger,
        expected_attribution_sha256=args.expected_attribution_sha256,
        output=args.out, required_license=args.required_license,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
