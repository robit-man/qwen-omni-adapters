import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("failure", ["quality", "weight_identity"])
def test_native_export_never_promotes_failed_or_replaced_head(tmp_path, failure):
    files = {
        name: tmp_path / name
        for name in ("head", "head-config", "tokenizer", "decoder-report", "lexicon", "out")
    }
    files["head"].write_bytes(b"replaced candidate weights")
    files["head-config"].write_text("{}")
    files["lexicon"].write_text('["word"]')
    files["tokenizer"].write_text("{}")
    files["decoder-report"].write_text(
        json.dumps(
            {
                "validation_gate": failure != "quality",
                "test_gate": True,
                "lexicon_training_only": True,
                "external_language_model": False,
                "head_sha256": "0" * 64,
                "lexicon_sha256": hashlib.sha256(files["lexicon"].read_bytes()).hexdigest(),
            }
        )
    )
    command = [sys.executable, str(Path(__file__).parents[1] / "runtime/export_audio_ctc_head.py")]
    for name, path in files.items():
        command.extend(["--" + name, str(path)])
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert ("qualified head" if failure == "quality" else "identities differ") in result.stderr
    assert not files["out"].exists()
