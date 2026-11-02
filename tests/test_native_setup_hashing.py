"""Native setup supports the Python3.10 shipped on qualified JetPack hosts."""
import hashlib

import pytest

from qwen_omni_adapters.audio_adapter import sha256
from scripts.build_native_ollama import sha
from scripts.prepare_native_ollama import identity


@pytest.mark.parametrize('function', [sha, identity, sha256], ids=['source-archive', 'model-binding', 'conditional-audio-binding'])
@pytest.mark.parametrize('contents,expected', [
    (b'', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
    (b'a' * 1000000, 'cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0'),
], ids=['empty', 'NIST-million-a'])
def test_native_setup_hashes_without_python311_file_digest(function, contents, expected, tmp_path, monkeypatch):
    artifact = tmp_path / 'artifact.bin'
    artifact.write_bytes(contents)
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    actual = function(artifact)
    if isinstance(actual, dict):
        assert actual == {'path': str(artifact.absolute()), 'sha256': expected}
    else:
        assert actual == expected
