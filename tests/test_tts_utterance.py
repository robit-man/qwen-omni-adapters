from dataclasses import replace
from pathlib import Path

import pytest

from runtime import tts_server as tts


@pytest.fixture
def utterance_worker(tmp_path, monkeypatch):
    binary = tmp_path / "tts"
    binary.write_text("""#!/usr/bin/env python3
import base64,json,sys
def frame(k,p=b''):
 sys.stdout.buffer.write(k.encode()+len(p).to_bytes(8,'little')+p)
 sys.stdout.buffer.flush()
active=None
previous=b''
frame('R')
for line in sys.stdin.buffer:
 if line.startswith(b'{'):
  assert '--tts-utterance-controls' in sys.argv
  value=json.loads(line)
  if value['kind']=='begin':
   assert active is None
   active=value['id']
  else:
   assert value['id']==active
   active=None
  previous=b''
  frame('C',line.rstrip(b'\\n'))
 else:
  text=base64.b64decode(line)
  frame('A',previous+text)
  if active: previous=text
  frame('D')
""")
    binary.chmod(0o755)
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"reference-A")
    weight = tmp_path / "weight.gguf"
    weight.write_bytes(b"immutable checkpoint")
    monkeypatch.setattr(tts, "_encode_reference", lambda *_: [[0] * 16])
    config = tts.Config(
        binary=binary,
        model=weight,
        projector=weight,
        reference_encoder=weight,
        continuous_batch=True,
        timeout_s=5,
    )
    worker = tts.PersistentTTSWorker(config)
    spec = tts._synthesis_spec(
        config,
        {"text": "alpha", "speaker_file": str(voice), "reference_text": "reference transcript"},
    )
    try:
        yield worker, spec
    finally:
        worker.close()


def test_continuation_is_scoped_to_one_batch_without_reloading_weights(utterance_worker):
    worker, spec = utterance_worker
    assert b"".join(worker.stream(spec)) == b"alpha"
    pid = worker.pid
    assert (
        b"".join(tts._stream_worker_batch(worker, [spec, replace(spec, text="beta")]))
        == b"alphaalphabeta"
    )
    assert worker.utterance_id is None and worker.pid == pid
    assert b"".join(worker.stream(spec)) == b"alpha"
    assert worker.pid == pid


def test_cancel_discards_the_active_utterance_and_unread_pcm(utterance_worker):
    worker, spec = utterance_worker
    stream = tts._stream_worker_batch(worker, [spec, replace(spec, text="beta")])
    assert next(stream) == b"alpha"
    pid = worker.pid
    assert worker.utterance_id
    stream.close()
    assert worker.pid is None and worker.utterance_id is None
    assert b"".join(worker.stream(spec)) == b"alpha"
    assert worker.pid != pid


def test_changed_reference_cannot_restart_or_synthesize_inside_an_utterance(utterance_worker):
    worker, spec = utterance_worker
    worker.begin_utterance(spec)
    assert b"".join(worker.stream(spec)) == b"alpha"
    Path(spec.speaker).write_bytes(b"reference-B")
    with pytest.raises(tts.TTSError, match="profile changed"):
        list(worker.stream(replace(spec, text="beta")))
    assert worker.pid is None and worker.utterance_id is None
    assert b"".join(worker.stream(spec)) == b"alpha"


def test_invalid_control_acknowledgement_terminates_the_worker(utterance_worker, monkeypatch):
    worker, spec = utterance_worker
    worker.ensure(spec)
    monkeypatch.setattr(worker, "_read_frame", lambda *_: ("A", b"unconsumed speech"))
    with pytest.raises(tts.TTSError, match="acknowledgement"):
        worker.begin_utterance(spec)
    assert worker.pid is None and worker.utterance_id is None


def test_embedding_and_single_block_requests_remain_independent(utterance_worker):
    worker, spec = utterance_worker
    assert b"".join(tts._stream_worker_batch(worker, [spec])) == b"alpha"
    embedding = replace(spec, reference_text="")
    assert (
        b"".join(tts._stream_worker_batch(worker, [embedding, replace(embedding, text="beta")]))
        == b"alphabeta"
    )
    assert worker.utterance_id is None


def test_continuous_block_limit_is_checked_before_synthesis(utterance_worker):
    worker, spec = utterance_worker
    with pytest.raises(tts.TTSError, match="at most 64"):
        tts._batch_specs(
            worker.config,
            {
                "blocks": ["text"] * 65,
                "reference_text": spec.reference_text,
                "speaker_file": spec.speaker,
            },
        )
    assert worker.pid is None
