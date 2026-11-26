"""Short offline speech recognition for the first spoken reply."""

from __future__ import annotations

import base64
import io
import json
import select
import subprocess
import sys
import threading
from pathlib import Path

MAX_AUDIO_BYTES = 16_000 * 2 * 8 + 1024


class FastVoiceRecognizer:
    """Keep the optional offline recognizer resident in its own Python environment."""

    def __init__(self, interpreter: str, model: str) -> None:
        self._command = [interpreter, str(Path(__file__).resolve()), model]
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._ready = False
        self._start()

    def _start(self) -> None:
        self._process = subprocess.Popen(  # noqa: S603 - configured interpreter and bundled worker
            self._command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._ready = False

    def _read(self, timeout: float) -> str:
        assert self._process is not None and self._process.stdout is not None
        if not select.select([self._process.stdout], [], [], timeout)[0]:
            raise TimeoutError("fast speech worker timed out")
        result = json.loads(self._process.stdout.readline())
        return result if isinstance(result, str) else ""

    def __call__(self, wav: bytes) -> str:
        with self._lock:
            try:
                if self._process is None or self._process.poll() is not None:
                    self._start()
                    return ""
                if not self._ready:
                    try:
                        self._ready = self._read(1.0) == "ready"
                    except TimeoutError:
                        return ""
                if not self._ready:
                    return ""
                assert self._process.stdin is not None
                self._process.stdin.write(base64.b64encode(wav) + b"\n")
                self._process.stdin.flush()
                return self._read(3.0)
            except (OSError, ValueError):
                self._close()
                return ""

    def _close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                pipe.close()

    def close(self) -> None:
        with self._lock:
            self._close()


def _transcribe(recognizer, wav: bytes) -> str:
    segments, _ = recognizer.transcribe(
        io.BytesIO(wav), beam_size=1, best_of=1, condition_on_previous_text=False,
        without_timestamps=True, language=None if recognizer.model.is_multilingual else "en",
    )
    segments = list(segments)
    if not segments or any(
        segment.avg_logprob < -0.65 or segment.no_speech_prob > 0.30
        or segment.compression_ratio > 2.4 for segment in segments
    ):
        return ""
    return " ".join(segment.text.strip() for segment in segments).strip()


if __name__ == "__main__":
    from faster_whisper import WhisperModel

    model_path = Path(sys.argv[1])
    if not (model_path / "model.bin").is_file():
        raise SystemExit("the configured fast speech model is unavailable")
    recognizer = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True)
    print(json.dumps("ready"), flush=True)
    for line in sys.stdin.buffer:
        audio = base64.b64decode(line.strip(), validate=True)
        print(json.dumps(_transcribe(recognizer, audio) if 0 < len(audio) <= MAX_AUDIO_BYTES else ""), flush=True)
