"""Short offline speech recognition for the first spoken reply."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

MAX_AUDIO_BYTES = 16_000 * 2 * 8 + 1024


def transcribe(wav: bytes, model_path: Path) -> str:
    from faster_whisper import WhisperModel

    if not (model_path / "model.bin").is_file():
        raise ValueError("the configured fast speech model is unavailable")
    recognizer = WhisperModel(
        str(model_path), device="cpu", compute_type="int8", cpu_threads=4,
        local_files_only=True,
    )
    segments, _ = recognizer.transcribe(
        io.BytesIO(wav), beam_size=1, best_of=1, condition_on_previous_text=False,
        without_timestamps=True,
    )
    segments = list(segments)
    if not segments or any(
        segment.avg_logprob < -0.65 or segment.no_speech_prob > 0.30
        or segment.compression_ratio > 2.4 for segment in segments
    ):
        return ""
    return " ".join(segment.text.strip() for segment in segments).strip()


if __name__ == "__main__":
    audio = sys.stdin.buffer.read(MAX_AUDIO_BYTES + 1)
    if not audio or len(audio) > MAX_AUDIO_BYTES:
        raise SystemExit("fast speech requires a short mono WAV")
    print(json.dumps(transcribe(audio, Path(sys.argv[1]))))
