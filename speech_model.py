"""Load a faster-whisper model and turn its output into a simple Transcript."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config


@dataclass
class Transcript:
    text: str
    words: list[str] = field(default_factory=list)
    speech_duration_s: float = 0.0


def load_model(name: str | None = None, compute_type: str | None = None) -> Any:
    """Create a CPU WhisperModel; downloads the model on first use."""
    from faster_whisper import WhisperModel  # imported here so tests and startup stay fast

    default_name, default_compute = config.whisper_settings()
    return WhisperModel(
        name or default_name,
        device="cpu",
        compute_type=compute_type or default_compute,
        cpu_threads=2 if config.cloud_mode() else 0,  # 0 = let ctranslate2 decide
    )


def transcribe(model: Any, wav_path: Path) -> Transcript:
    """Transcribe a 16 kHz WAV, skipping long silences."""
    segments, _info = model.transcribe(
        str(wav_path), language="en", beam_size=1, vad_filter=True, word_timestamps=True
    )
    texts: list[str] = []
    words: list[str] = []
    speech = 0.0
    for segment in segments:
        texts.append(segment.text.strip())
        speech += max(0.0, segment.end - segment.start)
        for word in segment.words or []:
            words.append(word.word.strip())
    return Transcript(text=" ".join(t for t in texts if t), words=words, speech_duration_s=speech)
