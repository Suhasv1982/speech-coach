"""Load a faster-whisper model and turn its output into a Transcript with timed words."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config

# Whisper tends to drop disfluencies ("um", "uh") from its output. Priming it with a prompt
# that contains them makes it far more likely to keep them, which filler counting needs.
FILLER_PROMPT = "Umm, so, uh, I think, um, we should... you know, er, hmm, let me see."

# Pause detection uses voice activity detection on the audio itself, because Whisper's word
# timestamps stretch across short silences. Silences down to 0.2 s are kept as gaps, and
# speech edges are padded only slightly (the default 0.4 s pad would hide short pauses).
PAUSE_VAD = {"min_silence_duration_ms": 200, "speech_pad_ms": 50}
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Word:
    text: str  # as transcribed, may include punctuation, e.g. "platform,"
    start: float  # seconds from the start of the recording
    end: float


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)
    speech_duration_s: float = 0.0
    # (start, end) seconds of each stretch of voice found in the audio; empty if not measured
    speech_segments: list[tuple[float, float]] = field(default_factory=list)


def detect_speech(wav_path: Path) -> list[tuple[float, float]]:
    """Return (start, end) seconds of the voiced stretches in a 16 kHz WAV."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = decode_audio(str(wav_path), sampling_rate=SAMPLE_RATE)
    stamps = get_speech_timestamps(audio, VadOptions(**PAUSE_VAD), sampling_rate=SAMPLE_RATE)
    return [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in stamps]


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
    """Transcribe a 16 kHz WAV with word timestamps, skipping long silences."""
    segments, _info = model.transcribe(
        str(wav_path),
        language="en",
        beam_size=1,
        vad_filter=True,
        word_timestamps=True,
        initial_prompt=FILLER_PROMPT,
    )
    texts: list[str] = []
    words: list[Word] = []
    speech = 0.0
    for segment in segments:
        texts.append(segment.text.strip())
        speech += max(0.0, segment.end - segment.start)
        for word in segment.words or []:
            text = word.word.strip()
            if text:
                words.append(Word(text, float(word.start), float(word.end)))
    return Transcript(text=" ".join(t for t in texts if t), words=words, speech_duration_s=speech)
