"""Process one take: save, convert, transcribe, measure, then store or clean up."""

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import audio
import config
import storage
from analysis import METRIC_UNITS, compute_metrics
from speech_model import Word, detect_speech, transcribe
from tasks import Task
from voice_analysis import analyse_voice, voice_metrics

log = logging.getLogger("speech_coach")


@dataclass
class TakeResult:
    task_id: str
    source: str
    duration_s: float
    transcript: str
    metrics: dict[str, float] = field(default_factory=dict)
    wav_path: str | None = None  # kept only in local mode
    model: str | None = None  # e.g. "small.en (int8)"
    details: dict[str, Any] = field(default_factory=dict)  # fillers, pace windows, voice, ...
    # (time_s, Hz) voiced pitch points for the chart; kept out of the JSON report and SQLite
    pitch_contour: list[tuple[float, float]] = field(default_factory=list)

    @property
    def warnings(self) -> list[str]:
        return self.details.get("warnings", [])


def process_take(
    *,
    task: Task,
    session_id: str,
    data: bytes,
    ext: str,
    source: str,
    model: Any,
    cloud: bool,
    model_label: str | None = None,
    recorder: dict[str, Any] | None = None,
    base_dir: Path | None = None,
    db: sqlite3.Connection | None = None,
) -> TakeResult:
    """Run the full analysis for one recording.

    Cloud mode: files live in base_dir (a temp folder) and are always deleted afterwards,
    whether analysis succeeds or fails. Local mode: files are kept and results go to SQLite.
    recorder: what the browser recorder reported about the microphone track, if any.
    """
    original = None
    try:
        original = audio.save_recording(session_id, task.id, data, ext, base_dir=base_dir)
        recording = {"source": source, **audio.probe(original), **(recorder or {})}
        log.info("Recording for task %s: %s", task.id, recording)
        wav = audio.convert_to_wav(original)
        duration = audio.wav_duration_s(wav)
        if recording.get("duration_s") is None:  # browser webm files carry no duration
            recording["duration_s"] = round(duration, 2)
        transcript = transcribe(model, wav)
        transcript.speech_segments = detect_speech(wav)
        analysis = compute_metrics(task, transcript, duration)
        analysis.details["recording"] = recording
        voice, contour = run_voice_analysis(wav, transcript.words, task.id)
        analysis.details["voice"] = voice
        analysis.metrics.update(voice_metrics(voice))
        result = TakeResult(
            task_id=task.id,
            source=source,
            duration_s=duration,
            transcript=transcript.text,
            metrics=analysis.metrics,
            wav_path=None if cloud else str(wav),
            model=model_label,
            details=analysis.details,
            pitch_contour=contour,
        )
        if not cloud:
            _store(db, session_id, task, result, original, wav)
        return result
    finally:
        if cloud and original is not None:
            # wav_path_for also covers a half-written WAV left by a failed conversion
            audio.cleanup(original, audio.wav_path_for(original))


def run_voice_analysis(
    wav: Path, words: list[Word], task_id: str
) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    """Pitch, intonation and loudness; a failure is reported in the result, never raised."""
    try:
        floor_hz, ceiling_hz = config.pitch_settings()
        return analyse_voice(wav, words, floor_hz, ceiling_hz)
    except Exception as err:
        log.exception("Voice analysis failed for task %s", task_id)
        return {"error": f"{type(err).__name__}: {err}"}, []


def _store(
    db: sqlite3.Connection | None,
    session_id: str,
    task: Task,
    result: TakeResult,
    original: Path,
    wav: Path,
) -> None:
    def write(conn: sqlite3.Connection) -> None:
        storage.record_task(
            conn,
            session_id,
            task.id,
            result.source,
            original,
            wav,
            result.duration_s,
            transcript=result.transcript,
            model=result.model,
            details=result.details,
        )
        storage.replace_metrics(
            conn,
            session_id,
            task.id,
            {name: (value, METRIC_UNITS.get(name)) for name, value in result.metrics.items()},
        )

    if db is not None:
        write(db)
    else:
        with closing(storage.connect()) as conn:
            write(conn)
