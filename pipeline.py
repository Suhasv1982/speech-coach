"""Process one take: save, convert, transcribe, measure, then store or clean up."""

import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import audio
import storage
from analysis import METRIC_UNITS, compute_metrics
from speech_model import transcribe
from tasks import Task


@dataclass
class TakeResult:
    task_id: str
    source: str
    duration_s: float
    transcript: str
    metrics: dict[str, float] = field(default_factory=dict)
    wav_path: str | None = None  # kept only in local mode


def process_take(
    *,
    task: Task,
    session_id: str,
    data: bytes,
    ext: str,
    source: str,
    model: Any,
    cloud: bool,
    base_dir: Path | None = None,
    db: sqlite3.Connection | None = None,
) -> TakeResult:
    """Run the full analysis for one recording.

    Cloud mode: files live in base_dir (a temp folder) and are always deleted afterwards,
    whether analysis succeeds or fails. Local mode: files are kept and results go to SQLite.
    """
    original = None
    try:
        original = audio.save_recording(session_id, task.id, data, ext, base_dir=base_dir)
        wav = audio.convert_to_wav(original)
        duration = audio.wav_duration_s(wav)
        transcript = transcribe(model, wav)
        metrics = compute_metrics(task, transcript, duration)
        result = TakeResult(
            task_id=task.id,
            source=source,
            duration_s=duration,
            transcript=transcript.text,
            metrics=metrics,
            wav_path=None if cloud else str(wav),
        )
        if not cloud:
            _store(db, session_id, task, result, original, wav)
        return result
    finally:
        if cloud and original is not None:
            # wav_path_for also covers a half-written WAV left by a failed conversion
            audio.cleanup(original, audio.wav_path_for(original))


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
