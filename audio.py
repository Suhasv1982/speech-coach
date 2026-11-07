"""Save recordings and convert them to 16 kHz mono WAV with ffmpeg."""

import subprocess
import wave
from pathlib import Path

import config

UPLOAD_TYPES = ["wav", "mp3", "m4a", "ogg", "webm", "mp4", "mov"]


class ConversionError(RuntimeError):
    """ffmpeg could not convert a recording."""


def session_dir(session_id: str, base_dir: Path | None = None) -> Path:
    path = (base_dir or config.RECORDINGS_DIR) / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_recording(
    session_id: str, task_id: str, data: bytes, ext: str, base_dir: Path | None = None
) -> Path:
    """Write <base_dir>/<session_id>/<task_id>.<ext>, replacing any earlier take.

    base_dir defaults to the recordings folder; cloud mode passes a temp folder.
    """
    ext = ext.lower().lstrip(".")
    folder = session_dir(session_id, base_dir)
    for old in folder.glob(f"{task_id}.*"):
        old.unlink()
    path = folder / f"{task_id}.{ext}"
    path.write_bytes(data)
    return path


def wav_path_for(src: Path) -> Path:
    return src.with_name(f"{src.stem}_16k.wav")


def convert_to_wav(src: Path) -> Path:
    """Convert any audio or video file to <stem>_16k.wav (16 kHz, mono, 16-bit) next to it."""
    dst = wav_path_for(src)
    cmd = [
        config.ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise ConversionError(f"Could not convert {src.name} to WAV: {detail}")
    return dst


def cleanup(*paths: Path | None) -> None:
    """Delete the given files, ignoring ones that are missing or None."""
    for path in paths:
        if path is not None:
            path.unlink(missing_ok=True)


def wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()
