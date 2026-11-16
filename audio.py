"""Save recordings and convert them to 16 kHz mono WAV with ffmpeg."""

import re
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


_STREAM = re.compile(r"Audio: .*?(\d+) Hz, ([^,]+)")
_DURATION = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")


def probe(src: Path) -> dict[str, float | int | str | None]:
    """Sample rate, channel layout and duration of the original recording, from ffmpeg -i."""
    result = subprocess.run(
        [config.ffmpeg_exe(), "-hide_banner", "-i", str(src)], capture_output=True, text=True
    )
    info: dict[str, float | int | str | None] = {
        "sample_rate_hz": None,
        "channels": None,
        "duration_s": None,
    }
    if stream := _STREAM.search(result.stderr):
        info["sample_rate_hz"] = int(stream.group(1))
        info["channels"] = stream.group(2).strip()
    if duration := _DURATION.search(result.stderr):
        hours, minutes, seconds = duration.groups()
        info["duration_s"] = round(int(hours) * 3600 + int(minutes) * 60 + float(seconds), 2)
    return info
