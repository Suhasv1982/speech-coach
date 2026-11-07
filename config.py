"""Project settings, loaded from environment variables and an optional .env file."""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")


def _path_setting(name: str, default: str) -> Path:
    path = Path(os.getenv(name) or default)
    return path if path.is_absolute() else ROOT / path


RECORDINGS_DIR = _path_setting("RECORDINGS_DIR", "recordings")
REPORTS_DIR = _path_setting("REPORTS_DIR", "reports")
DB_PATH = _path_setting("DB_PATH", "data/speech_coach.db")
TASKS_FILE = ROOT / "assessment_tasks.yaml"


def ffmpeg_exe() -> str:
    """Return the ffmpeg executable: FFMPEG_PATH, then PATH, then imageio-ffmpeg's bundled copy."""
    configured = os.getenv("FFMPEG_PATH")
    if configured:
        return configured
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()
