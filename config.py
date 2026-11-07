"""Project settings: Streamlit secrets first, then environment variables / .env, then defaults.

Setting values are never logged; settings_summary() reports only where each one came from.
"""

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")

SETTING_NAMES = (
    "CLOUD_MODE",
    "WHISPER_MODEL",
    "WHISPER_COMPUTE_TYPE",
    "FFMPEG_PATH",
    "RECORDINGS_DIR",
    "REPORTS_DIR",
    "DB_PATH",
)

CLOUD_WHISPER = ("small.en", "int8")
LOCAL_WHISPER = ("medium.en", "auto")


def _secrets() -> Mapping[str, Any]:
    """Return st.secrets as a mapping, or {} when there is no secrets file."""
    try:
        import streamlit as st

        return dict(st.secrets)
    except Exception:  # no secrets.toml, or parse error: fall back to env
        return {}


def _lookup(name: str) -> tuple[str | None, str]:
    secrets = _secrets()
    if name in secrets and str(secrets[name]) != "":
        return str(secrets[name]), "secrets"
    value = os.getenv(name)
    if value:
        return value, "env"
    return None, "default"


def get_setting(name: str, default: str | None = None) -> str | None:
    value, _ = _lookup(name)
    return default if value is None else value


def get_bool(name: str, default: bool = False) -> bool:
    value = get_setting(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def cloud_mode() -> bool:
    return get_bool("CLOUD_MODE")


def whisper_settings() -> tuple[str, str]:
    """Return (model name, compute type): lighter defaults in cloud mode, overridable."""
    model, compute_type = CLOUD_WHISPER if cloud_mode() else LOCAL_WHISPER
    return (
        get_setting("WHISPER_MODEL", model),
        get_setting("WHISPER_COMPUTE_TYPE", compute_type),
    )


def settings_summary() -> str:
    """Describe where each setting comes from, without revealing any value."""
    return ", ".join(f"{name}={_lookup(name)[1]}" for name in SETTING_NAMES)


def _path_setting(name: str, default: str) -> Path:
    path = Path(get_setting(name, default))
    return path if path.is_absolute() else ROOT / path


RECORDINGS_DIR = _path_setting("RECORDINGS_DIR", "recordings")
REPORTS_DIR = _path_setting("REPORTS_DIR", "reports")
DB_PATH = _path_setting("DB_PATH", "data/speech_coach.db")
TASKS_FILE = ROOT / "assessment_tasks.yaml"


def ffmpeg_exe() -> str:
    """Return the ffmpeg executable: FFMPEG_PATH, then PATH, then imageio-ffmpeg's bundled copy."""
    configured = get_setting("FFMPEG_PATH")
    if configured:
        return configured
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()
