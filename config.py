"""Project settings: Streamlit secrets first, then environment variables / .env, then defaults.

On Streamlit Community Cloud (detected by the app running from /mount/src) the defaults
switch to cloud mode; secrets and env still override. Setting values are never logged;
startup_report() says only where each one came from.
"""

import os
import platform
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
COMMUNITY_CLOUD_ROOT = "/mount/src/"

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


def _read_secrets() -> tuple[Mapping[str, Any], str]:
    """Return (st.secrets as a dict, status). Status never includes secret values."""
    try:
        import streamlit as st

        secrets = dict(st.secrets)
    except Exception as err:
        # Streamlit uses one error class for "no file" and "file could not be parsed".
        if str(err).startswith("No secrets found"):
            return {}, "no secrets file"
        return {}, f"unreadable ({type(err).__name__}); check the TOML syntax"
    return secrets, f"loaded ({len(secrets)} keys)"


def _secrets() -> Mapping[str, Any]:
    """Return st.secrets as a mapping, or {} when there is none (then env is used)."""
    return _read_secrets()[0]


def on_community_cloud() -> bool:
    """True when running on Streamlit Community Cloud, which serves apps from /mount/src."""
    return ROOT.as_posix().startswith(COMMUNITY_CLOUD_ROOT)


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
    """CLOUD_MODE from secrets/env; defaults to on when Community Cloud is detected."""
    return get_bool("CLOUD_MODE", default=on_community_cloud())


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


def startup_report() -> str:
    """One log line for startup: environment detection and setting sources, no values."""
    return (
        f"Community Cloud detected: {on_community_cloud()} (app root {ROOT.as_posix()}); "
        f"cloud mode: {cloud_mode()}; Python {platform.python_version()}; "
        f"secrets: {_read_secrets()[1]}; sources: {settings_summary()}"
    )


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
