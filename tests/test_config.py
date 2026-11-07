import imageio_ffmpeg
import pytest
import streamlit as st

import config

SENTINEL = "s3cr3t-value-must-not-leak"
REAL_SECRETS_READER = config._secrets  # captured before conftest stubs it per test


@pytest.fixture
def secrets(monkeypatch):
    """Replace st.secrets with a dict the test can fill in."""
    values: dict[str, str] = {}
    monkeypatch.setattr(config, "_secrets", lambda: values)
    return values


def test_secret_wins_over_env(secrets, monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "from-env")
    secrets["WHISPER_MODEL"] = "from-secrets"
    assert config.get_setting("WHISPER_MODEL") == "from-secrets"


def test_env_used_when_no_secret(secrets, monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "from-env")
    assert config.get_setting("WHISPER_MODEL") == "from-env"


def test_default_used_when_neither_is_set(secrets, monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    assert config.get_setting("WHISPER_MODEL", "fallback") == "fallback"


def test_empty_secret_falls_through_to_env(secrets, monkeypatch):
    secrets["WHISPER_MODEL"] = ""
    monkeypatch.setenv("WHISPER_MODEL", "from-env")
    assert config.get_setting("WHISPER_MODEL") == "from-env"


def test_real_secrets_reader_uses_st_secrets(monkeypatch):
    monkeypatch.setattr(config, "_secrets", REAL_SECRETS_READER)
    monkeypatch.setattr(st, "secrets", {"WHISPER_MODEL": "from-st-secrets"})
    assert config.get_setting("WHISPER_MODEL") == "from-st-secrets"


def test_missing_secrets_file_falls_back_to_env(monkeypatch):
    """The real reader must swallow Streamlit's 'no secrets file' error."""

    class NoSecretsFile:
        def keys(self):
            raise FileNotFoundError("No secrets files found")

        def __iter__(self):
            raise FileNotFoundError("No secrets files found")

    monkeypatch.setattr(config, "_secrets", REAL_SECRETS_READER)
    monkeypatch.setattr(st, "secrets", NoSecretsFile())
    monkeypatch.setenv("WHISPER_MODEL", "from-env")
    assert config.get_setting("WHISPER_MODEL") == "from-env"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("TRUE", True), ("1", True), ("yes", True), ("on", True),
     ("false", False), ("0", False), ("no", False)],
)  # fmt: skip
def test_cloud_mode_parsing(secrets, monkeypatch, raw, expected):
    monkeypatch.setenv("CLOUD_MODE", raw)
    assert config.cloud_mode() is expected


def test_cloud_mode_defaults_to_false(secrets, monkeypatch):
    monkeypatch.delenv("CLOUD_MODE", raising=False)
    assert config.cloud_mode() is False


def test_cloud_mode_from_secrets(secrets, monkeypatch):
    monkeypatch.setenv("CLOUD_MODE", "false")
    secrets["CLOUD_MODE"] = "true"
    assert config.cloud_mode() is True


def test_settings_summary_shows_sources_never_values(secrets, monkeypatch):
    secrets["WHISPER_MODEL"] = SENTINEL
    monkeypatch.setenv("FFMPEG_PATH", SENTINEL)
    summary = config.settings_summary()
    assert SENTINEL not in summary
    assert "WHISPER_MODEL=secrets" in summary
    assert "FFMPEG_PATH=env" in summary


def test_ffmpeg_path_setting_wins(monkeypatch):
    monkeypatch.setenv("FFMPEG_PATH", "/custom/ffmpeg")
    assert config.ffmpeg_exe() == "/custom/ffmpeg"


def test_ffmpeg_on_path_is_used_next(monkeypatch):
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.setattr(config.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert config.ffmpeg_exe() == "/usr/bin/ffmpeg"


def test_bundled_ffmpeg_is_the_fallback(monkeypatch):
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.setattr(config.shutil, "which", lambda name: None)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "/bundled/ffmpeg")
    assert config.ffmpeg_exe() == "/bundled/ffmpeg"
