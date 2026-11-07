import platform
from pathlib import PurePosixPath

import imageio_ffmpeg
import pytest
import streamlit as st

import config

SENTINEL = "s3cr3t-value-must-not-leak"
REAL_SECRETS_READER = config._secrets  # captured before conftest stubs it per test
REAL_READ_SECRETS = config._read_secrets


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


@pytest.fixture
def on_cloud(monkeypatch):
    """Pretend the app runs from Community Cloud's /mount/src folder."""
    monkeypatch.setattr(config, "ROOT", PurePosixPath("/mount/src/speech-coach"))
    monkeypatch.delenv("CLOUD_MODE", raising=False)


def test_community_cloud_detected_from_mount_src(on_cloud):
    assert config.on_community_cloud() is True


def test_not_detected_elsewhere(monkeypatch):
    for root in ("/home/runner/work/speech-coach", "/mount/srcfoo/app", "C:/speech-coach"):
        monkeypatch.setattr(config, "ROOT", PurePosixPath(root))
        assert config.on_community_cloud() is False, root


def test_community_cloud_defaults_to_cloud_mode_and_small_model(on_cloud, secrets):
    assert config.cloud_mode() is True
    assert config.whisper_settings() == ("small.en", "int8")


def test_secret_can_turn_cloud_mode_off_on_community_cloud(on_cloud, secrets):
    secrets["CLOUD_MODE"] = False  # TOML boolean, as pasted into the Secrets box
    assert config.cloud_mode() is False
    assert config.whisper_settings() == ("medium.en", "auto")


def test_toml_boolean_true_secret_turns_cloud_mode_on(secrets, monkeypatch):
    monkeypatch.delenv("CLOUD_MODE", raising=False)
    secrets["CLOUD_MODE"] = True  # `CLOUD_MODE = true` without quotes
    assert config.cloud_mode() is True


def test_env_overrides_detection_on_community_cloud(on_cloud, monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "base.en")
    assert config.whisper_settings() == ("base.en", "int8")


def test_local_run_defaults_to_local_mode(secrets, monkeypatch):
    monkeypatch.delenv("CLOUD_MODE", raising=False)
    monkeypatch.setattr(config, "ROOT", PurePosixPath("C:/speech-coach"))
    assert config.cloud_mode() is False


def test_startup_report_logs_detection_and_no_values(on_cloud, secrets, monkeypatch):
    secrets["WHISPER_MODEL"] = SENTINEL
    monkeypatch.setattr(config, "_read_secrets", lambda: (secrets, "loaded (1 keys)"))
    report = config.startup_report()
    assert "Community Cloud detected: True" in report
    assert "cloud mode: True" in report
    assert "secrets: loaded (1 keys)" in report
    assert SENTINEL not in report


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (FileNotFoundError("No secrets found. Valid paths ..."), "no secrets file"),
        (FileNotFoundError("Error parsing secrets file at /x: bad"), "unreadable"),
    ],
)
def test_secrets_status_distinguishes_missing_from_broken(monkeypatch, error, status):
    class BrokenSecrets:
        def keys(self):
            raise error

        def __iter__(self):
            raise error

    monkeypatch.setattr(st, "secrets", BrokenSecrets())
    secrets, message = REAL_READ_SECRETS()
    assert secrets == {}
    assert message.startswith(status)
    assert "/x" not in message


def test_startup_report_includes_python_version():
    assert f"Python {platform.python_version()}" in config.startup_report()


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
