"""Shared test setup: temp data folders, no secrets or API keys, no network, a fake model.

CLOUD_MODE is deliberately left alone so the whole suite can run under each mode
(CI runs it with CLOUD_MODE=false and CLOUD_MODE=true). Tests that need a specific
mode set it themselves.
"""

import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

import config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Point data folders at a temp dir and remove secrets, API keys and model overrides."""
    monkeypatch.setattr(config, "RECORDINGS_DIR", tmp_path / "recordings")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(config, "_secrets", lambda: {})
    for name in list(os.environ):
        if name.endswith(("_API_KEY", "_TOKEN", "_SECRET")) or name.startswith("WHISPER_"):
            monkeypatch.delenv(name)


LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that connects beyond this machine; mock API calls instead.

    Loopback stays allowed because asyncio uses it internally on Windows.
    """
    real_connect = socket.socket.connect
    real_create_connection = socket.create_connection

    def check(address):
        if not (isinstance(address, tuple) and address[0] in LOCAL_HOSTS):
            raise RuntimeError("Network access is disabled in tests. Mock the API call instead.")

    def guarded_connect(sock, address):
        check(address)
        return real_connect(sock, address)

    def guarded_create_connection(address, *args, **kwargs):
        check(address)
        return real_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)


@pytest.fixture
def sample_wav() -> Path:
    return FIXTURES / "sample.wav"


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel: returns fixed segments, never downloads."""

    def __init__(self, text: str = "Our proposal is to invest in a unified data platform."):
        self.text = text
        self.calls: list[str] = []

    def transcribe(self, path, **kwargs):
        self.calls.append(path)
        self.kwargs = kwargs
        tokens = self.text.split()
        step = 0.4 / len(tokens)
        words = [
            SimpleNamespace(word=f" {w}", start=i * step, end=(i + 0.9) * step)
            for i, w in enumerate(tokens)
        ]
        segment = SimpleNamespace(text=f" {self.text}", start=0.0, end=0.4, words=words)
        return iter([segment]), SimpleNamespace(language="en")


@pytest.fixture
def fake_model() -> FakeWhisperModel:
    return FakeWhisperModel()
