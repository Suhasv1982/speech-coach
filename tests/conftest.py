"""Shared test setup: temporary data folders, no API keys and no network access."""

import os
import socket
from pathlib import Path

import pytest

import config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Point data folders at a temp dir and remove any API keys from the environment."""
    monkeypatch.setattr(config, "RECORDINGS_DIR", tmp_path / "recordings")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    for name in list(os.environ):
        if name.endswith(("_API_KEY", "_TOKEN", "_SECRET")):
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
