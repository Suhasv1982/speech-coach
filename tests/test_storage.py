import re
import sqlite3
from contextlib import closing

import pytest

import config
import storage


@pytest.fixture
def conn():
    with closing(storage.connect()) as connection:
        yield connection


def test_connect_creates_database_and_tables(conn):
    assert config.DB_PATH.exists()
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sessions", "tasks", "metrics"} <= tables


def test_create_session_id_format(conn):
    session_id = storage.create_session(conn)
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}", session_id)
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    assert row["started_at"] and row["completed_at"] is None


def test_record_task_replaces_earlier_take(conn, tmp_path):
    session_id = storage.create_session(conn)
    storage.record_task(
        conn, session_id, "intro", "mic", tmp_path / "a.wav", tmp_path / "a_16k.wav", 5
    )
    storage.record_task(
        conn, session_id, "intro", "upload", tmp_path / "b.mp4", tmp_path / "b_16k.wav", 7.5
    )

    rows = storage.get_session_tasks(conn, session_id)
    assert len(rows) == 1
    assert rows[0]["source"] == "upload"
    assert rows[0]["duration_s"] == 7.5


def test_record_task_rejects_unknown_source(conn, tmp_path):
    session_id = storage.create_session(conn)
    with pytest.raises(sqlite3.IntegrityError):
        storage.record_task(conn, session_id, "intro", "email", tmp_path, tmp_path, 1)


def test_old_database_gets_transcript_column():
    """A database created by session 1 (no transcript column) is upgraded on connect."""
    config.DB_PATH.parent.mkdir(parents=True)
    with closing(sqlite3.connect(config.DB_PATH)) as old, old:
        old.execute(
            "CREATE TABLE tasks (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,"
            " task_id TEXT NOT NULL, source TEXT NOT NULL, original_path TEXT NOT NULL,"
            " wav_path TEXT NOT NULL, duration_s REAL, recorded_at TEXT NOT NULL,"
            " UNIQUE (session_id, task_id))"
        )

    with closing(storage.connect()) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        assert "transcript" in columns
        session_id = storage.create_session(conn)
        storage.record_task(conn, session_id, "intro", "mic", "a", "b", 1.0, transcript="hi")
        assert storage.get_session_tasks(conn, session_id)[0]["transcript"] == "hi"


def test_add_metric_and_complete_session(conn):
    session_id = storage.create_session(conn)
    storage.add_metric(conn, session_id, "read_aloud", "words_per_minute", 142.0, "wpm")
    storage.complete_session(conn, session_id)

    metric = conn.execute("SELECT * FROM metrics").fetchone()
    assert (metric["name"], metric["value"], metric["unit"]) == ("words_per_minute", 142.0, "wpm")
    session = conn.execute("SELECT completed_at FROM sessions").fetchone()
    assert session["completed_at"] is not None
