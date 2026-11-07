"""SQLite storage for assessment sessions, recorded tasks and metrics."""

import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id            INTEGER PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(id),
    task_id       TEXT NOT NULL,
    source        TEXT NOT NULL CHECK (source IN ('mic', 'upload')),
    original_path TEXT NOT NULL,
    wav_path      TEXT NOT NULL,
    duration_s    REAL,
    recorded_at   TEXT NOT NULL,
    UNIQUE (session_id, task_id)
);

CREATE TABLE IF NOT EXISTS metrics (
    id         INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    task_id    TEXT NOT NULL,
    name       TEXT NOT NULL,
    value      REAL NOT NULL,
    unit       TEXT,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the database, creating the file and tables if needed."""
    path = path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def new_session_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def create_session(conn: sqlite3.Connection) -> str:
    session_id = new_session_id()
    with conn:
        conn.execute("INSERT INTO sessions (id, started_at) VALUES (?, ?)", (session_id, _now()))
    return session_id


def record_task(
    conn: sqlite3.Connection,
    session_id: str,
    task_id: str,
    source: str,
    original_path: Path,
    wav_path: Path,
    duration_s: float | None,
) -> None:
    """Store a recorded task, replacing any earlier take of the same task in this session."""
    with conn:
        conn.execute(
            """
            INSERT INTO tasks
                (session_id, task_id, source, original_path, wav_path, duration_s, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (session_id, task_id) DO UPDATE SET
                source = excluded.source,
                original_path = excluded.original_path,
                wav_path = excluded.wav_path,
                duration_s = excluded.duration_s,
                recorded_at = excluded.recorded_at
            """,
            (session_id, task_id, source, str(original_path), str(wav_path), duration_s, _now()),
        )


def add_metric(
    conn: sqlite3.Connection,
    session_id: str,
    task_id: str,
    name: str,
    value: float,
    unit: str | None = None,
) -> None:
    with conn:
        conn.execute(
            "INSERT INTO metrics (session_id, task_id, name, value, unit, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, task_id, name, value, unit, _now()),
        )


def get_session_tasks(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM tasks WHERE session_id = ? ORDER BY recorded_at", (session_id,)
    ).fetchall()


def complete_session(conn: sqlite3.Connection, session_id: str) -> None:
    with conn:
        conn.execute("UPDATE sessions SET completed_at = ? WHERE id = ?", (_now(), session_id))
