"""Repeat offenders: reference phrases mis-transcribed in two or more saved sessions.

Sessions come from the local SQLite history (local mode) and from earlier metrics JSON
reports the speaker uploads (either mode; cloud mode keeps no history of its own).
Each "different" entry of a scripted task's comparison is placed on the reference words it
covers; reference words flagged in 2+ sessions are merged into phrases. Entries whose
expected and heard text are equal once numbers are normalised ("forty percent" vs "40",
from reports made before numbers were compared by value) are ignored.
"""

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from analysis import Token, canonicalise, normalise, reference_tokens
from tasks import Task

MIN_SESSIONS = 2

# session id -> task id -> comparison dict (as stored in details["comparison"])
Comparisons = dict[str, dict[str, dict[str, Any]]]


def comparisons_from_report(payload: Mapping[str, Any]) -> Comparisons:
    """Comparisons in a metrics JSON report (as downloaded from the summary page)."""
    found = {
        task["task_id"]: task["comparison"]
        for task in payload.get("tasks", [])
        if isinstance(task, Mapping) and task.get("comparison")
    }
    return {str(payload.get("session_id", "?")): found} if found else {}


def comparisons_from_db(conn: sqlite3.Connection) -> Comparisons:
    """Comparisons stored with every recorded task in the local database."""
    sessions: Comparisons = defaultdict(dict)
    for row in conn.execute(
        "SELECT session_id, task_id, details FROM tasks WHERE details IS NOT NULL"
    ):
        details = json.loads(row["details"])
        if details.get("comparison"):
            sessions[row["session_id"]][row["task_id"]] = details["comparison"]
    return dict(sessions)


def merge(*sources: Comparisons) -> Comparisons:
    """Combine sources; a session seen twice (database and upload) counts once."""
    merged: Comparisons = {}
    for source in sources:
        for session_id, tasks in source.items():
            merged.setdefault(session_id, {}).update(tasks)
    return merged


def _canonical(text: str) -> list[str]:
    return [t.word for t in canonicalise([Token(w, raw=w) for w in normalise(text)])]


def _same_numbers(expected: list[str], heard: list[str]) -> bool:
    """Equal once "%" is ignored: older reports dropped it ("forty percent" heard as "40")."""
    return [w.rstrip("%") for w in expected] == [w.rstrip("%") for w in heard]


def _locate(ref: list[str], phrase: list[str], position: int | None, taken: set[int]) -> int:
    """Start index of phrase in ref: at its recorded position if it fits, else the first
    occurrence not already used by this session; -1 if not found."""
    n = len(phrase)
    if position is not None and ref[position - 1 : position - 1 + n] == phrase:
        return position - 1
    starts = [i for i in range(len(ref) - n + 1) if ref[i : i + n] == phrase]
    for start in starts:
        if start not in taken:
            return start
    return starts[0] if starts else -1


def repeat_offenders(
    sessions: Comparisons, tasks: Iterable[Task], min_sessions: int = MIN_SESSIONS
) -> list[dict[str, Any]]:
    """Reference phrases mis-transcribed in at least min_sessions sessions, most first."""
    offenders: list[dict[str, Any]] = []
    for task in tasks:
        if not task.reference_text:
            continue
        ref_tokens = reference_tokens(task.reference_text)
        ref = [t.word for t in ref_tokens]
        # reference index -> {session id: what was heard there}
        flagged: dict[int, dict[str, str]] = defaultdict(dict)
        for session_id, comparisons in sessions.items():
            comparison = comparisons.get(task.id)
            if not comparison:
                continue
            taken: set[int] = set()
            for entry in comparison.get("different", []):
                expected = _canonical(entry.get("expected", ""))
                if not expected or _same_numbers(expected, _canonical(entry.get("heard", ""))):
                    continue
                start = _locate(ref, expected, entry.get("position"), taken)
                if start < 0:
                    continue
                taken.add(start)
                for index in range(start, start + len(expected)):
                    flagged[index][session_id] = entry.get("heard", "")
        repeated = sorted(i for i, seen in flagged.items() if len(seen) >= min_sessions)
        for run in _runs(repeated):
            heard: dict[str, str] = {}
            for index in run:
                heard.update(flagged[index])
            offenders.append(
                {
                    "task_id": task.id,
                    "phrase": " ".join(ref_tokens[i].shown for i in run),
                    "sessions": len(heard),
                    "heard": [
                        {"session_id": session_id, "heard": text}
                        for session_id, text in sorted(heard.items())
                    ],
                }
            )
    return sorted(offenders, key=lambda o: -o["sessions"])


def _runs(indexes: list[int]) -> list[list[int]]:
    """Group sorted indexes into runs of consecutive numbers."""
    runs: list[list[int]] = []
    for index in indexes:
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs
