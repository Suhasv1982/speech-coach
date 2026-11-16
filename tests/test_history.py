import json
from contextlib import closing
from pathlib import Path

import storage
from history import comparisons_from_db, comparisons_from_report, merge, repeat_offenders
from tasks import load_tasks

TASKS = load_tasks()


def report(session_id, read_aloud=(), word_list=()):
    """A metrics JSON payload like the downloaded reports (only the fields used here)."""
    tasks = []
    for task_id, different in (("read_aloud", read_aloud), ("word_list", word_list)):
        entries = [{"expected": e, "heard": h, "time_s": 1.0} for e, h in different]
        tasks.append({"task_id": task_id, "comparison": {"different": entries, "missed": []}})
    tasks.append({"task_id": "introduction", "title": "Introduction"})
    return {"session_id": session_id, "tasks": tasks}


# Mis-transcriptions from the saved sessions of 2026-11-09 to 2026-11-16 (small.en)
SAVED = [
    report(
        "20261109",
        [("us to adopt", "are stored out"), ("forty percent", "40"), ("governed", "government")],
        [("databricks", "data bricks")],
    ),
    report(
        "20261112",
        [("positions us to adopt", "produces a start up"), ("forty percent", "40")],
        [("databricks", "data bricks"), ("catalog", "kerala")],
    ),
    report(
        "20261113",
        [("positions us", "purchase has"), ("eighteen", "18")],
        [("databricks", "database")],
    ),
    report(
        "20261116",
        [("positions us to adopt", "publishers are stored up"), ("quarter", "quorum")],
        [("databricks", "development data breaks")],
    ),
]


def offenders_from(*payloads):
    return repeat_offenders(merge(*map(comparisons_from_report, payloads)), TASKS)


def test_phrases_misheard_in_two_or_more_sessions():
    offenders = offenders_from(*SAVED)
    found = {(o["task_id"], o["phrase"]): o for o in offenders}
    assert set(found) == {("read_aloud", "positions us to adopt"), ("word_list", "databricks")}
    databricks = found[("word_list", "databricks")]
    assert databricks["sessions"] == 4
    assert [h["heard"] for h in databricks["heard"]] == [
        "data bricks",
        "data bricks",
        "database",
        "development data breaks",
    ]
    # different boundaries in each session ("us to adopt", "positions us") join into one phrase
    assert found[("read_aloud", "positions us to adopt")]["sessions"] == 4


def test_one_off_mistakes_and_old_number_differences_are_not_offenders():
    phrases = [o["phrase"] for o in offenders_from(*SAVED)]
    assert "quarter" not in phrases and "catalog" not in phrases  # one session each
    assert "forty percent" not in phrases  # "40" is right; old reports flagged it
    assert "eighteen" not in phrases


def test_a_single_session_has_no_repeat_offenders():
    assert offenders_from(SAVED[0]) == []


def test_repeated_word_is_placed_by_position_when_known():
    # "would" appears twice in the read-aloud text; positions keep them apart
    ref_would = [{"expected": "would", "heard": "will", "position": 83}]  # 2nd "would"
    sessions = {
        "a": {"read_aloud": {"different": ref_would}},
        "b": {"read_aloud": {"different": [{"expected": "would", "heard": "should"}]}},
    }
    assert repeat_offenders(sessions, TASKS) == []  # first "would" vs the later one


def test_same_session_from_database_and_upload_counts_once():
    assert offenders_from(SAVED[0], SAVED[0]) == []


def test_reports_without_comparisons_are_ignored():
    assert comparisons_from_report({"session_id": "x", "tasks": []}) == {}
    assert comparisons_from_report({"tasks": [{"task_id": "introduction"}]}) == {}


def test_comparisons_are_read_from_the_local_database():
    with closing(storage.connect()) as conn:
        for payload in SAVED[:2]:
            session_id = storage.create_session(conn)
            for task in payload["tasks"]:
                if "comparison" in task:
                    storage.record_task(
                        conn,
                        session_id,
                        task["task_id"],
                        "mic",
                        Path("take.webm"),
                        Path("take_16k.wav"),
                        10.0,
                        details={"comparison": task["comparison"]},
                    )
        sessions = comparisons_from_db(conn)
    assert len(sessions) == 2
    offenders = repeat_offenders(sessions, TASKS)
    assert ("word_list", "databricks", 2) in {
        (o["task_id"], o["phrase"], o["sessions"]) for o in offenders
    }
    json.dumps(offenders)  # stored in the metrics JSON
