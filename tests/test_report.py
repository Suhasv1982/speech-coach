import json

from pipeline import TakeResult
from report import build_json, build_markdown
from tasks import load_tasks

TASKS = load_tasks()
RESULTS = {
    "read_aloud": TakeResult(
        task_id="read_aloud",
        source="mic",
        duration_s=61.2,
        transcript="Our proposal is to invest.",
        metrics={"duration_s": 61.2, "word_count": 5.0, "words_per_minute": 4.9,
                 "reference_coverage": 0.04},
    )
}  # fmt: skip


def test_markdown_lists_every_task_and_marks_missing_ones():
    report = build_markdown("s1", TASKS, RESULTS)
    for task in TASKS:
        assert f"## {task.title}" in report
    assert "Our proposal is to invest." in report
    assert "**Reference words spoken:** 4%" in report
    assert report.count("_Not recorded._") == len(TASKS) - 1


def test_json_round_trips():
    data = json.loads(build_json("s1", TASKS, RESULTS))
    assert data["session_id"] == "s1"
    first = data["tasks"][0]
    assert first["task_id"] == "read_aloud"
    assert first["metrics"]["words_per_minute"] == 4.9
    assert data["tasks"][1] == {"task_id": "word_list", "title": "Word list", "recorded": False}
