import pytest

from tasks import load_tasks

EXPECTED_WORDS = """
architecture analytics governance infrastructure strategic migration compliance lineage
Databricks Azure catalog pipeline development executive initiative capability technology
organization determine competitive regulatory enterprise revenue priority framework category
data schedule process focus
""".split()


@pytest.fixture
def tasks():
    return {task.id: task for task in load_tasks()}


def test_five_tasks_in_order():
    assert [task.id for task in load_tasks()] == [
        "read_aloud",
        "word_list",
        "introduction",
        "executive_question",
        "project_story",
    ]


def test_time_limits(tasks):
    assert {task_id: task.time_limit_s for task_id, task in tasks.items()} == {
        "read_aloud": 90,
        "word_list": 90,
        "introduction": 90,
        "executive_question": 120,
        "project_story": 180,
    }


def test_word_list_has_the_30_words_one_per_line(tasks):
    assert tasks["word_list"].reference_text.splitlines() == EXPECTED_WORDS


def test_read_aloud_is_about_150_words(tasks):
    assert 130 <= len(tasks["read_aloud"].reference_text.split()) <= 170


def test_only_scripted_tasks_have_reference_text(tasks):
    for task in tasks.values():
        assert bool(task.reference_text) == (task.type == "scripted"), task.id


def test_executive_question_is_hidden_until_start(tasks):
    task = tasks["executive_question"]
    assert task.reveal_on_start
    assert "$2 million" in task.prompt
    assert "$2 million" not in task.instructions


def test_invalid_task_file_is_rejected(tmp_path):
    bad = tmp_path / "tasks.yaml"
    bad.write_text(
        "tasks:\n  - {id: a, title: A, type: scripted, time_limit_s: 10, instructions: x}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reference_text"):
        load_tasks(bad)
