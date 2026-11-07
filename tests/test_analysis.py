import pytest

from analysis import compute_metrics, normalise
from speech_model import Transcript
from tasks import load_tasks


@pytest.fixture
def tasks():
    return {task.id: task for task in load_tasks()}


def test_normalise_strips_punctuation_and_case():
    assert normalise("Hello, World! It's $2 million.") == ["hello", "world", "it's", "2", "million"]


def test_spontaneous_task_has_pace_but_no_coverage(tasks):
    metrics = compute_metrics(tasks["introduction"], Transcript("one two three four"), 2.0)
    assert metrics == {"duration_s": 2.0, "word_count": 4.0, "words_per_minute": 120.0}


def test_zero_duration_gives_zero_pace(tasks):
    assert compute_metrics(tasks["introduction"], Transcript("hi"), 0.0)["words_per_minute"] == 0


def test_read_aloud_full_reading_has_full_coverage(tasks):
    task = tasks["read_aloud"]
    metrics = compute_metrics(task, Transcript(task.reference_text), 60.0)
    assert metrics["reference_coverage"] == 1.0


def test_word_list_partial_coverage(tasks):
    task = tasks["word_list"]
    first_fifteen = " ".join(task.reference_text.splitlines()[:15])
    metrics = compute_metrics(task, Transcript(first_fifteen), 30.0)
    assert metrics["reference_coverage"] == 0.5
    assert metrics["words_per_minute"] == 30.0
