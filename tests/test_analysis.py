import pytest

from analysis import (
    Token,
    compare_to_reference,
    compute_metrics,
    find_fillers,
    normalise,
    pace_windows,
    pause_metrics,
)
from speech_model import Transcript, Word
from tasks import load_tasks


@pytest.fixture
def tasks():
    return {task.id: task for task in load_tasks()}


def timed(*items: tuple[str, float, float]) -> Transcript:
    words = [Word(text, start, end) for text, start, end in items]
    return Transcript(" ".join(w.text for w in words), words)


def evenly_spaced(text: str, start: float = 0.0, step: float = 0.4) -> list[Word]:
    """Words back to back (0.1 s gaps), step seconds apart."""
    return [
        Word(word, start + i * step, start + i * step + step - 0.1)
        for i, word in enumerate(text.split())
    ]


def test_normalise_strips_punctuation_and_case():
    assert normalise("Hello, World! It's $2 million.") == ["hello", "world", "it's", "2", "million"]


# --- Fillers ---


def test_fillers_found_with_timestamps_including_two_word_fillers():
    tokens = [
        Token("so", 0.0),
        Token("um", 1.2),
        Token("we", 1.6),
        Token("you", 2.0),
        Token("know", 2.2),
        Token("uh", 3.46),
        Token("like", 4.0),
    ]
    fillers, used = find_fillers(tokens)
    assert fillers == [
        {"word": "um", "time_s": 1.2},
        {"word": "you know", "time_s": 2.0},
        {"word": "uh", "time_s": 3.5},
    ]
    assert used == {1, 3, 4, 5}


def test_fillers_per_minute(tasks):
    transcript = timed(("Um,", 0.0, 0.3), ("I", 0.4, 0.5), ("uh", 0.6, 0.8), ("agree.", 0.9, 1.2))
    metrics = compute_metrics(tasks["introduction"], transcript, 30.0).metrics
    assert metrics["filler_count"] == 2
    assert metrics["fillers_per_minute"] == 4.0


# --- Pauses, hesitations, stretches ---


def test_pause_counts_and_mid_phrase_hesitations():
    words = [
        Word("We", 0.0, 0.2),
        Word("invest", 0.3, 0.7),  # 0.1 s gap
        Word("in", 1.5, 1.6),  # 0.8 s gap after "invest" -> mid-phrase hesitation
        Word("data.", 1.7, 2.0),
        Word("Then", 4.0, 4.2),  # 2.0 s gap after "data." -> long pause, not mid-phrase
        Word("we", 4.3, 4.4),
        Word("grow,", 4.5, 4.8),
        Word("fast", 5.5, 5.8),  # 0.7 s gap after comma -> pause, not mid-phrase
    ]
    metrics = pause_metrics(words)
    assert metrics["pauses_over_0_5s"] == 3
    assert metrics["pauses_over_1_5s"] == 1
    assert metrics["mid_phrase_hesitations"] == 1
    # stretches: 0.0-0.7, 1.5-2.0, 4.0-4.8, 5.5-5.8 -> longest 0.8 s
    assert metrics["longest_stretch_s"] == 0.8


def test_longest_stretch_without_pauses_is_whole_speech():
    words = evenly_spaced("one two three four five")
    assert pause_metrics(words)["longest_stretch_s"] == pytest.approx(1.9)
    assert pause_metrics(words)["pauses_over_0_5s"] == 0


def test_no_words_gives_zero_pause_metrics():
    assert pause_metrics([]) == {
        "pauses_over_0_5s": 0,
        "pauses_over_1_5s": 0,
        "mid_phrase_hesitations": 0,
        "longest_stretch_s": 0,
    }


# --- Pace windows ---


def test_pace_per_30_second_window():
    words = evenly_spaced(" ".join(["w"] * 60), start=0.0, step=0.5)  # 60 words in 0-30 s
    words += evenly_spaced(" ".join(["w"] * 30), start=30.0, step=1.0)  # 30 words in 30-60 s
    windows = pace_windows(words, 60.0)
    assert windows == [
        {"start_s": 0.0, "end_s": 30.0, "wpm": 120.0},
        {"start_s": 30.0, "end_s": 60.0, "wpm": 60.0},
    ]


def test_short_final_window_is_merged():
    words = evenly_spaced(" ".join(["w"] * 66), step=0.5)  # 0-33 s
    windows = pace_windows(words, 33.0)
    assert windows == [{"start_s": 0.0, "end_s": 33.0, "wpm": 120.0}]


def test_partial_final_window_uses_its_own_length():
    words = evenly_spaced(" ".join(["w"] * 60), step=0.5) + evenly_spaced("a b c d e", 30.0, 2.0)
    windows = pace_windows(words, 40.0)
    assert windows[1] == {"start_s": 30.0, "end_s": 40.0, "wpm": 30.0}


# --- Reference comparison ---


def test_comparison_lists_differently_transcribed_and_missed_words():
    reference = "the platform positions us to adopt machine learning responsibly"
    spoken = [
        Token(w, t)
        for w, t in [
            ("the", 0.0),
            ("platform", 0.3),
            ("purchases", 0.9),
            ("to", 1.5),
            ("adopt", 1.7),
            ("learning", 2.4),
            ("responsibly", 2.9),
        ]
    ]
    comparison = compare_to_reference(reference, spoken)
    assert comparison["different"] == [
        {"expected": "positions us", "heard": "purchases", "time_s": 0.9}
    ]
    assert comparison["missed"] == [{"expected": "machine", "position": 7}]
    assert comparison["matched_words"] == 6
    assert comparison["reference_words"] == 9


def test_scripted_task_gets_comparison_and_fillers_are_not_mismatches(tasks):
    task = tasks["word_list"]
    words = task.reference_text.splitlines()
    spoken = " ".join(words[:5] + ["um"] + words[5:])
    metrics = compute_metrics(task, Transcript(spoken), 60.0)
    assert metrics.metrics["reference_coverage"] == 1.0
    assert metrics.details["comparison"]["different"] == []
    assert metrics.details["comparison"]["extra"] == []
    assert metrics.details["fillers"] == [{"word": "um", "time_s": None}]


def test_read_aloud_full_reading_has_full_coverage(tasks):
    task = tasks["read_aloud"]
    analysis = compute_metrics(task, Transcript(task.reference_text), 60.0)
    assert analysis.metrics["reference_coverage"] == 1.0
    assert analysis.details["warnings"] == []


def test_spontaneous_task_has_no_comparison(tasks):
    analysis = compute_metrics(tasks["introduction"], Transcript("one two three four"), 2.0)
    assert "reference_coverage" not in analysis.metrics
    assert "comparison" not in analysis.details
    assert analysis.metrics["words_per_minute"] == 120.0


def test_zero_duration_gives_zero_rates(tasks):
    metrics = compute_metrics(tasks["introduction"], Transcript("um hi"), 0.0).metrics
    assert metrics["words_per_minute"] == 0
    assert metrics["fillers_per_minute"] == 0


# --- Word list coverage warning ---


def test_word_list_ending_early_warns_and_suggests_retake(tasks):
    task = tasks["word_list"]
    first_twelve = " ".join(task.reference_text.splitlines()[:12])
    analysis = compute_metrics(task, Transcript(first_twelve), 20.0)

    assert analysis.metrics["reference_coverage"] == 0.4
    [warning] = analysis.details["warnings"]
    assert "Only 12 of 30 words" in warning
    assert "last 18 words" in warning
    assert "ended before you finished" in warning
    assert "re-take" in warning


def test_word_list_at_80_percent_has_no_warning(tasks):
    task = tasks["word_list"]
    first_24 = " ".join(task.reference_text.splitlines()[:24])
    analysis = compute_metrics(task, Transcript(first_24), 60.0)
    assert analysis.metrics["reference_coverage"] == 0.8
    assert analysis.details["warnings"] == []


def test_word_list_low_coverage_mid_list_does_not_blame_the_ending(tasks):
    task = tasks["word_list"]
    words = task.reference_text.splitlines()
    spoken = " ".join(words[:5] + ["mumble"] * 10 + words[15:20] + words[-2:])
    [warning] = compute_metrics(task, Transcript(spoken), 60.0).details["warnings"]
    assert "Only 12 of 30" in warning
    assert "ended before" not in warning


def test_low_coverage_on_read_aloud_has_no_warning(tasks):
    """Only tasks with min_coverage (the word list) warn."""
    analysis = compute_metrics(tasks["read_aloud"], Transcript("our proposal"), 10.0)
    assert analysis.details["warnings"] == []


# --- Pauses measured on the audio (voice activity), not word timestamps ---


def test_pauses_come_from_speech_segments_when_available():
    # Whisper stretched "invest" over the silence, so word gaps show no pause...
    words = [
        Word("Our", 0.0, 0.2),
        Word("proposal", 0.2, 0.54),
        Word("is", 0.54, 1.0),
        Word("to", 1.0, 1.24),
        Word("invest", 1.72, 2.98),
        Word("in", 2.98, 3.3),
        Word("data.", 3.3, 4.1),
        Word("Today", 7.3, 7.6),
        Word("we", 7.7, 8.0),
    ]
    assert pause_metrics(words)["mid_phrase_hesitations"] == 0
    # ...but the audio shows a 1.18 s silence after "to" and 3.16 s after "data."
    segments = [(0.0, 1.25), (2.43, 4.1), (7.26, 8.0)]
    metrics = pause_metrics(words, segments)
    assert metrics["pauses_over_0_5s"] == 2
    assert metrics["pauses_over_1_5s"] == 1
    assert metrics["mid_phrase_hesitations"] == 1
    assert metrics["longest_stretch_s"] == 1.7


def test_pause_after_last_word_is_not_a_hesitation():
    words = [Word("one", 0.0, 0.5), Word("two", 0.6, 1.0)]
    metrics = pause_metrics(words, [(0.0, 1.0), (3.0, 3.4)])  # trailing noise, no words
    assert metrics["pauses_over_0_5s"] == 1
    assert metrics["mid_phrase_hesitations"] == 0


def test_trailing_ellipsis_counts_as_hesitation():
    words = [Word("to...", 1.0, 1.2), Word("invest", 2.5, 2.9)]
    assert pause_metrics(words)["mid_phrase_hesitations"] == 1


def test_word_list_has_no_mid_phrase_hesitation_metric(tasks):
    words = evenly_spaced("architecture analytics governance", step=1.5)
    segments = [(w.start, w.start + 0.6) for w in words]
    transcript = Transcript("architecture analytics governance", words, 0.0, segments)
    metrics = compute_metrics(tasks["word_list"], transcript, 5.0).metrics
    assert "mid_phrase_hesitations" not in metrics
    assert metrics["pauses_over_0_5s"] == 2
    read = compute_metrics(tasks["read_aloud"], transcript, 5.0).metrics
    assert read["mid_phrase_hesitations"] == 2
