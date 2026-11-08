import json

from pipeline import TakeResult
from report import build_json, build_markdown, format_time
from tasks import load_tasks

TASKS = load_tasks()
DETAILS = {
    "fillers": [{"word": "um", "time_s": 12.3}, {"word": "you know", "time_s": 65.0}],
    "pace_windows": [
        {"start_s": 0.0, "end_s": 30.0, "wpm": 142.0},
        {"start_s": 30.0, "end_s": 61.2, "wpm": 118.0},
    ],
    "comparison": {
        "reference_words": 149,
        "matched_words": 140,
        "last_matched_position": 149,
        "missed": [{"expected": "machine", "position": 120}],
        "different": [{"expected": "positions us", "heard": "purchases", "time_s": 50.2}],
        "extra": [],
    },
    "warnings": [],
}
METRICS = {
    "duration_s": 61.2,
    "word_count": 140.0,
    "words_per_minute": 137.3,
    "filler_count": 2.0,
    "fillers_per_minute": 2.0,
    "pauses_over_0_5s": 6.0,
    "pauses_over_1_5s": 1.0,
    "mid_phrase_hesitations": 2.0,
    "longest_stretch_s": 14.2,
    "reference_coverage": 0.94,
}
RESULTS = {
    "read_aloud": TakeResult(
        task_id="read_aloud",
        source="mic",
        duration_s=61.2,
        transcript="Our proposal is to invest.",
        metrics=METRICS,
        model="small.en (int8)",
        details=DETAILS,
    ),
    "word_list": TakeResult(
        task_id="word_list",
        source="mic",
        duration_s=20.0,
        transcript="architecture analytics",
        metrics={"duration_s": 20.0, "reference_coverage": 0.067},
        model="small.en (int8)",
        details={"warnings": ["Only 2 of 30 words were recognised. Consider a re-take."]},
    ),
}


def test_format_time():
    assert format_time(5.0) == "0:05.0"
    assert format_time(65.25) == "1:05.2"
    assert format_time(None) == "?"


def test_header_names_the_speech_model():
    report = build_markdown("s1", TASKS, RESULTS)
    assert "Speech model: small.en (int8)" in report.split("## ")[0]


def test_header_lists_every_model_when_takes_differ():
    results = {**RESULTS}
    results["word_list"] = TakeResult(**{**vars(RESULTS["word_list"]), "model": "base.en (int8)"})
    assert "Speech model: base.en (int8), small.en (int8)" in build_markdown("s1", TASKS, results)


def test_markdown_shows_all_new_metrics():
    report = build_markdown("s1", TASKS, RESULTS)
    for text in [
        "**Fillers per minute:** 2.0",
        "**Pauses > 0.5 s:** 6",
        "**Pauses > 1.5 s:** 1",
        "**Mid-phrase hesitations:** 2",
        "**Longest unbroken stretch:** 14.2 s",
        "**Pace by 30 s window:** 0:00.0–0:30.0: 142 wpm · 0:30.0–1:01.2: 118 wpm",
        "**Fillers heard:** “um” at 0:12.3, “you know” at 1:05.0",
    ]:
        assert text in report, text


def test_markdown_shows_pronunciation_clues():
    report = build_markdown("s1", TASKS, RESULTS)
    assert "| 0:50.2 | positions us | purchases |" in report
    assert "Missed (not heard at all): “machine”" in report


def test_markdown_shows_warnings_and_unrecorded_tasks():
    report = build_markdown("s1", TASKS, RESULTS)
    assert "> ⚠️ Only 2 of 30 words were recognised." in report
    assert report.count("_Not recorded._") == len(TASKS) - 2


def test_json_includes_model_metrics_and_details():
    data = json.loads(build_json("s1", TASKS, RESULTS))
    assert data["speech_models"] == ["small.en (int8)"]
    first = data["tasks"][0]
    assert first["model"] == "small.en (int8)"
    assert first["metrics"]["mid_phrase_hesitations"] == 2.0
    assert first["fillers"][0] == {"word": "um", "time_s": 12.3}
    assert first["pace_windows"][1]["wpm"] == 118.0
    assert first["comparison"]["different"][0]["heard"] == "purchases"
    assert data["tasks"][1]["warnings"]
    assert data["tasks"][2] == {
        "task_id": "introduction",
        "title": "Introduction",
        "recorded": False,
    }


def test_long_missed_passages_are_shortened_in_markdown_but_not_json():
    long_phrase = " ".join(f"w{i}" for i in range(20))
    result = TakeResult(
        **{
            **vars(RESULTS["read_aloud"]),
            "details": {
                **DETAILS,
                "comparison": {**DETAILS["comparison"], "missed": [{"expected": long_phrase}]},
            },
        }
    )
    report = build_markdown("s1", TASKS, {"read_aloud": result})
    assert "w0 w1 w2 w3 w4 w5 w6 w7 … (20 words)" in report
    assert "w19" not in report
    data = json.loads(build_json("s1", TASKS, {"read_aloud": result}))
    assert data["tasks"][0]["comparison"]["missed"][0]["expected"] == long_phrase
