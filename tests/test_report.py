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
    "recording": {
        "source": "mic",
        "sample_rate_hz": 48000,
        "channels": "mono",
        "duration_s": 61.2,
        "autoGainControl": False,
        "noiseSuppression": False,
        "echoCancellation": False,
    },
    "voice": {
        "pitch_floor_hz": 75.0,
        "pitch_ceiling_hz": 300.0,
        "voiced_s": 30.5,
        "endings": {"falling": 1, "flat": 1, "rising": 1, "unclear": 1},
        "endings_pct": {"falling": 33.3, "flat": 33.3, "rising": 33.3},
        "statement_falling_pct": 50.0,
        "pitch_range_st": 3.8,
        "monotone": True,
        "trailing_off": [{"time_s": 12.4, "text": "It would also reduce costs.", "drop_db": 8.2}],
        "sentences": [
            {
                "start_s": 0.0,
                "end_s": 6.1,
                "text": "Our proposal is to invest.",
                "ending": "falling",
                "change_st": -2.4,
                "question": False,
                "mean_db": 70.0,
                "last_word_db": 68.0,
                "trailing_off": False,
            },
            {
                "start_s": 6.5,
                "end_s": 12.0,
                "text": "Is that clear?",
                "ending": "rising",
                "change_st": 3.1,
                "question": True,
                "mean_db": 70.0,
                "last_word_db": 69.0,
                "trailing_off": False,
            },
            {
                "start_s": 12.4,
                "end_s": 18.0,
                "text": "It would also reduce costs.",
                "ending": "flat",
                "change_st": 0.2,
                "question": False,
                "mean_db": 70.0,
                "last_word_db": 61.8,
                "trailing_off": True,
            },
            {
                "start_s": 18.5,
                "end_s": 20.0,
                "text": "Thanks.",
                "ending": "unclear",
                "change_st": None,
                "question": False,
                "mean_db": None,
                "last_word_db": None,
                "trailing_off": False,
            },
        ],
    },
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
    "pitch_range_st": 3.8,
    "falling_endings_pct": 50.0,
    "trailing_off_count": 1.0,
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
        pitch_contour=[(0.1, 150.0), (0.15, 152.0)],
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


def test_markdown_shows_voice_section():
    report = build_markdown("s1", TASKS, RESULTS)
    for text in [
        "**Voice: pitch, intonation and volume**",
        "**Sentence endings:** falling 1 (33%) · flat 1 (33%) · rising 1 (33%) · unclear 1",
        "**Falling statement endings:** 50%",
        "**Pitch range:** 3.8 semitones ⚠️ monotone (under 5 st)",
        "**Trailing off:** 0:12.4 “It would also reduce costs.” (−8 dB)",
        "Pitch tracked between 75 and 300 Hz",
        "| 0:00.0 | falling | -2.4 st | Our proposal is to invest. |",
        "| 0:12.4 | flat · trails off | +0.2 st | It would also reduce costs. |",
        "| 0:18.5 | unclear | – | Thanks. |",
        "**Pitch range:** 3.8 st",
        "**Falling statement endings:** 50%",
        "**Trailing-off sentences:** 1",
        "**Recording:** 48000 Hz, mono, 61.2 s, browser processing off",
    ]:
        assert text in report, text


def test_markdown_flags_missing_voice_analysis():
    report = build_markdown("s1", TASKS, RESULTS)
    word_list = report.split("## Word list")[1].split("## ")[0]
    assert "⚠️ Voice analysis was not run" in word_list


def test_markdown_notes_too_little_voiced_speech():
    voice = {**DETAILS["voice"], "pitch_range_st": None, "monotone": False, "voiced_s": 0.4}
    result = TakeResult(**{**vars(RESULTS["read_aloud"]), "details": {**DETAILS, "voice": voice}})
    report = build_markdown("s1", TASKS, {"read_aloud": result})
    assert "**Pitch range:** not measured (only 0.4 s of voiced pitch)" in report


def test_json_includes_voice_but_not_the_contour():
    data = json.loads(build_json("s1", TASKS, RESULTS))
    first = data["tasks"][0]
    voice = first["voice"]
    assert voice["endings"] == {"falling": 1, "flat": 1, "rising": 1, "unclear": 1}
    assert voice["endings_pct"]["falling"] == 33.3
    assert voice["pitch_range_st"] == 3.8 and voice["monotone"] is True
    assert voice["trailing_off"][0]["time_s"] == 12.4
    assert [s["ending"] for s in voice["sentences"]] == ["falling", "rising", "flat", "unclear"]
    assert first["metrics"]["falling_endings_pct"] == 50.0
    assert first["recording"]["sample_rate_hz"] == 48000
    assert "pitch_contour" not in first


def test_markdown_says_when_there_are_no_sentences():
    voice = {**DETAILS["voice"], "sentences": [], "trailing_off": []}
    result = TakeResult(**{**vars(RESULTS["read_aloud"]), "details": {**DETAILS, "voice": voice}})
    report = build_markdown("s1", TASKS, {"read_aloud": result})
    assert "No sentences in the transcript" in report


CLARITY = [
    {
        "word": "architecture",
        "heard": "architecture",
        "time_s": 1.0,
        "confidence": 0.95,
        "status": "clear",
    },
    {
        "word": "analytics",
        "heard": "analytics",
        "time_s": 2.5,
        "confidence": 0.42,
        "status": "unclear",
    },
    {
        "word": "databricks",
        "heard": "data bricks",
        "time_s": 15.4,
        "confidence": None,
        "status": "misheard",
    },
    {"word": "azure", "heard": None, "time_s": None, "confidence": None, "status": "missed"},
]


def word_list_result():
    return TakeResult(
        **{
            **vars(RESULTS["word_list"]),
            "details": {"warnings": [], "word_clarity": CLARITY},
            "metrics": {"duration_s": 20.0, "clear_words_pct": 25.0},
        }
    )


def test_word_list_shows_clarity_instead_of_voice():
    report = build_markdown("s1", TASKS, {"word_list": word_list_result()})
    section = report.split("## Word list")[1].split("## ")[0]
    assert "**Word clarity**" in section
    assert "| architecture | clear |  | 0:01.0 | 95% |" in section
    assert "| analytics | unclear |  | 0:02.5 | 42% |" in section
    assert "| databricks | misheard | data bricks | 0:15.4 | – |" in section
    assert "| azure | missed |  | – | – |" in section
    assert "**Clear words:** 25%" in section
    assert "Voice" not in section and "Sentence endings" not in section


OFFENDERS = [
    {
        "task_id": "word_list",
        "phrase": "databricks",
        "sessions": 2,
        "heard": [
            {"session_id": "a", "heard": "data bricks"},
            {"session_id": "b", "heard": "database"},
        ],
    }
]


def test_repeat_offenders_section_and_json():
    report = build_markdown("s1", TASKS, RESULTS, offenders=OFFENDERS)
    assert "## Repeat offenders" in report
    assert "| databricks | Word list | 2 | “data bricks”; “database” |" in report
    data = json.loads(build_json("s1", TASKS, RESULTS, offenders=OFFENDERS))
    assert data["repeat_offenders"] == OFFENDERS
    empty = build_markdown("s1", TASKS, RESULTS, offenders=[])
    assert "No phrase was mis-transcribed in two or more saved sessions" in empty


def test_header_says_which_mode_chose_the_model():
    cloud = build_markdown("s1", TASKS, RESULTS, mode="cloud")
    assert "Speech model: small.en (int8) (cloud practice mode" in cloud
    assert "local runs use medium.en" in cloud
    assert json.loads(build_json("s1", TASKS, RESULTS, mode="cloud"))["mode"] == "cloud"
    assert "(local mode)" in build_markdown("s1", TASKS, RESULTS, mode="local")
