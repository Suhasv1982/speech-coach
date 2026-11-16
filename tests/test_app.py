import pytest
from streamlit.testing.v1 import AppTest

import config
from app import CLOUD_BANNER
from pipeline import TakeResult


def run_app() -> AppTest:
    return AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=30).run()


def banner_shown(at: AppTest) -> bool:
    return any(CLOUD_BANNER in info.value for info in at.info)


def test_start_page_lists_the_tasks():
    at = run_app()
    assert not at.exception
    assert at.title[0].value == "🎙️ Speech Coach"
    assert at.button[0].label == "Start new assessment"


def test_starting_a_session_opens_first_task():
    at = run_app()
    at.button[0].click().run()

    assert not at.exception
    assert at.header[0].value == "Read aloud"
    session_id = at.session_state.session_id
    if config.cloud_mode():
        assert at.session_state.temp_dir
        assert not (config.RECORDINGS_DIR / session_id).exists()
    else:
        assert (config.RECORDINGS_DIR / session_id).is_dir()


def test_executive_question_hidden_until_start():
    at = run_app()
    at.button[0].click().run()
    at.session_state.page = 4
    at.run()

    assert at.header[0].value == "Executive question"
    assert not any("$2 million" in info.value for info in at.info)

    next(b for b in at.button if b.label == "Start").click().run()
    assert any("$2 million" in info.value for info in at.info)


@pytest.mark.parametrize("cloud", [True, False])
def test_banner_only_in_cloud_mode(monkeypatch, cloud):
    monkeypatch.setenv("CLOUD_MODE", "true" if cloud else "false")
    assert banner_shown(run_app()) is cloud


def test_word_list_warning_offers_retake_that_clears_the_result():
    at = run_app()
    at.button[0].click().run()
    at.session_state.results = {
        "word_list": TakeResult(
            task_id="word_list",
            source="mic",
            duration_s=20.0,
            transcript="architecture analytics",
            metrics={"duration_s": 20.0, "reference_coverage": 0.067},
            model="small.en (int8)",
            details={
                "warnings": ["Only 2 of 30 words were recognised. Consider a re-take."],
                "comparison": {"missed": [], "different": [], "extra": []},
            },
        )
    }
    at.session_state.page = 2
    at.run()

    assert not at.exception
    assert any("Only 2 of 30 words" in w.value for w in at.warning)
    assert any("small.en (int8)" in s.value for s in at.success)
    at.button(key="retake_word_list").click().run()

    assert "word_list" not in at.session_state.results
    assert at.session_state.takes["word_list"] == 2
    assert not any("Only 2 of 30" in w.value for w in at.warning)


@pytest.mark.parametrize("cloud", [True, False])
def test_summary_offers_downloads(monkeypatch, cloud):
    monkeypatch.setenv("CLOUD_MODE", "true" if cloud else "false")
    at = run_app()
    at.button[0].click().run()
    at.session_state.results = {
        "introduction": TakeResult(
            task_id="introduction",
            source="mic",
            duration_s=30.0,
            transcript="Hello",
            metrics={"duration_s": 30.0, "word_count": 1.0, "words_per_minute": 2.0},
        )
    }
    at.session_state.page = 6
    at.run()

    assert not at.exception
    assert at.header[0].value == "Session summary"
    labels = [button.proto.label for button in at.get("download_button")]
    assert labels == ["Download report (.md)", "Download metrics (.json)"]
    session_id = at.session_state.session_id
    assert (config.REPORTS_DIR / f"{session_id}.md").exists() is not cloud


def voice_result(voice: dict, contour=None) -> TakeResult:
    return TakeResult(
        task_id="read_aloud",
        source="mic",
        duration_s=2.0,
        transcript="This is.",
        metrics={"duration_s": 2.0, "pitch_range_st": 3.0},
        details={"warnings": [], "voice": voice},
        pitch_contour=contour or [],
    )


def show_read_aloud(result: TakeResult) -> AppTest:
    at = run_app()
    at.button[0].click().run()
    at.session_state.results = {"read_aloud": result}
    at.session_state.page = 1
    at.run()
    return at


def test_voice_section_and_pitch_chart_are_shown():
    voice = {
        "pitch_floor_hz": 75.0,
        "pitch_ceiling_hz": 300.0,
        "voiced_s": 1.9,
        "endings": {"falling": 1, "flat": 0, "rising": 0, "unclear": 0},
        "endings_pct": {"falling": 100.0, "flat": 0.0, "rising": 0.0},
        "statement_falling_pct": 100.0,
        "pitch_range_st": 3.0,
        "monotone": True,
        "trailing_off": [],
        "sentences": [
            {
                "start_s": 0.1,
                "end_s": 2.0,
                "text": "This is.",
                "ending": "falling",
                "change_st": -5.9,
                "question": False,
                "mean_db": 70.0,
                "last_word_db": 70.0,
                "trailing_off": False,
            }
        ],
    }
    at = show_read_aloud(voice_result(voice, [(0.1, 160.0), (1.9, 125.0)]))

    assert not at.exception
    assert len(at.get("vega_lite_chart")) == 1
    text = " ".join(m.value for m in at.markdown)
    assert "Pitch contour" in text
    assert "monotone" in text
    assert "| 0:00.1 | falling | -5.9 st | This is. |" in text


def test_voice_failure_shows_a_warning():
    at = show_read_aloud(voice_result({"error": "RuntimeError: praat crashed"}))
    assert not at.exception
    assert any("Voice analysis failed" in w.value for w in at.warning)
    assert "retake_read_aloud" not in [b.key for b in at.button]
