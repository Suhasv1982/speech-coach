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
