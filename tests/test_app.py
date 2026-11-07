from streamlit.testing.v1 import AppTest

import config


def run_app() -> AppTest:
    return AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=30).run()


def test_start_page_lists_the_tasks():
    at = run_app()
    assert not at.exception
    assert at.title[0].value == "🎙️ Speech Coach"
    assert at.button[0].label == "Start new assessment"


def test_starting_a_session_opens_first_task_and_creates_folder():
    at = run_app()
    at.button[0].click().run()

    assert not at.exception
    assert at.header[0].value == "Read aloud"
    session_id = at.session_state.session_id
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
