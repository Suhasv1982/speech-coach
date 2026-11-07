"""Speech Coach: record or upload an answer for each assessment task."""

import time
from contextlib import closing
from pathlib import Path

import streamlit as st

import audio
import storage
from tasks import Task, load_tasks

st.set_page_config(page_title="Speech Coach", page_icon="🎙️")

TASKS = load_tasks()
SUMMARY_PAGE = len(TASKS) + 1


def init_state() -> None:
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("page", 0)  # 0 = start, 1..N = tasks, N+1 = summary
    st.session_state.setdefault("started_at", {})  # task id -> countdown start time
    st.session_state.setdefault("processed", {})  # "task:source" -> file id already saved
    st.session_state.setdefault("results", {})  # task id -> (wav path, duration)


def go_to(page: int) -> None:
    st.session_state.page = page


def start_session() -> None:
    with closing(storage.connect()) as conn:
        session_id = storage.create_session(conn)
    audio.session_dir(session_id)
    st.session_state.session_id = session_id
    st.session_state.started_at = {}
    st.session_state.processed = {}
    st.session_state.results = {}
    go_to(1)


def start_countdown(task_id: str) -> None:
    st.session_state.started_at[task_id] = time.time()


@st.fragment(run_every=1)
def countdown(task: Task) -> None:
    started = st.session_state.started_at.get(task.id)
    if started is None:
        st.caption(f"Time limit: {task.time_limit_s} s. Click Start when you are ready.")
        return
    remaining = max(0, task.time_limit_s - int(time.time() - started))
    minutes, seconds = divmod(remaining, 60)
    if remaining:
        st.progress(remaining / task.time_limit_s, text=f"⏱ {minutes}:{seconds:02d} remaining")
    else:
        st.progress(0.0, text="⏱ Time is up. Stop your recording.")


def save_take(task: Task, uploaded, source: str) -> None:
    """Save, convert and store a recording once per new file from each widget."""
    widget = f"{task.id}:{source}"
    if st.session_state.processed.get(widget) == uploaded.file_id:
        return
    session_id = st.session_state.session_id
    ext = Path(uploaded.name).suffix or ".wav"
    try:
        original = audio.save_recording(session_id, task.id, uploaded.getvalue(), ext)
        wav_path = audio.convert_to_wav(original)
    except audio.ConversionError as err:
        st.error(str(err))
        return
    duration = audio.wav_duration_s(wav_path)
    with closing(storage.connect()) as conn:
        storage.record_task(conn, session_id, task.id, source, original, wav_path, duration)
    st.session_state.processed[widget] = uploaded.file_id
    st.session_state.results[task.id] = (str(wav_path), duration)


def task_page(index: int, task: Task) -> None:
    st.progress(index / len(TASKS), text=f"Task {index} of {len(TASKS)}")
    st.header(task.title)
    st.write(task.instructions)

    started = task.id in st.session_state.started_at
    if task.reference_text:
        st.text_area("Read this", task.reference_text, height=260, disabled=True)

    st.button(
        "Restart timer" if started else "Start",
        on_click=start_countdown,
        args=(task.id,),
        type="secondary" if started else "primary",
    )
    if task.prompt and (started or not task.reveal_on_start):
        st.info(task.prompt)
    countdown(task)

    key = f"{st.session_state.session_id}_{task.id}"
    record_tab, upload_tab = st.tabs(["🎙️ Record", "📁 Upload"])
    with record_tab:
        recording = st.audio_input("Record your answer", key=f"mic_{key}")
        if recording is not None:
            save_take(task, recording, "mic")
    with upload_tab:
        upload = st.file_uploader(
            "Upload an audio or video file", type=audio.UPLOAD_TYPES, key=f"upload_{key}"
        )
        if upload is not None:
            save_take(task, upload, "upload")

    if task.id in st.session_state.results:
        wav_path, duration = st.session_state.results[task.id]
        st.success(f"Saved: {duration:.1f} s, 16 kHz mono WAV")
        st.audio(wav_path)

    back, _, forward = st.columns([1, 2, 1])
    back.button("← Back", on_click=go_to, args=(index - 1,), disabled=index == 1)
    forward.button(
        "Next →" if index < len(TASKS) else "Finish →", on_click=go_to, args=(index + 1,)
    )


def summary_page() -> None:
    session_id = st.session_state.session_id
    st.header("Session summary")
    st.caption(f"Session {session_id}")
    with closing(storage.connect()) as conn:
        rows = {row["task_id"]: row for row in storage.get_session_tasks(conn, session_id)}
    for task in TASKS:
        row = rows.get(task.id)
        if row:
            st.write(f"✅ **{task.title}**: {row['duration_s']:.1f} s ({row['source']})")
        else:
            st.write(f"⬜ **{task.title}**: not recorded")

    back, _, finish = st.columns([1, 2, 1])
    back.button("← Back", on_click=go_to, args=(len(TASKS),))
    if finish.button("Complete session", type="primary"):
        with closing(storage.connect()) as conn:
            storage.complete_session(conn, session_id)
        st.session_state.session_id = None
        go_to(0)
        st.rerun()


def start_page() -> None:
    st.title("🎙️ Speech Coach")
    st.write(
        f"This assessment has {len(TASKS)} short speaking tasks. For each one you can record "
        "in the browser or upload a file recorded on your laptop or phone."
    )
    for task in TASKS:
        st.write(f"- **{task.title}** ({task.time_limit_s} s)")
    st.button("Start new assessment", type="primary", on_click=start_session)


init_state()
page = st.session_state.page
if page == 0 or st.session_state.session_id is None:
    start_page()
elif page == SUMMARY_PAGE:
    summary_page()
else:
    task_page(page, TASKS[page - 1])
