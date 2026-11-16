"""Speech Coach: record or upload an answer for each assessment task, then analyse it."""

import logging
import shutil
import tempfile
import time
from contextlib import closing
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import audio
import config
import recorder
import speech_model
import storage
from pipeline import TakeResult, process_take
from report import (
    build_json,
    build_markdown,
    comparison_lines,
    filler_line,
    format_metric,
    pace_line,
    voice_lines,
)
from tasks import Task, load_tasks

st.set_page_config(page_title="Speech Coach", page_icon="🎙️")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("speech_coach")

TASKS = load_tasks()
SUMMARY_PAGE = len(TASKS) + 1
CLOUD = config.cloud_mode()
CLOUD_BANNER = (
    "Cloud practice mode: lighter speech model, nothing is saved after you close the tab."
)


@st.cache_resource
def log_settings_once() -> None:
    log.info("Speech Coach startup: %s", config.startup_report())


@st.cache_resource(show_spinner="Loading speech model, about a minute on first use…")
def get_model(name: str, compute_type: str):
    return speech_model.load_model(name, compute_type)


def init_state() -> None:
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("page", 0)  # 0 = start, 1..N = tasks, N+1 = summary
    st.session_state.setdefault("started_at", {})  # task id -> countdown start time
    st.session_state.setdefault("processed", {})  # "task:source" -> file id already analysed
    st.session_state.setdefault("results", {})  # task id -> TakeResult
    st.session_state.setdefault("takes", {})  # task id -> take number, bumped by a re-take
    st.session_state.setdefault("temp_dir", None)  # cloud mode only


def go_to(page: int) -> None:
    st.session_state.page = page


def discard_temp_dir() -> None:
    if st.session_state.temp_dir:
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        st.session_state.temp_dir = None


def start_session() -> None:
    discard_temp_dir()
    if CLOUD:
        session_id = storage.new_session_id()
        st.session_state.temp_dir = tempfile.mkdtemp(prefix="speech-coach-")
    else:
        with closing(storage.connect()) as conn:
            session_id = storage.create_session(conn)
        audio.session_dir(session_id)
    st.session_state.session_id = session_id
    st.session_state.started_at = {}
    st.session_state.processed = {}
    st.session_state.results = {}
    st.session_state.takes = {}
    go_to(1)


def retake(task_id: str) -> None:
    """Clear a task's result and give its recorder and uploader fresh, empty widgets."""
    st.session_state.results.pop(task_id, None)
    st.session_state.started_at.pop(task_id, None)
    for source in ("mic", "upload"):
        st.session_state.processed.pop(f"{task_id}:{source}", None)
    st.session_state.takes[task_id] = st.session_state.takes.get(task_id, 1) + 1


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


def analyse_take(task: Task, uploaded, source: str) -> None:
    """Analyse a recording once per new file from each widget.

    uploaded is a Streamlit UploadedFile or a recorder.Recording (same interface).
    """
    widget = f"{task.id}:{source}"
    if st.session_state.processed.get(widget) == uploaded.file_id:
        return
    model_name, compute_type = config.whisper_settings()
    model = get_model(model_name, compute_type)
    try:
        with st.spinner("Analysing your take…"):
            result = process_take(
                task=task,
                session_id=st.session_state.session_id,
                data=uploaded.getvalue(),
                ext=Path(uploaded.name).suffix or ".wav",
                source=source,
                model=model,
                model_label=f"{model_name} ({compute_type})",
                cloud=CLOUD,
                base_dir=Path(st.session_state.temp_dir) if CLOUD else None,
                recorder=recorder_info(uploaded),
            )
    except audio.ConversionError as err:
        st.error(f"That file could not be read as audio. {err}")
        return
    except Exception:
        log.exception("Analysis failed for task %s", task.id)
        st.error("Something went wrong while analysing this take. Please try recording again.")
        return
    st.session_state.processed[widget] = uploaded.file_id
    st.session_state.results[task.id] = result


def recorder_info(uploaded) -> dict | None:
    """What the browser recorder reported (track settings, length); None for uploads."""
    if not isinstance(uploaded, recorder.Recording):
        return None
    return {"browser_seconds": uploaded.seconds, **uploaded.settings}


# Fixed categorical order from the dataviz reference palette: slot 1 for the contour,
# slots 2-4 for the ending labels. Shape and legend carry the label too, never colour alone.
CONTOUR_COLOR = "#2a78d6"
ENDING_STYLE = {
    "falling": ("#eb6834", "triangle-down"),
    "rising": ("#1baf7a", "triangle-up"),
    "flat": ("#eda100", "square"),
}


def pitch_chart(contour: list[tuple[float, float]], sentences: list[dict]) -> alt.LayerChart:
    """Pitch over time, with each sentence ending marked by its label."""
    points = pd.DataFrame(contour, columns=["time_s", "pitch_hz"])
    dots = (
        alt.Chart(points)
        .mark_circle(size=14, opacity=0.7, color=CONTOUR_COLOR)
        .encode(
            x=alt.X("time_s:Q", title="Time (s)"),
            y=alt.Y("pitch_hz:Q", title="Pitch (Hz)", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("time_s:Q", title="Time (s)", format=".1f"),
                alt.Tooltip("pitch_hz:Q", title="Pitch (Hz)", format=".0f"),
            ],
        )
    )
    ends = []
    for sentence in sentences:
        if sentence["ending"] not in ENDING_STYLE:
            continue
        near = [p for p in contour if p[0] <= sentence["end_s"] + 0.2]
        if near:
            ends.append(
                {
                    "time_s": near[-1][0],
                    "pitch_hz": near[-1][1],
                    "ending": sentence["ending"],
                    "change": sentence["change_st"],
                    "sentence": sentence["text"][:80],
                }
            )
    if not ends:
        return dots
    labels = list(ENDING_STYLE)
    markers = (
        alt.Chart(pd.DataFrame(ends))
        .mark_point(size=120, filled=True, stroke="white", strokeWidth=2)
        .encode(
            x="time_s:Q",
            y="pitch_hz:Q",
            color=alt.Color(
                "ending:N",
                title="Sentence ending",
                scale=alt.Scale(domain=labels, range=[ENDING_STYLE[k][0] for k in labels]),
            ),
            shape=alt.Shape(
                "ending:N",
                scale=alt.Scale(domain=labels, range=[ENDING_STYLE[k][1] for k in labels]),
            ),
            tooltip=[
                alt.Tooltip("ending:N", title="Ending"),
                alt.Tooltip("change:Q", title="Change (st)", format="+.1f"),
                alt.Tooltip("sentence:N", title="Sentence"),
            ],
        )
    )
    return alt.layer(dots, markers)


def show_voice(result: TakeResult) -> None:
    voice = result.details.get("voice")
    if voice and voice.get("error"):
        st.warning(f"Voice analysis failed, so there is no pitch or volume data: {voice['error']}")
        return
    if result.pitch_contour:
        st.markdown("**Pitch contour**")
        st.altair_chart(pitch_chart(result.pitch_contour, voice["sentences"]))
    st.markdown("\n".join(voice_lines(voice)))


def show_result(task: Task, result: TakeResult) -> None:
    st.success(f"Analysed: {result.duration_s:.1f} s · speech model {result.model or 'n/a'}")
    for warning in result.warnings:
        st.warning(warning, icon="⚠️")
    if result.warnings:
        st.button("Re-take this task", key=f"retake_{task.id}", on_click=retake, args=(task.id,))

    items = list(result.metrics.items())
    for row in range(0, len(items), 4):
        for column, (name, value) in zip(st.columns(4), items[row : row + 4], strict=False):
            column.metric(*format_metric(name, value))

    details = result.details
    if details.get("pace_windows"):
        st.markdown(f"**Pace by 30 s window:** {pace_line(details['pace_windows'])}")
    if details.get("fillers"):
        st.markdown(f"**Fillers heard:** {filler_line(details['fillers'])}")
    if "comparison" in details:
        st.markdown("\n".join(comparison_lines(details["comparison"])))
    show_voice(result)

    st.markdown("**Transcript**")
    st.write(result.transcript or "_No speech detected._")
    if result.wav_path:
        st.audio(result.wav_path)


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

    take = st.session_state.takes.get(task.id, 1)
    key = f"{st.session_state.session_id}_{task.id}_{take}"
    record_tab, upload_tab = st.tabs(["🎙️ Record", "📁 Upload"])
    with record_tab:
        recording = recorder.raw_recorder(key=f"mic_{key}")
        if recording is not None:
            analyse_take(task, recording, "mic")
    with upload_tab:
        upload = st.file_uploader(
            "Upload an audio or video file", type=audio.UPLOAD_TYPES, key=f"upload_{key}"
        )
        if upload is not None:
            analyse_take(task, upload, "upload")

    if task.id in st.session_state.results:
        show_result(task, st.session_state.results[task.id])

    back, _, forward = st.columns([1, 2, 1])
    back.button("← Back", on_click=go_to, args=(index - 1,), disabled=index == 1)
    forward.button(
        "Next →" if index < len(TASKS) else "Finish →", on_click=go_to, args=(index + 1,)
    )


def summary_page() -> None:
    session_id = st.session_state.session_id
    results = st.session_state.results
    st.header("Session summary")
    st.caption(f"Session {session_id}")
    for task in TASKS:
        result = results.get(task.id)
        if result:
            wpm = result.metrics.get("words_per_minute", 0)
            st.write(
                f"✅ **{task.title}**: {result.duration_s:.1f} s, {wpm:.0f} wpm ({result.source})"
            )
        else:
            st.write(f"⬜ **{task.title}**: not recorded")

    report_md = build_markdown(session_id, TASKS, results)
    metrics_json = build_json(session_id, TASKS, results)
    if not CLOUD:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.REPORTS_DIR / f"{session_id}.md").write_text(report_md, encoding="utf-8")
        (config.REPORTS_DIR / f"{session_id}.json").write_text(metrics_json, encoding="utf-8")

    left, right = st.columns(2)
    left.download_button(
        "Download report (.md)", report_md, f"speech-coach-{session_id}.md", "text/markdown"
    )
    right.download_button(
        "Download metrics (.json)",
        metrics_json,
        f"speech-coach-{session_id}.json",
        "application/json",
    )
    if CLOUD:
        st.caption("Download your report now: nothing is kept after you close this tab.")

    back, _, finish = st.columns([1, 2, 1])
    back.button("← Back", on_click=go_to, args=(len(TASKS),))
    if finish.button("Complete session", type="primary"):
        if CLOUD:
            discard_temp_dir()
        else:
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


log_settings_once()
init_state()
if CLOUD:
    st.info(CLOUD_BANNER, icon="☁️")
page = st.session_state.page
if page == 0 or st.session_state.session_id is None:
    start_page()
elif page == SUMMARY_PAGE:
    summary_page()
else:
    task_page(page, TASKS[page - 1])
