"""Build the downloadable session report (Markdown) and metrics file (JSON)."""

import json
from collections.abc import Mapping
from typing import Any

from pipeline import TakeResult
from tasks import Task

METRIC_LABELS = {
    "duration_s": ("Duration", "{:.1f} s"),
    "word_count": ("Words", "{:.0f}"),
    "words_per_minute": ("Pace", "{:.0f} wpm"),
    "filler_count": ("Fillers", "{:.0f}"),
    "fillers_per_minute": ("Fillers per minute", "{:.1f}"),
    "pauses_over_0_5s": ("Pauses > 0.5 s", "{:.0f}"),
    "pauses_over_1_5s": ("Pauses > 1.5 s", "{:.0f}"),
    "mid_phrase_hesitations": ("Mid-phrase hesitations", "{:.0f}"),
    "longest_stretch_s": ("Longest unbroken stretch", "{:.1f} s"),
    "reference_coverage": ("Reference words spoken", "{:.0%}"),
    "pitch_range_st": ("Pitch range", "{:.1f} st"),
    "falling_endings_pct": ("Falling statement endings", "{:.0f}%"),
    "trailing_off_count": ("Trailing-off sentences", "{:.0f}"),
}

DEFINITIONS = (
    "_Pauses are silences between stretches of speech, measured on the audio; those over "
    "1.5 s are also counted over 0.5 s. A mid-phrase hesitation is a pause over 0.5 s that "
    "doesn't follow punctuation (not reported for the word list). The longest unbroken "
    "stretch has no pause over 0.5 s. Fillers counted: um, uh, er, ah, hmm and similar, plus "
    '"you know" and "I mean"; they depend on the speech model writing them down._'
)
VOICE_DEFINITIONS = (
    "_Voice is measured on the audio with Praat. A sentence ending is the pitch movement over "
    "the last ~400 ms of the sentence: falling below −1 semitone, rising above +1, flat in "
    "between, unclear when there is too little voiced pitch. Falling statement endings leave "
    "out questions. Pitch range runs from the 10th to the 90th percentile of your pitch; under "
    "5 semitones is flagged as monotone. A sentence trails off when its last word is more than "
    "6 dB quieter than the sentence average._"
)
MAX_PHRASE_WORDS = 8  # longer missed/different passages are shortened in the Markdown


def shorten(phrase: str) -> str:
    words = phrase.split()
    if len(words) <= MAX_PHRASE_WORDS:
        return phrase
    return f"{' '.join(words[:MAX_PHRASE_WORDS])} … ({len(words)} words)"


def format_metric(name: str, value: float) -> tuple[str, str]:
    label, fmt = METRIC_LABELS.get(name, (name, "{}"))
    return label, fmt.format(value)


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}:{secs:04.1f}"


def models_used(results: Mapping[str, TakeResult]) -> list[str]:
    return sorted({r.model for r in results.values() if r.model})


def pace_line(windows: list[dict[str, float]]) -> str:
    return " · ".join(
        f"{format_time(w['start_s'])}–{format_time(w['end_s'])}: {w['wpm']:.0f} wpm"
        for w in windows
    )


def filler_line(fillers: list[dict[str, Any]]) -> str:
    return ", ".join(f"“{f['word']}” at {format_time(f['time_s'])}" for f in fillers)


def comparison_lines(comparison: dict[str, Any]) -> list[str]:
    lines = ["", "**Pronunciation clues** (reference vs transcript)", ""]
    if not comparison["different"] and not comparison["missed"]:
        return [*lines, "All reference words were recognised as written."]
    if comparison["different"]:
        lines += ["| Time | Reference | Transcribed as |", "|---|---|---|"]
        for diff in comparison["different"]:
            lines.append(
                f"| {format_time(diff['time_s'])} | {shorten(diff['expected'])} "
                f"| {shorten(diff['heard'])} |"
            )
    if comparison["missed"]:
        missed = ", ".join(f"“{shorten(m['expected'])}”" for m in comparison["missed"])
        lines += ["", f"Missed (not heard at all): {missed}"]
    return lines


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0f}%"


def recording_line(recording: dict[str, Any]) -> str:
    parts = []
    if recording.get("sample_rate_hz"):
        parts.append(f"{recording['sample_rate_hz']} Hz")
    if recording.get("channels"):
        parts.append(str(recording["channels"]))
    if recording.get("duration_s") is not None:
        parts.append(f"{recording['duration_s']:.1f} s")
    processing = [
        name
        for key, name in (
            ("autoGainControl", "auto gain"),
            ("noiseSuppression", "noise suppression"),
            ("echoCancellation", "echo cancellation"),
        )
        if recording.get(key)
    ]
    if any(key in recording for key in ("autoGainControl", "noiseSuppression")):
        parts.append(
            f"browser processing on: {', '.join(processing)}"
            if processing
            else "browser processing off"
        )
    return ", ".join(parts)


def voice_lines(voice: dict[str, Any] | None) -> list[str]:
    """Markdown for the pitch, intonation and volume section of one task."""
    lines = ["", "**Voice: pitch, intonation and volume**", ""]
    if not voice:
        return [*lines, "> ⚠️ Voice analysis was not run for this take."]
    if voice.get("error"):
        return [
            *lines,
            f"> ⚠️ Voice analysis failed, so there is no pitch or volume data: {voice['error']}",
        ]
    if not voice["sentences"]:
        lines.append(
            "- _No sentences in the transcript, so endings and trailing off were not measured._"
        )
    counts, pcts = voice["endings"], voice["endings_pct"]
    endings = " · ".join(
        f"{name} {counts[name]} ({_pct(pcts[name])})" for name in ("falling", "flat", "rising")
    )
    if counts.get("unclear"):
        endings += f" · unclear {counts['unclear']}"
    lines.append(f"- **Sentence endings:** {endings}")
    lines.append(f"- **Falling statement endings:** {_pct(voice['statement_falling_pct'])}")
    if voice["pitch_range_st"] is None:
        lines.append(
            f"- **Pitch range:** not measured (only {voice['voiced_s']:.1f} s of voiced pitch)"
        )
    else:
        flag = " ⚠️ monotone (under 5 st)" if voice["monotone"] else ""
        lines.append(f"- **Pitch range:** {voice['pitch_range_st']:.1f} semitones{flag}")
    if voice["trailing_off"]:
        trailing = "; ".join(
            f"{format_time(t['time_s'])} “{shorten(t['text'])}” (−{t['drop_db']:.0f} dB)"
            for t in voice["trailing_off"]
        )
        lines.append(f"- **Trailing off:** {trailing}")
    else:
        lines.append("- **Trailing off:** none")
    lines.append(
        f"- _Pitch tracked between {voice['pitch_floor_hz']:.0f} and "
        f"{voice['pitch_ceiling_hz']:.0f} Hz_"
    )
    if voice["sentences"]:
        lines += ["", "| Time | Ending | Change | Sentence |", "|---|---|---|---|"]
        for sentence in voice["sentences"]:
            change = sentence["change_st"]
            shown = "–" if change is None else f"{change:+.1f} st"
            tail = " · trails off" if sentence["trailing_off"] else ""
            lines.append(
                f"| {format_time(sentence['start_s'])} | {sentence['ending']}{tail} | {shown} "
                f"| {shorten(sentence['text'])} |"
            )
    return lines


def task_section(task: Task, result: TakeResult | None) -> list[str]:
    lines = [f"## {task.title}", ""]
    if result is None:
        return [*lines, "_Not recorded._", ""]
    if result.model:
        lines += [f"_Speech model: {result.model}_", ""]
    for warning in result.warnings:
        lines += [f"> ⚠️ {warning}", ""]
    for name, value in result.metrics.items():
        label, shown = format_metric(name, value)
        lines.append(f"- **{label}:** {shown}")
    details = result.details
    if details.get("pace_windows"):
        lines.append(f"- **Pace by 30 s window:** {pace_line(details['pace_windows'])}")
    if details.get("fillers"):
        lines.append(f"- **Fillers heard:** {filler_line(details['fillers'])}")
    if details.get("recording"):
        lines.append(f"- **Recording:** {recording_line(details['recording'])}")
    if "comparison" in details:
        lines += comparison_lines(details["comparison"])
    lines += voice_lines(details.get("voice"))
    lines += ["", "**Transcript**", "", f"> {result.transcript or '(no speech detected)'}", ""]
    return lines


def build_markdown(session_id: str, tasks: list[Task], results: Mapping[str, TakeResult]) -> str:
    models = models_used(results)
    lines = [
        "# Speech Coach report",
        "",
        f"Session: `{session_id}`  ",
        f"Speech model: {', '.join(models) if models else 'n/a'}",
        "",
        DEFINITIONS,
        "",
        VOICE_DEFINITIONS,
        "",
    ]
    for task in tasks:
        lines += task_section(task, results.get(task.id))
    return "\n".join(lines)


def build_json(session_id: str, tasks: list[Task], results: Mapping[str, TakeResult]) -> str:
    entries = []
    for task in tasks:
        entry: dict[str, Any] = {"task_id": task.id, "title": task.title}
        result = results.get(task.id)
        if result is None:
            entry["recorded"] = False
        else:
            entry.update(
                source=result.source,
                model=result.model,
                transcript=result.transcript,
                metrics=result.metrics,
                **result.details,
            )
        entries.append(entry)
    payload = {"session_id": session_id, "speech_models": models_used(results), "tasks": entries}
    return json.dumps(payload, indent=2)
