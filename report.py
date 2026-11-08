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
}

DEFINITIONS = (
    "_Pauses are silences between stretches of speech, measured on the audio; those over "
    "1.5 s are also counted over 0.5 s. A mid-phrase hesitation is a pause over 0.5 s that "
    "doesn't follow punctuation (not reported for the word list). The longest unbroken "
    "stretch has no pause over 0.5 s. Fillers counted: um, uh, er, ah, hmm and similar, plus "
    '"you know" and "I mean"; they depend on the speech model writing them down._'
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
    if "comparison" in details:
        lines += comparison_lines(details["comparison"])
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
