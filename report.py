"""Build the downloadable session report (Markdown) and metrics file (JSON)."""

import json
from collections.abc import Mapping

from pipeline import TakeResult
from tasks import Task

METRIC_LABELS = {
    "duration_s": ("Duration", "{:.1f} s"),
    "word_count": ("Words", "{:.0f}"),
    "words_per_minute": ("Pace", "{:.0f} wpm"),
    "reference_coverage": ("Reference words spoken", "{:.0%}"),
}


def format_metric(name: str, value: float) -> tuple[str, str]:
    label, fmt = METRIC_LABELS.get(name, (name, "{}"))
    return label, fmt.format(value)


def build_markdown(session_id: str, tasks: list[Task], results: Mapping[str, TakeResult]) -> str:
    lines = ["# Speech Coach report", "", f"Session: `{session_id}`", ""]
    for task in tasks:
        lines.append(f"## {task.title}")
        result = results.get(task.id)
        if result is None:
            lines += ["", "_Not recorded._", ""]
            continue
        lines.append("")
        for name, value in result.metrics.items():
            label, shown = format_metric(name, value)
            lines.append(f"- **{label}:** {shown}")
        lines += ["", "**Transcript**", "", f"> {result.transcript or '(no speech detected)'}", ""]
    return "\n".join(lines)


def build_json(session_id: str, tasks: list[Task], results: Mapping[str, TakeResult]) -> str:
    payload = {
        "session_id": session_id,
        "tasks": [
            {
                "task_id": task.id,
                "title": task.title,
                **(
                    {
                        "source": results[task.id].source,
                        "transcript": results[task.id].transcript,
                        "metrics": results[task.id].metrics,
                    }
                    if task.id in results
                    else {"recorded": False}
                ),
            }
            for task in tasks
        ],
    }
    return json.dumps(payload, indent=2)
