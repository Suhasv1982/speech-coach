"""Simple speaking metrics computed from a transcript."""

import re

from speech_model import Transcript
from tasks import Task

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def normalise(text: str) -> list[str]:
    """Lowercase words with punctuation removed."""
    return _WORD.findall(text.lower())


def compute_metrics(task: Task, transcript: Transcript, duration_s: float) -> dict[str, float]:
    spoken = normalise(transcript.text)
    metrics = {
        "duration_s": round(duration_s, 1),
        "word_count": float(len(spoken)),
        "words_per_minute": round(len(spoken) / duration_s * 60, 1) if duration_s > 0 else 0.0,
    }
    if task.reference_text:
        reference = set(normalise(task.reference_text))
        said = set(spoken)
        coverage = len(reference & said) / len(reference) if reference else 0.0
        metrics["reference_coverage"] = round(coverage, 3)
    return metrics


METRIC_UNITS = {
    "duration_s": "s",
    "word_count": "words",
    "words_per_minute": "wpm",
    "reference_coverage": "fraction",
}
