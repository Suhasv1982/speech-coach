"""Speaking metrics computed from a transcript with word timestamps.

Definitions (used in the report):
- Fillers: um, uh, er, ah, hmm and variants, plus "you know" and "I mean". "Like" and "so"
  are not counted because they are usually real words.
- Pause: a gap between the end of one word and the start of the next. Pauses over 1.5 s
  are also counted in the over-0.5 s total.
- Pauses are measured on the audio with voice activity detection (Whisper's word timestamps
  stretch across short silences); word timestamps are the fallback.
- Mid-phrase hesitation: a pause over 0.5 s where the previous word does not end with
  punctuation (or ends with "..."), i.e. a stop inside a phrase rather than between
  sentences or clauses. Not reported for word lists, where pausing between words is expected.
- Longest unbroken stretch: the longest run of speech with no pause over 0.5 s.
- Pace windows: words per minute in each 30-second window of the recording.
- Reference comparison (scripted tasks): the transcript aligned word by word with the
  reference text; missed and differently transcribed words are pronunciation clues.
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from speech_model import Transcript, Word
from tasks import Task

PAUSE_S = 0.5
LONG_PAUSE_S = 1.5
WINDOW_S = 30.0
MIN_WINDOW_S = 5.0  # a shorter final window is merged into the previous one

FILLER_WORDS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "ahh", "hmm", "hm", "mm", "mhm"}
FILLER_PHRASES = {("you", "know"), ("i", "mean")}

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_PHRASE_END = re.compile(r"[.,!?;:—–-]$")
_TRAILING_OFF = re.compile(r"(\.\.\.|…)$")  # Whisper writes "to..." for a trailing-off pause


def ends_phrase(text: str) -> bool:
    """True if the word ends a sentence or clause; an ellipsis marks a hesitation instead."""
    return bool(_PHRASE_END.search(text)) and not _TRAILING_OFF.search(text)


def normalise(text: str) -> list[str]:
    """Lowercase words with punctuation removed."""
    return _WORD.findall(text.lower())


@dataclass
class Token:
    """One normalised word with the time it was spoken (if known)."""

    word: str
    start: float | None = None


@dataclass
class Analysis:
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


def _tokens(transcript: Transcript) -> list[Token]:
    if transcript.words:
        return [Token(w, word.start) for word in transcript.words for w in normalise(word.text)]
    return [Token(w) for w in normalise(transcript.text)]


def find_fillers(tokens: list[Token]) -> tuple[list[dict[str, Any]], set[int]]:
    """Return the fillers found ({word, time_s}) and the token indexes they occupy."""
    fillers: list[dict[str, Any]] = []
    used: set[int] = set()
    i = 0
    while i < len(tokens):
        pair = (tokens[i].word, tokens[i + 1].word) if i + 1 < len(tokens) else None
        if pair in FILLER_PHRASES:
            fillers.append({"word": " ".join(pair), "time_s": _round(tokens[i].start)})
            used.update({i, i + 1})
            i += 2
            continue
        if tokens[i].word in FILLER_WORDS:
            fillers.append({"word": tokens[i].word, "time_s": _round(tokens[i].start)})
            used.add(i)
        i += 1
    return fillers, used


def _word_before(words: list[Word], t: float) -> int | None:
    """Index of the last word that starts before time t."""
    index = None
    for i, word in enumerate(words):
        if word.start >= t:
            break
        index = i
    return index


def pause_metrics(
    words: list[Word], speech_segments: list[tuple[float, float]] | None = None
) -> dict[str, float]:
    """Pauses, hesitations and the longest unbroken stretch.

    Pauses come from the voiced segments found in the audio when available (Whisper's word
    timings stretch over short silences), otherwise from the gaps between word timestamps.
    """
    spans = speech_segments or [(w.start, w.end) for w in words]
    silences = [
        (a_end, b_start) for (_, a_end), (b_start, _) in zip(spans, spans[1:], strict=False)
    ]
    pauses = [(start, end) for start, end in silences if end - start > PAUSE_S]

    hesitations = 0
    for start, _ in pauses:
        i = _word_before(words, start)
        # mid-phrase: speech continues after the pause and the word before it doesn't end a phrase
        if i is not None and i < len(words) - 1 and not ends_phrase(words[i].text):
            hesitations += 1

    longest = 0.0
    run_start = spans[0][0] if spans else 0.0
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:], strict=False):
        if next_start - prev_end > PAUSE_S:
            longest = max(longest, prev_end - run_start)
            run_start = next_start
    if spans:
        longest = max(longest, spans[-1][1] - run_start)

    return {
        "pauses_over_0_5s": float(len(pauses)),
        "pauses_over_1_5s": float(sum(end - start > LONG_PAUSE_S for start, end in pauses)),
        "mid_phrase_hesitations": float(hesitations),
        "longest_stretch_s": round(longest, 1),
    }


def pace_windows(words: list[Word], duration_s: float) -> list[dict[str, float]]:
    """Words per minute in consecutive 30 s windows; a short final window is merged."""
    if duration_s <= 0:
        return []
    bounds: list[list[float]] = []
    start = 0.0
    while start < duration_s:
        end = min(start + WINDOW_S, duration_s)
        if bounds and end - start < MIN_WINDOW_S:
            bounds[-1][1] = end
        else:
            bounds.append([start, end])
        start = end
    windows = []
    for i, (lo, hi) in enumerate(bounds):
        last = i == len(bounds) - 1
        count = sum(1 for w in words if lo <= w.start < hi or (last and w.start >= hi))
        windows.append(
            {
                "start_s": round(lo, 1),
                "end_s": round(hi, 1),
                "wpm": round(count / (hi - lo) * 60, 1),
            }
        )
    return windows


def compare_to_reference(reference: str, spoken: list[Token]) -> dict[str, Any]:
    """Align spoken words with the reference text and list the differences."""
    ref = normalise(reference)
    said = [t.word for t in spoken]
    matcher = SequenceMatcher(a=ref, b=said, autojunk=False)
    missed: list[dict[str, Any]] = []
    different: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "delete":
            missed.append({"expected": " ".join(ref[i1:i2]), "position": i1 + 1})
        elif op == "replace":
            different.append(
                {
                    "expected": " ".join(ref[i1:i2]),
                    "heard": " ".join(said[j1:j2]),
                    "time_s": _round(spoken[j1].start),
                }
            )
        elif op == "insert":
            extra.append({"heard": " ".join(said[j1:j2]), "time_s": _round(spoken[j1].start)})
    matched = sum(block.size for block in matcher.get_matching_blocks())
    last_matched = max(
        (block.a + block.size for block in matcher.get_matching_blocks() if block.size), default=0
    )
    return {
        "reference_words": len(ref),
        "matched_words": matched,
        "last_matched_position": last_matched,
        "missed": missed,
        "different": different,
        "extra": extra,
    }


def coverage_warning(task: Task, comparison: dict[str, Any]) -> str | None:
    """Warn when too few reference words were recognised (e.g. the recording stopped early)."""
    total = comparison["reference_words"]
    if task.min_coverage is None or not total:
        return None
    matched = comparison["matched_words"]
    if matched / total >= task.min_coverage:
        return None
    message = (
        f"Only {matched} of {total} words were recognised ({matched / total:.0%}, "
        f"below {task.min_coverage:.0%})."
    )
    after_last = total - comparison["last_matched_position"]
    if after_last >= total / 4:
        message += (
            f" The last {after_last} words of the list were not heard, so the recording may "
            "have ended before you finished."
        )
    return message + " Consider a re-take."


def is_word_list(task: Task) -> bool:
    """A scripted task whose reference text has one word per line."""
    lines = (task.reference_text or "").splitlines()
    return bool(lines) and all(len(line.split()) == 1 for line in lines)


def compute_metrics(task: Task, transcript: Transcript, duration_s: float) -> Analysis:
    tokens = _tokens(transcript)
    fillers, filler_idx = find_fillers(tokens)
    minutes = duration_s / 60 if duration_s > 0 else 0.0

    metrics: dict[str, float] = {
        "duration_s": round(duration_s, 1),
        "word_count": float(len(tokens)),
        "words_per_minute": round(len(tokens) / minutes, 1) if minutes else 0.0,
        "filler_count": float(len(fillers)),
        "fillers_per_minute": round(len(fillers) / minutes, 1) if minutes else 0.0,
        **pause_metrics(transcript.words, transcript.speech_segments),
    }
    if is_word_list(task):
        # Pausing between list words is expected, so "mid-phrase" doesn't apply.
        del metrics["mid_phrase_hesitations"]
    details: dict[str, Any] = {
        "fillers": fillers,
        "pace_windows": pace_windows(transcript.words, duration_s),
        "warnings": [],
    }
    if task.reference_text:
        spoken = [t for i, t in enumerate(tokens) if i not in filler_idx]
        comparison = compare_to_reference(task.reference_text, spoken)
        total = comparison["reference_words"]
        metrics["reference_coverage"] = round(comparison["matched_words"] / total, 3)
        details["comparison"] = comparison
        if warning := coverage_warning(task, comparison):
            details["warnings"].append(warning)
    return Analysis(metrics=metrics, details=details)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


METRIC_UNITS = {
    "duration_s": "s",
    "word_count": "words",
    "words_per_minute": "wpm",
    "filler_count": "fillers",
    "fillers_per_minute": "per min",
    "pauses_over_0_5s": "pauses",
    "pauses_over_1_5s": "pauses",
    "mid_phrase_hesitations": "pauses",
    "longest_stretch_s": "s",
    "reference_coverage": "fraction",
}
