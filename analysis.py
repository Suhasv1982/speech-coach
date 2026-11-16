"""Speaking metrics computed from a transcript with word timestamps.

Definitions (used in the report):
- Fillers: um, uh, er, ah, hmm and variants, "you know" and "I mean", plus three that are
  often real words and so are only counted in filler positions:
  - "like" followed by a comma, after a form of "be" ("it was like rolled"), or before a
    number or after "about"/"just" ("like two hours", "about like 30%"); never "would
    like", "feel like" or "-like" ("government-like"), or "like" meaning "such as";
  - "kind of" / "sort of" unless it follows a determiner ("a kind of", "what kind of");
  - "So," opening a sentence (start of the answer or after . ? !, followed by a comma).
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
  Numbers are compared by value, so "forty percent" matches "40%" and "eighteen" "18".
- Word clarity (word list): for each listed word, clear (recognised with confidence of at
  least 60%), unclear (recognised, lower confidence), misheard or missed.
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
_SENTENCE_END = re.compile(r"[.?!][\"')]*$")

BE_FORMS = {"is", "was", "were", "are", "am", "be", "been", "being", "it's", "that's"}
BE_FORMS |= {"he's", "she's", "they're", "we're", "you're", "i'm", "there's"}
NOT_FILLER_BEFORE_LIKE = {"would", "i'd", "you'd", "we'd", "they'd", "he'd", "she'd"}
NOT_FILLER_BEFORE_LIKE |= {"feel", "feels", "felt", "look", "looks", "looked", "seem", "seems"}
NOT_FILLER_BEFORE_LIKE |= {"seemed", "sound", "sounds", "sounded"}
APPROX_WORDS = {"about", "just", "around", "roughly"}
KIND_DETERMINERS = {"a", "an", "the", "what", "this", "that", "these", "those", "some", "any"}
KIND_DETERMINERS |= {"every", "each", "one", "which", "same", "different", "no", "other"}
CLEAR_CONFIDENCE = 0.6

UNITS = {
    name: value
    for value, name in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
TENS = {
    name: 10 * (i + 2)
    for i, name in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())
}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}


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
    raw: str = ""  # the word as transcribed, with punctuation
    last: bool = True  # last token of its transcribed word (so trailing punctuation applies)
    joined: bool = False  # part of a hyphenated word, e.g. "government-like"
    probability: float | None = None
    display: str = ""  # how to show it (canonical number tokens keep their spoken form)

    @property
    def shown(self) -> str:
        return self.display or self.word


@dataclass
class Analysis:
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


def _tokens(transcript: Transcript) -> list[Token]:
    if not transcript.words:
        return [Token(w, raw=w) for w in normalise(transcript.text)]
    tokens = []
    for word in transcript.words:
        parts = normalise(word.text)
        for k, part in enumerate(parts):
            tokens.append(
                Token(
                    part,
                    word.start,
                    raw=word.text,
                    last=k == len(parts) - 1,
                    joined=len(parts) > 1 and "-" in word.text,
                    probability=word.probability,
                )
            )
    return tokens


def _is_number(token: Token) -> bool:
    return token.word.isdigit() or token.word in UNITS or token.word in TENS


def _ends_with_comma(token: Token) -> bool:
    return token.last and token.raw.rstrip().endswith(",")


def _filler_like(tokens: list[Token], i: int) -> bool:
    token = tokens[i]
    if token.word != "like" or token.joined:
        return False
    prev = tokens[i - 1].word if i > 0 else ""
    if prev in NOT_FILLER_BEFORE_LIKE:
        return False
    if _ends_with_comma(token) or prev in BE_FORMS or prev in APPROX_WORDS:
        return True
    return i + 1 < len(tokens) and _is_number(tokens[i + 1])


def _filler_kind_of(tokens: list[Token], i: int) -> bool:
    if i + 1 >= len(tokens) or tokens[i + 1].word != "of":
        return False
    if tokens[i].word not in {"kind", "sort"}:
        return False
    return i == 0 or tokens[i - 1].word not in KIND_DETERMINERS


def _filler_so(tokens: list[Token], i: int) -> bool:
    if tokens[i].word != "so" or not _ends_with_comma(tokens[i]):
        return False
    return i == 0 or bool(_SENTENCE_END.search(tokens[i - 1].raw.rstrip()))


def find_fillers(tokens: list[Token]) -> tuple[list[dict[str, Any]], set[int]]:
    """Return the fillers found ({word, time_s}) and the token indexes they occupy."""
    fillers: list[dict[str, Any]] = []
    used: set[int] = set()
    i = 0
    while i < len(tokens):
        pair = (tokens[i].word, tokens[i + 1].word) if i + 1 < len(tokens) else None
        if pair in FILLER_PHRASES or _filler_kind_of(tokens, i):
            fillers.append({"word": " ".join(pair), "time_s": _round(tokens[i].start)})
            used.update({i, i + 1})
            i += 2
            continue
        if tokens[i].word in FILLER_WORDS or _filler_like(tokens, i) or _filler_so(tokens, i):
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


def _number_value(tokens: list[Token], i: int) -> tuple[int, int] | None:
    """Parse spelled-out number words from tokens[i]; return (value, next index) or None."""
    total = current = 0
    j = i
    last = None  # kind of the previous number word, so "one two" isn't read as 3
    while j < len(tokens):
        word = tokens[j].word
        if word in UNITS and last not in {"unit", "teen"}:
            current += UNITS[word]
            last = "teen" if UNITS[word] >= 10 else "unit"
        elif word in TENS and last in {None, "scale"}:
            current += TENS[word]
            last = "tens"
        elif word in SCALES and last is not None and last != "scale":
            if SCALES[word] == 100:
                current *= 100
            else:
                total += current * SCALES[word]
                current = 0
            last = "scale"
        elif word == "and" and last == "scale" and j + 1 < len(tokens):
            following = tokens[j + 1].word
            if following not in UNITS and following not in TENS:
                break
        else:
            break
        j += 1
    if last is None:
        return None
    return total + current, j


def canonicalise(tokens: list[Token]) -> list[Token]:
    """Numbers as digits so spoken and written forms compare equal; "40%" = "forty percent"."""
    out: list[Token] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        parsed = _number_value(tokens, i)
        if parsed is not None:
            value, j = parsed
            word, display = str(value), " ".join(t.word for t in tokens[i:j])
        elif token.word.isdigit():
            word, j = token.word, i + 1
            display = token.raw.strip(" ,.;:!?") or token.word
        else:
            out.append(token)
            i += 1
            continue
        percent = "%" in tokens[j - 1].raw
        if not percent and j < len(tokens) and tokens[j].word == "percent":
            percent, display, j = True, f"{display} percent", j + 1
        out.append(
            Token(
                word + ("%" if percent else ""),
                token.start,
                raw=token.raw,
                probability=token.probability,
                display=display,
            )
        )
        i = j
    return out


def reference_tokens(reference: str) -> list[Token]:
    return canonicalise([Token(w, raw=w) for w in normalise(reference)])


def _shown(tokens: list[Token]) -> str:
    return " ".join(t.shown for t in tokens)


def compare_to_reference(reference: str, spoken: list[Token]) -> dict[str, Any]:
    """Align spoken words with the reference text and list the differences."""
    ref_tokens = reference_tokens(reference)
    spoken = canonicalise(spoken)
    ref = [t.word for t in ref_tokens]
    said = [t.word for t in spoken]
    matcher = SequenceMatcher(a=ref, b=said, autojunk=False)
    missed: list[dict[str, Any]] = []
    different: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "delete":
            missed.append({"expected": _shown(ref_tokens[i1:i2]), "position": i1 + 1})
        elif op == "replace":
            different.append(
                {
                    "expected": _shown(ref_tokens[i1:i2]),
                    "heard": _shown(spoken[j1:j2]),
                    "time_s": _round(spoken[j1].start),
                    "position": i1 + 1,
                }
            )
        elif op == "insert":
            extra.append({"heard": _shown(spoken[j1:j2]), "time_s": _round(spoken[j1].start)})
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


def word_clarity(reference: str, spoken: list[Token]) -> list[dict[str, Any]]:
    """Per listed word: clear, unclear (low confidence), misheard or missed."""
    ref_tokens = reference_tokens(reference)
    spoken = canonicalise(spoken)
    matcher = SequenceMatcher(
        a=[t.word for t in ref_tokens], b=[t.word for t in spoken], autojunk=False
    )
    rows: list[dict[str, Any]] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "insert":
            continue
        for k in range(i1, i2):
            row: dict[str, Any] = {
                "word": ref_tokens[k].shown,
                "heard": None,
                "time_s": None,
                "confidence": None,
            }
            if op == "equal":
                token = spoken[j1 + k - i1]
                confidence = token.probability
                clear = confidence is None or confidence >= CLEAR_CONFIDENCE
                row.update(
                    heard=token.shown,
                    time_s=_round(token.start),
                    confidence=None if confidence is None else round(confidence, 2),
                    status="clear" if clear else "unclear",
                )
            elif op == "replace" and j1 + k - i1 < j2:
                # pair listed and heard words one to one; the last listed word of the block
                # takes any extra heard words ("databricks" -> "data bricks")
                end = j2 if k == i2 - 1 else j1 + k - i1 + 1
                row.update(
                    heard=_shown(spoken[j1 + k - i1 : end]),
                    time_s=_round(spoken[j1 + k - i1].start),
                    status="misheard",
                )
            else:
                row["status"] = "missed"
            rows.append(row)
    return rows


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
        if is_word_list(task):
            clarity = word_clarity(task.reference_text, spoken)
            details["word_clarity"] = clarity
            clear = sum(row["status"] == "clear" for row in clarity)
            metrics["clear_words_pct"] = round(100 * clear / len(clarity), 1) if clarity else 0.0
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
    "clear_words_pct": "%",
    "pitch_range_st": "st",
    "falling_endings_pct": "%",
    "trailing_off_count": "sentences",
}
