"""Pitch, intonation and loudness measured on the audio with Praat (praat-parselmouth).

Definitions (used in the report):
- Sentences: transcript words grouped up to a word ending in ".", "?" or "!" (an ellipsis is
  a hesitation, not an end); words left over at the end form a final sentence.
- Sentence ending: the pitch movement over the last ~400 ms of voice in the sentence's last
  word, from a straight-line fit in semitones. Below -1 st is falling, above +1 st rising,
  in between flat; too little voiced pitch there is "unclear".
- Pitch range: the 10th to 90th percentile of all voiced pitch, in semitones. Under 5 st is
  flagged as monotone.
- Trailing off: a sentence of two or more words whose last word is more than 6 dB quieter
  than the sentence average.
"""

from pathlib import Path
from typing import Any

import numpy as np

from speech_model import Word

TIME_STEP_S = 0.01
ENDING_WINDOW_S = 0.4
ENDING_GRACE_S = 0.15  # Whisper's word ends can be early or stretched; look a little past
ENDING_THRESHOLD_ST = 1.0
MIN_ENDING_FRAMES = 5
MONOTONE_RANGE_ST = 5.0
MIN_VOICED_S = 1.0  # less voiced pitch than this and the range is not reported
TRAILING_OFF_DB = 6.0
CONTOUR_STEP_S = 0.05
SENTENCE_END = (".", "?", "!")
ENDINGS = ("falling", "flat", "rising")


def semitones(hz: np.ndarray | float, ref_hz: float = 100.0) -> np.ndarray | float:
    return 12 * np.log2(np.asarray(hz) / ref_hz)


def split_sentences(words: list[Word]) -> list[list[Word]]:
    sentences: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        current.append(word)
        text = word.text.rstrip("\"')")
        if text.endswith(SENTENCE_END) and not text.endswith(("...", "…")):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def classify_ending(change_st: float | None) -> str:
    if change_st is None:
        return "unclear"
    if change_st < -ENDING_THRESHOLD_ST:
        return "falling"
    if change_st > ENDING_THRESHOLD_ST:
        return "rising"
    return "flat"


def ending_change(times: np.ndarray, f0: np.ndarray, last: Word) -> float | None:
    """Semitone change over the last ~400 ms of voice in the sentence's last word."""
    voiced = f0 > 0
    in_word = voiced & (times >= last.start) & (times <= last.end + ENDING_GRACE_S)
    if not in_word.any():
        return None
    anchor = times[in_word][-1]
    window = voiced & (times >= anchor - ENDING_WINDOW_S) & (times <= anchor)
    if window.sum() < MIN_ENDING_FRAMES:
        return None
    t, st = times[window], semitones(f0[window])
    slope = np.polyfit(t, st, 1)[0]
    return float(slope * (t[-1] - t[0]))


def mean_db(times: np.ndarray, db: np.ndarray, start: float, end: float) -> float | None:
    """Energy-averaged intensity between start and end, in dB."""
    mask = (times >= start) & (times <= end)
    if not mask.any():
        return None
    return float(10 * np.log10(np.mean(10 ** (db[mask] / 10))))


def _pct(part: int, whole: int) -> float | None:
    return round(100 * part / whole, 1) if whole else None


def analyse_voice(
    wav_path: Path, words: list[Word], floor_hz: float = 75.0, ceiling_hz: float = 300.0
) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    """Return (voice details for the report, pitch contour as (time_s, Hz) for the chart)."""
    import parselmouth  # imported here so app startup and most tests stay fast

    sound = parselmouth.Sound(str(wav_path))
    if sound.n_channels > 1:
        sound = sound.convert_to_mono()
    pitch = sound.to_pitch(time_step=TIME_STEP_S, pitch_floor=floor_hz, pitch_ceiling=ceiling_hz)
    times = np.asarray(pitch.xs())
    f0 = np.nan_to_num(np.asarray(pitch.selected_array["frequency"]), nan=0.0)
    intensity = sound.to_intensity(minimum_pitch=floor_hz)
    i_times = np.asarray(intensity.xs())
    i_db = np.asarray(intensity.values[0])

    voiced_hz = f0[f0 > 0]
    pitch_range = None
    if len(voiced_hz) * TIME_STEP_S >= MIN_VOICED_S:
        p10, p90 = np.percentile(voiced_hz, [10, 90])
        pitch_range = round(float(12 * np.log2(p90 / p10)), 1)

    sentences: list[dict[str, Any]] = []
    for group in split_sentences(words):
        first, last = group[0], group[-1]
        change = ending_change(times, f0, last)
        word_dbs = [mean_db(i_times, i_db, w.start, w.end) for w in group]
        known = [d for d in word_dbs if d is not None]
        sentence_db = float(10 * np.log10(np.mean(10 ** (np.array(known) / 10)))) if known else None
        last_db = word_dbs[-1]
        drop = None
        if len(group) >= 2 and sentence_db is not None and last_db is not None:
            drop = sentence_db - last_db
        sentences.append(
            {
                "start_s": round(first.start, 1),
                "end_s": round(last.end, 1),
                "text": " ".join(w.text for w in group),
                "ending": classify_ending(change),
                "change_st": None if change is None else round(change, 1),
                "question": last.text.rstrip("\"')").endswith("?"),
                "mean_db": None if sentence_db is None else round(sentence_db, 1),
                "last_word_db": None if last_db is None else round(last_db, 1),
                "trailing_off": drop is not None and drop > TRAILING_OFF_DB,
            }
        )

    counts = {name: sum(s["ending"] == name for s in sentences) for name in ENDINGS}
    counts["unclear"] = sum(s["ending"] == "unclear" for s in sentences)
    classified = sum(counts[name] for name in ENDINGS)
    statements = [s for s in sentences if not s["question"] and s["ending"] != "unclear"]
    voice = {
        "pitch_floor_hz": floor_hz,
        "pitch_ceiling_hz": ceiling_hz,
        "voiced_s": round(len(voiced_hz) * TIME_STEP_S, 1),
        "endings": counts,
        "endings_pct": {name: _pct(counts[name], classified) for name in ENDINGS},
        "statement_falling_pct": _pct(
            sum(s["ending"] == "falling" for s in statements), len(statements)
        ),
        "pitch_range_st": pitch_range,
        "monotone": pitch_range is not None and pitch_range < MONOTONE_RANGE_ST,
        "trailing_off": [
            {
                "time_s": s["start_s"],
                "text": s["text"],
                "drop_db": round(s["mean_db"] - s["last_word_db"], 1),
            }
            for s in sentences
            if s["trailing_off"]
        ],
        "sentences": sentences,
    }
    return voice, contour(times, f0)


def contour(times: np.ndarray, f0: np.ndarray) -> list[tuple[float, float]]:
    """Voiced pitch points every CONTOUR_STEP_S seconds, small enough for a chart."""
    step = max(1, round(CONTOUR_STEP_S / TIME_STEP_S))
    return [
        (round(float(t), 2), round(float(hz), 1))
        for t, hz in zip(times[::step], f0[::step], strict=True)
        if hz > 0
    ]


def voice_metrics(voice: dict[str, Any]) -> dict[str, float]:
    """Headline numbers for the metric tiles and the metrics table."""
    metrics: dict[str, float] = {}
    if voice.get("pitch_range_st") is not None:
        metrics["pitch_range_st"] = voice["pitch_range_st"]
    if voice.get("statement_falling_pct") is not None:
        metrics["falling_endings_pct"] = voice["statement_falling_pct"]
    if "trailing_off" in voice:
        metrics["trailing_off_count"] = float(len(voice["trailing_off"]))
    return metrics
