"""Voice analysis on synthetic, voice-like signals: harmonic tones with known pitch glides."""

import wave
from pathlib import Path

import numpy as np
import pytest

from speech_model import Word
from voice_analysis import analyse_voice, classify_ending, split_sentences, voice_metrics

RATE = 16000


def write_wav(path: Path, f0: np.ndarray, gain: np.ndarray | None = None) -> Path:
    """A buzzy, voice-like tone (fundamental plus harmonics) following the f0 track (Hz)."""
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    signal = sum(np.sin(k * phase) / k for k in range(1, 8))
    if gain is not None:
        signal = signal * gain
    peak = np.max(np.abs(signal)) or 1.0
    samples = (0.3 * signal / peak * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(samples.tobytes())
    return path


def track(*segments: tuple[float, float, float]) -> np.ndarray:
    """Concatenate (seconds, start Hz, end Hz) glides; frequency moves linearly in semitones."""
    parts = [np.geomspace(start, end, int(seconds * RATE)) for seconds, start, end in segments]
    return np.concatenate(parts)


# Two words: "This is." spoken over 0-1.0 s and 1.0-2.0 s; the ending is the last 400 ms.
WORDS = [Word("This", 0.05, 0.95), Word("is.", 1.0, 1.95)]


def classify(tmp_path, f0, gain=None, words=WORDS):
    voice, contour = analyse_voice(write_wav(tmp_path / "take.wav", f0, gain), words)
    return voice, contour


def test_falling_glide_is_classified_falling(tmp_path):
    voice, contour = classify(tmp_path, track((1.4, 160, 160), (0.6, 200, 120)))
    assert voice["sentences"][0]["ending"] == "falling"
    assert voice["sentences"][0]["change_st"] < -1
    assert voice["endings"] == {"falling": 1, "flat": 0, "rising": 0, "unclear": 0}
    assert voice["statement_falling_pct"] == 100.0
    assert contour and all(75 <= hz <= 300 for _, hz in contour)


def test_rising_glide_is_classified_rising(tmp_path):
    voice, _ = classify(tmp_path, track((1.4, 140, 140), (0.6, 120, 200)))
    assert voice["sentences"][0]["ending"] == "rising"
    assert voice["sentences"][0]["change_st"] > 1
    assert voice["endings_pct"]["rising"] == 100.0
    assert voice["statement_falling_pct"] == 0.0


def test_steady_tone_is_flat_and_flagged_monotone(tmp_path):
    voice, _ = classify(tmp_path, track((2.0, 150, 150)))
    assert voice["sentences"][0]["ending"] == "flat"
    assert voice["pitch_range_st"] < 1
    assert voice["monotone"] is True
    assert voice_metrics(voice)["pitch_range_st"] == voice["pitch_range_st"]


def test_wide_pitch_range_is_not_monotone(tmp_path):
    voice, _ = classify(tmp_path, track((1.0, 110, 220), (1.0, 220, 110)))
    assert voice["pitch_range_st"] > 5
    assert voice["monotone"] is False


@pytest.mark.parametrize(("drop_db", "trailing"), [(12, True), (3, False)])
def test_quiet_last_word_is_trailing_off(tmp_path, drop_db, trailing):
    f0 = track((2.0, 150, 150))
    gain = np.ones_like(f0)
    gain[RATE:] = 10 ** (-drop_db / 20)
    voice, _ = classify(tmp_path, f0, gain)
    assert voice["sentences"][0]["trailing_off"] is trailing
    assert len(voice["trailing_off"]) == int(trailing)
    if trailing:
        assert voice["trailing_off"][0]["time_s"] == 0.1
        assert voice["trailing_off"][0]["drop_db"] > 6
    assert voice_metrics(voice)["trailing_off_count"] == float(trailing)


def test_silence_gives_unclear_endings_and_no_range(tmp_path):
    voice, contour = classify(tmp_path, np.zeros(2 * RATE))
    assert voice["sentences"][0]["ending"] == "unclear"
    assert voice["pitch_range_st"] is None
    assert voice["statement_falling_pct"] is None
    assert contour == []


def test_questions_are_left_out_of_statement_percentage(tmp_path):
    words = [Word("Ready?", 0.05, 0.95), Word("Yes.", 1.0, 1.95)]
    f0 = track((0.4, 140, 140), (0.6, 120, 200), (0.4, 160, 160), (0.6, 200, 120))
    voice, _ = classify(tmp_path, f0, words=words)
    assert [s["ending"] for s in voice["sentences"]] == ["rising", "falling"]
    assert voice["sentences"][0]["question"] is True
    assert voice["endings_pct"] == {"falling": 50.0, "flat": 0.0, "rising": 50.0}
    assert voice["statement_falling_pct"] == 100.0


def test_split_sentences_ignores_ellipsis():
    words = [Word(t, i, i + 0.5) for i, t in enumerate(["We", "go...", "now.", "Then", "stop"])]
    assert [[w.text for w in s] for s in split_sentences(words)] == [
        ["We", "go...", "now."],
        ["Then", "stop"],
    ]


@pytest.mark.parametrize(
    ("change", "label"),
    [(-1.5, "falling"), (-0.5, "flat"), (0.9, "flat"), (1.2, "rising"), (None, "unclear")],
)
def test_classify_ending_thresholds(change, label):
    assert classify_ending(change) == label
