import subprocess
import wave

import pytest

import audio
import config


def test_save_recording_uses_session_and_task_path(sample_wav):
    path = audio.save_recording("s1", "read_aloud", sample_wav.read_bytes(), ".WAV")
    assert path == config.RECORDINGS_DIR / "s1" / "read_aloud.wav"
    assert path.read_bytes() == sample_wav.read_bytes()


def test_save_recording_replaces_earlier_take_with_other_extension():
    audio.save_recording("s1", "intro", b"old", "m4a")
    path = audio.save_recording("s1", "intro", b"new", "webm")
    assert sorted(p.name for p in path.parent.iterdir()) == ["intro.webm"]


def test_convert_to_wav_gives_16khz_mono_16bit(sample_wav):
    src = audio.save_recording("s1", "word_list", sample_wav.read_bytes(), "wav")
    dst = audio.convert_to_wav(src)

    assert dst.name == "word_list_16k.wav"
    with wave.open(str(dst), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
    assert audio.wav_duration_s(dst) == pytest.approx(0.5, abs=0.02)


def test_convert_to_wav_raises_clear_error_when_ffmpeg_fails(tmp_path, monkeypatch):
    def failing_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Invalid data found")

    monkeypatch.setattr(audio.subprocess, "run", failing_run)
    src = tmp_path / "broken.mp4"
    src.write_bytes(b"not a video")

    with pytest.raises(audio.ConversionError, match="broken.mp4.*Invalid data found"):
        audio.convert_to_wav(src)
