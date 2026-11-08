import sys
from types import ModuleType, SimpleNamespace

import pytest

import config
import speech_model


@pytest.fixture
def mode(monkeypatch):
    def set_mode(cloud: bool) -> None:
        monkeypatch.setenv("CLOUD_MODE", "true" if cloud else "false")

    return set_mode


def test_cloud_mode_uses_small_int8(mode):
    mode(True)
    assert config.whisper_settings() == ("small.en", "int8")


def test_local_mode_uses_larger_model(mode):
    mode(False)
    assert config.whisper_settings() == ("medium.en", "auto")


@pytest.mark.parametrize("cloud", [True, False])
def test_whisper_model_setting_overrides_default(mode, monkeypatch, cloud):
    mode(cloud)
    monkeypatch.setenv("WHISPER_MODEL", "base.en")
    model, compute_type = config.whisper_settings()
    assert model == "base.en"
    assert compute_type == ("int8" if cloud else "auto")


def test_compute_type_can_be_overridden(mode, monkeypatch):
    mode(True)
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "float32")
    assert config.whisper_settings() == ("small.en", "float32")


@pytest.mark.parametrize(("cloud", "threads"), [(True, 2), (False, 0)])
def test_load_model_passes_settings_to_faster_whisper(mode, monkeypatch, cloud, threads):
    created = {}

    class StubWhisperModel:
        def __init__(self, name, **kwargs):
            created.update(name=name, **kwargs)

    stub = ModuleType("faster_whisper")
    stub.WhisperModel = StubWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", stub)
    mode(cloud)

    speech_model.load_model()

    expected_name, expected_compute = config.whisper_settings()
    assert created == {
        "name": expected_name,
        "device": "cpu",
        "compute_type": expected_compute,
        "cpu_threads": threads,
    }


def test_transcribe_joins_segments_and_counts_speech(tmp_path):
    segments = [
        SimpleNamespace(
            text=" Hello there.",
            start=0.0,
            end=1.5,
            words=[
                SimpleNamespace(word=" Hello", start=0.1, end=0.5),
                SimpleNamespace(word=" ", start=0.5, end=0.6),
            ],
        ),
        SimpleNamespace(text=" ", start=2.0, end=2.1, words=[]),
        SimpleNamespace(text=" Second part.", start=3.0, end=4.0, words=None),
    ]
    received = {}

    class Model:
        def transcribe(self, path, **kwargs):
            received.update(kwargs)
            return iter(segments), None

    transcript = speech_model.transcribe(Model(), tmp_path / "x.wav")
    assert transcript.text == "Hello there. Second part."
    assert transcript.words == [speech_model.Word("Hello", 0.1, 0.5)]
    assert transcript.speech_duration_s == pytest.approx(2.6)
    assert received["vad_filter"] is True
    assert received["word_timestamps"] is True


def test_transcribe_primes_whisper_to_keep_fillers(fake_model, tmp_path):
    speech_model.transcribe(fake_model, tmp_path / "x.wav")
    assert fake_model.kwargs["initial_prompt"] == speech_model.FILLER_PROMPT
    assert "um" in speech_model.FILLER_PROMPT.lower()


def test_detect_speech_converts_samples_to_seconds_with_light_padding(monkeypatch, sample_wav):
    import faster_whisper.vad as vad

    received = {}

    def fake_timestamps(audio, vad_options, sampling_rate):
        received.update(options=vad_options, rate=sampling_rate)
        return [{"start": 0, "end": 8000}, {"start": 24000, "end": 32000}]

    monkeypatch.setattr(vad, "get_speech_timestamps", fake_timestamps)
    assert speech_model.detect_speech(sample_wav) == [(0.0, 0.5), (1.5, 2.0)]
    assert received["rate"] == 16000
    assert received["options"].speech_pad_ms == 50
    assert received["options"].min_silence_duration_ms == 200
