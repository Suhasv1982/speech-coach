"""Checks that the installed speech stack matches requirements.txt and actually works.

The av/faster-whisper pairing broke on Community Cloud (av 19 removed an argument that
faster-whisper 1.2.1 passes), so these tests exercise the real libraries, not mocks.
"""

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
from packaging.requirements import Requirement

import config
import speech_model

REQUIREMENTS = config.ROOT / "requirements.txt"
TEST_MODEL = "tiny.en"


def pinned_requirements() -> list[Requirement]:
    reqs = []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            reqs.append(Requirement(line))
    return reqs


def test_every_requirement_is_pinned_exactly():
    for req in pinned_requirements():
        specs = list(req.specifier)
        assert len(specs) == 1 and specs[0].operator == "==", f"{req} must use =="


def test_installed_versions_match_requirements():
    mismatches = []
    for req in pinned_requirements():
        if req.marker and not req.marker.evaluate():
            continue
        wanted = next(iter(req.specifier)).version
        try:
            installed = version(req.name)
        except PackageNotFoundError:
            installed = "not installed"
        if installed != wanted:
            mismatches.append(f"{req.name}: requirements.txt {wanted}, installed {installed}")
    assert not mismatches, "Run `pip install -r requirements-dev.txt`:\n" + "\n".join(mismatches)


def test_faster_whisper_decodes_sample_wav_with_pinned_av(sample_wav):
    """This is the exact call that crashed with av 19 (unexpected 'metadata_errors')."""
    from faster_whisper import decode_audio

    samples = decode_audio(str(sample_wav), sampling_rate=16000)
    assert len(samples) == pytest.approx(8000, abs=200)  # 0.5 s at 16 kHz


@pytest.fixture
def cached_tiny_model(monkeypatch):
    """Load tiny.en from the local model cache only; never download during tests."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(TEST_MODEL, device="cpu", compute_type="int8", local_files_only=True)
    except Exception as err:
        message = (
            f"Whisper model {TEST_MODEL} is not in the local cache ({type(err).__name__}). "
            f'Download it once with: python -c "from faster_whisper import download_model; '
            f"download_model('{TEST_MODEL}')\""
        )
        if os.getenv("REQUIRE_REAL_MODEL") == "1":
            pytest.fail(message)
        pytest.skip(message)


@pytest.mark.real_model
def test_real_transcription_of_sample_wav(cached_tiny_model, sample_wav, tmp_path):
    """Full pinned stack: av decode, onnxruntime VAD, ctranslate2 + tokenizers decoding."""
    wav = Path(tmp_path / "sample.wav")
    wav.write_bytes(sample_wav.read_bytes())

    transcript = speech_model.transcribe(cached_tiny_model, wav)

    assert isinstance(transcript, speech_model.Transcript)
    assert isinstance(transcript.text, str)
    # The sample is a pure tone, so the silence filter should find little or no speech.
    assert 0.0 <= transcript.speech_duration_s <= 0.6
