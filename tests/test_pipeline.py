import json
from contextlib import closing

import pytest

import audio
import config
import pipeline
import storage
from pipeline import process_take
from report import build_markdown
from tasks import load_tasks

TASK = {task.id: task for task in load_tasks()}["read_aloud"]


def run(sample_wav, model, *, cloud, base_dir=None, session_id="s1"):
    return process_take(
        task=TASK,
        session_id=session_id,
        data=sample_wav.read_bytes(),
        ext=".wav",
        source="upload",
        model=model,
        model_label="tiny.en (int8)",
        cloud=cloud,
        base_dir=base_dir,
    )


def files_in(folder):
    return sorted(p.name for p in folder.rglob("*") if p.is_file())


def test_cloud_mode_deletes_audio_after_analysis(sample_wav, fake_model, tmp_path):
    temp = tmp_path / "cloud-temp"
    result = run(sample_wav, fake_model, cloud=True, base_dir=temp)

    assert result.transcript.startswith("Our proposal")
    assert result.metrics["word_count"] == 10
    assert result.wav_path is None
    assert fake_model.calls, "the model should have been called"
    assert files_in(temp) == []
    assert not config.RECORDINGS_DIR.exists()
    assert not config.DB_PATH.exists()


def test_cloud_mode_deletes_audio_when_transcription_fails(sample_wav, tmp_path):
    class BrokenModel:
        def transcribe(self, path, **kwargs):
            raise RuntimeError("model crashed")

    temp = tmp_path / "cloud-temp"
    with pytest.raises(RuntimeError, match="model crashed"):
        run(sample_wav, BrokenModel(), cloud=True, base_dir=temp)
    assert files_in(temp) == []


def test_cloud_mode_deletes_partial_wav_when_conversion_fails(sample_wav, tmp_path, monkeypatch):
    def failing_convert(src):
        audio.wav_path_for(src).write_bytes(b"half written")
        raise audio.ConversionError("bad file")

    monkeypatch.setattr(audio, "convert_to_wav", failing_convert)
    temp = tmp_path / "cloud-temp"
    with pytest.raises(audio.ConversionError):
        run(sample_wav, object(), cloud=True, base_dir=temp)
    assert files_in(temp) == []


def test_local_mode_keeps_files_and_stores_results(sample_wav, fake_model):
    with closing(storage.connect()) as conn:
        session_id = storage.create_session(conn)

    result = run(sample_wav, fake_model, cloud=False, session_id=session_id)

    assert files_in(config.RECORDINGS_DIR / session_id) == ["read_aloud.wav", "read_aloud_16k.wav"]
    assert result.wav_path.endswith("read_aloud_16k.wav")
    with closing(storage.connect()) as conn:
        row = storage.get_session_tasks(conn, session_id)[0]
        metrics = {
            r["name"]: (r["value"], r["unit"])
            for r in conn.execute("SELECT * FROM metrics WHERE session_id = ?", (session_id,))
        }
    assert row["transcript"] == result.transcript
    assert row["model"] == "tiny.en (int8)"
    assert json.loads(row["details"]) == result.details
    assert metrics["words_per_minute"][1] == "wpm"
    assert metrics["mid_phrase_hesitations"][1] == "pauses"
    assert set(metrics) == set(result.metrics)


def test_result_carries_model_and_session_2_details(sample_wav, fake_model, tmp_path):
    result = run(sample_wav, fake_model, cloud=True, base_dir=tmp_path / "t")
    assert result.model == "tiny.en (int8)"
    for name in (
        "fillers_per_minute",
        "pauses_over_0_5s",
        "pauses_over_1_5s",
        "mid_phrase_hesitations",
        "longest_stretch_s",
        "reference_coverage",
    ):
        assert name in result.metrics, name
    assert {"fillers", "pace_windows", "comparison", "warnings"} <= set(result.details)


def test_local_mode_retake_replaces_metrics(sample_wav, fake_model):
    with closing(storage.connect()) as conn:
        session_id = storage.create_session(conn)
    first = run(sample_wav, fake_model, cloud=False, session_id=session_id)
    run(sample_wav, fake_model, cloud=False, session_id=session_id)

    with closing(storage.connect()) as conn:
        count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    assert count == len(first.metrics)  # one set of metrics, not two


def test_result_has_voice_and_recording_details(sample_wav, fake_model, tmp_path):
    result = run(sample_wav, fake_model, cloud=True, base_dir=tmp_path / "t")
    voice = result.details["voice"]
    assert "error" not in voice
    assert {"endings", "endings_pct", "pitch_range_st", "monotone", "trailing_off"} <= set(voice)
    assert voice["sentences"][0]["text"].startswith("Our proposal")
    assert result.metrics["trailing_off_count"] == 0.0
    recording = result.details["recording"]
    assert recording["sample_rate_hz"] == 22050
    assert recording["duration_s"] == 0.5


def test_recorder_settings_are_kept_with_the_recording(sample_wav, fake_model, tmp_path):
    result = process_take(
        task=TASK,
        session_id="s1",
        data=sample_wav.read_bytes(),
        ext=".wav",
        source="mic",
        model=fake_model,
        cloud=True,
        base_dir=tmp_path / "t",
        recorder={"sampleRate": 48000, "autoGainControl": False, "noiseSuppression": False},
    )
    assert result.details["recording"]["autoGainControl"] is False
    assert result.details["recording"]["source"] == "mic"


def test_voice_analysis_failure_is_reported_not_raised(
    sample_wav, fake_model, tmp_path, monkeypatch
):
    def broken(*args, **kwargs):
        raise RuntimeError("praat crashed")

    monkeypatch.setattr(pipeline, "analyse_voice", broken)
    result = run(sample_wav, fake_model, cloud=True, base_dir=tmp_path / "t")

    assert result.details["voice"] == {"error": "RuntimeError: praat crashed"}
    assert result.transcript.startswith("Our proposal")  # the rest of the analysis survives
    assert "pitch_range_st" not in result.metrics
    assert result.warnings == []  # no re-take prompt for a tool failure
    report = build_markdown("s1", [TASK], {TASK.id: result})
    assert "⚠️ Voice analysis failed" in report
    assert "praat crashed" in report
