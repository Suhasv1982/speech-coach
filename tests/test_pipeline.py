from contextlib import closing

import pytest

import audio
import config
import storage
from pipeline import process_take
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
    assert metrics["words_per_minute"][1] == "wpm"
    assert set(metrics) == set(result.metrics)


def test_local_mode_retake_replaces_metrics(sample_wav, fake_model):
    with closing(storage.connect()) as conn:
        session_id = storage.create_session(conn)
    run(sample_wav, fake_model, cloud=False, session_id=session_id)
    run(sample_wav, fake_model, cloud=False, session_id=session_id)

    with closing(storage.connect()) as conn:
        count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    assert count == 4  # duration, words, pace, coverage: one set, not two
