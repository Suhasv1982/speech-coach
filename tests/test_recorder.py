import base64

import recorder


def test_payload_decodes_to_an_uploadedfile_like_recording():
    payload = {
        "audio": base64.b64encode(b"webm-bytes").decode(),
        "mime": "audio/webm;codecs=opus",
        "seconds": 12.345,
        "settings": {"sampleRate": 48000, "autoGainControl": False, "deviceId": "secret-id"},
    }
    rec = recorder.from_payload(payload)
    assert rec.getvalue() == b"webm-bytes"
    assert rec.name == "recording.webm"
    assert rec.seconds == 12.35
    assert rec.settings == {"sampleRate": 48000, "autoGainControl": False}  # no device id
    assert rec.file_id == recorder.from_payload(payload).file_id
    other = recorder.from_payload({**payload, "audio": base64.b64encode(b"other").decode()})
    assert other.file_id != rec.file_id


def test_safari_mp4_gets_m4a_extension():
    rec = recorder.from_payload({"audio": base64.b64encode(b"x").decode(), "mime": "audio/mp4"})
    assert rec.name == "recording.m4a"


def test_empty_or_missing_payload_is_no_recording():
    assert recorder.from_payload(None) is None
    assert recorder.from_payload({}) is None
    assert recorder.from_payload({"audio": ""}) is None


def test_recorder_asks_for_raw_audio():
    assert recorder.RAW_AUDIO["autoGainControl"] is False
    assert recorder.RAW_AUDIO["noiseSuppression"] is False
    assert recorder.RAW_AUDIO["echoCancellation"] is False
