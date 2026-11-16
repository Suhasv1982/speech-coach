"""Browser microphone recorder with the browser's voice processing switched off.

st.audio_input records with the browser defaults, which turn on echo cancellation, noise
suppression and auto gain control. Auto gain evens out loudness (hiding a voice that trails
off) and noise suppression can clip quiet sentence endings, so this recorder asks for the raw
microphone signal instead and reports the settings the browser actually applied.
"""

import base64
import hashlib
from dataclasses import dataclass, field
from typing import Any

import streamlit as st

# Settings requested from getUserMedia; the browser reports back what it actually applied.
RAW_AUDIO = {
    "echoCancellation": False,
    "noiseSuppression": False,
    "autoGainControl": False,
    "channelCount": 1,
}
TRACK_SETTINGS = (
    "sampleRate",
    "channelCount",
    "echoCancellation",
    "noiseSuppression",
    "autoGainControl",
)
EXTENSIONS = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/wav": ".wav"}

CSS = """
.rec { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;
       font-family: var(--st-font); color: var(--st-text-color); }
.rec button { font: inherit; padding: 0.4rem 1rem; border-radius: 0.5rem; cursor: pointer;
              border: 1px solid var(--st-border-color, rgba(128,128,128,0.4));
              background: var(--st-secondary-background-color); color: var(--st-text-color); }
.rec button.on { background: var(--st-primary-color); color: white; border-color: transparent; }
.rec button:disabled { opacity: 0.5; cursor: default; }
.rec .status { font-size: 0.9rem; opacity: 0.8; }
"""

JS = """
export default function (component) {
  const { parentElement, setTriggerValue, data } = component;
  if (parentElement.querySelector('.rec')) return;  // already mounted
  const root = document.createElement('div');
  root.className = 'rec';
  root.innerHTML = '<button type="button">● Record</button><span class="status"></span>';
  parentElement.appendChild(root);
  const button = root.querySelector('button');
  const status = root.querySelector('.status');
  status.textContent = 'Microphone with browser processing off (no auto gain or noise filter).';
  let recorder = null, stream = null, timer = null, started = 0;

  const pickType = () => ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4']
    .find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || '';

  const toBase64 = (buffer) => {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i += 0x8000) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(binary);
  };

  async function start() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: data.constraints });
    } catch (err) {
      status.textContent = 'Microphone not available: ' + err.message;
      return;
    }
    const track = stream.getAudioTracks()[0];
    const applied = track.getSettings ? track.getSettings() : {};
    const settings = {};
    for (const name of data.report) if (name in applied) settings[name] = applied[name];
    const chunks = [];
    const type = pickType();
    recorder = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = async () => {
      clearInterval(timer);
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunks, { type: recorder.mimeType || type || 'audio/webm' });
      button.disabled = true;
      status.textContent = 'Sending recording…';
      const audio = toBase64(await blob.arrayBuffer());
      setTriggerValue('recording', {
        audio, mime: blob.type, seconds: (Date.now() - started) / 1000, settings,
      });
      button.disabled = false;
      button.textContent = '● Record again';
      status.textContent = 'Recorded ' + ((Date.now() - started) / 1000).toFixed(1) + ' s.';
    };
    recorder.start(1000);
    started = Date.now();
    button.textContent = '■ Stop';
    button.classList.add('on');
    timer = setInterval(() => {
      status.textContent = 'Recording… ' + ((Date.now() - started) / 1000).toFixed(0) + ' s';
    }, 500);
  }

  function stop() {
    button.classList.remove('on');
    if (recorder && recorder.state !== 'inactive') recorder.stop();
  }

  button.onclick = () => (recorder && recorder.state === 'recording' ? stop() : start());
  return () => {
    clearInterval(timer);
    if (stream) stream.getTracks().forEach((t) => t.stop());
  };
}
"""

_component = st.components.v2.component("speech_coach.raw_recorder", css=CSS, js=JS)


@dataclass
class Recording:
    """Looks like an UploadedFile to the rest of the app: getvalue(), name, file_id."""

    data: bytes
    mime: str
    seconds: float | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "recording" + EXTENSIONS.get(self.mime.split(";")[0].strip(), ".webm")

    @property
    def file_id(self) -> str:
        return hashlib.sha256(self.data).hexdigest()[:16]

    def getvalue(self) -> bytes:
        return self.data


def from_payload(payload: dict[str, Any] | None) -> Recording | None:
    """Decode what the browser sent; None when there is no (or an empty) recording."""
    if not payload or not payload.get("audio"):
        return None
    data = base64.b64decode(payload["audio"])
    if not data:
        return None
    seconds = payload.get("seconds")
    return Recording(
        data=data,
        mime=str(payload.get("mime") or "audio/webm"),
        seconds=None if seconds is None else round(float(seconds), 2),
        settings={k: v for k, v in (payload.get("settings") or {}).items() if k in TRACK_SETTINGS},
    )


def raw_recorder(key: str) -> Recording | None:
    """Show the recorder; returns the new recording on the rerun right after it stops."""
    result = _component(
        key=key,
        data={"constraints": RAW_AUDIO, "report": list(TRACK_SETTINGS)},
        on_recording_change=lambda: None,
    )
    return from_payload(result.recording)
