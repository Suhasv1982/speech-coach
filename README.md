# Speech Coach

A Streamlit app for practising executive speaking. Each assessment session has five
tasks: read aloud, a word list, an introduction, an unseen executive question and a
project story. For each task you record in the browser or upload an audio/video file.
The app converts it to 16 kHz mono WAV, transcribes it with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), and shows simple metrics
(duration, word count, pace in words per minute, and for reading tasks the share of
reference words spoken). Each session ends with a downloadable report and metrics JSON.

## Two modes

| | Local mode (default) | Cloud practice mode (`CLOUD_MODE=true`) |
|---|---|---|
| Speech model | `medium.en`, compute type `auto` | `small.en`, compute type `int8` |
| Recordings | Kept in `recordings/<session_id>/` | Temp folder, deleted after each analysis |
| History | SQLite in `data/speech_coach.db`; reports in `reports/` | Current browser tab only; download the report |

Override the model with `WHISPER_MODEL` / `WHISPER_COMPUTE_TYPE` in either mode.

## Run locally

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    Mac/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py --server.maxUploadSize 1024
```

Then open http://localhost:8501 and allow microphone access. In VS Code, F5 with the
"Speech Coach" configuration does the same. Settings go in `.env` (copy `.env.example`).
The first analysis downloads the speech model (about 1.5 GB for `medium.en`).

`.streamlit/config.toml` sets a 200 MB upload limit so the cloud app stays within memory;
the `--server.maxUploadSize 1024` flag raises it for local runs.

Lint and test: `ruff check .`, `ruff format --check .`, `pytest`.

Supported Python versions: 3.11 and 3.14. To test on 3.14 locally with
[uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.14
uv venv --python 3.14 .venv314
uv pip install --python .venv314/Scripts/python.exe -r requirements-dev.txt   # Mac/Linux: .venv314/bin/python
.venv314/Scripts/python.exe -m pytest
```

The real-transcription test needs the `tiny.en` model cached once:
`python -c "from faster_whisper import download_model; download_model('tiny.en')"`.

## Deploy to Streamlit Community Cloud

The free tier gives about 1 GB of guaranteed memory (up to about 2.7 GB) and 2 CPU cores.
Storage is wiped on every restart and the app sleeps after 12 hours without visitors,
which is why cloud mode keeps nothing on disk.

1. **Merge to `main`.** Make sure the CI check passes and the repository is public on GitHub.
2. **Sign in.** Go to https://share.streamlit.io and choose **Continue with GitHub**.
   Authorise Streamlit to read your repositories when asked.
3. **Create the app.** Click **Create app**, then **Deploy a public app from GitHub**, and fill in:
   - Repository: `Suhasv1982/speech-coach`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: choose a subdomain, for example `speech-coach-practice`
4. **Leave Advanced settings alone.** No secrets and no Python choice are needed:
   - The app detects Community Cloud (it runs from `/mount/src`) and then defaults to
     cloud mode with `small.en` / `int8`.
   - Python 3.14 (the cloud default) and 3.11 (local) are both supported and tested in CI.

   Secrets are only for overrides, for example (see `.streamlit/secrets.example.toml`):
   ```toml
   WHISPER_MODEL = "base.en"
   ```
5. **Click Deploy.** The first build installs `requirements.txt` and takes a few minutes.
   Every package is pinned to an exact version, so the cloud gets the same tested stack
   as CI. No `packages.txt` is needed: ffmpeg comes from the `imageio-ffmpeg` package.
6. **First use.** The first analysis downloads the speech model and shows
   "Loading speech model, about a minute on first use". After a restart or after the app
   wakes from sleep, the model downloads again.
7. **Use it.** Allow microphone access, record or upload each task, and on the summary page
   use **Download report (.md)** and **Download metrics (.json)** before closing the tab.

**Later changes.** Pushing to `main` redeploys automatically. Change secrets under the
app's **⋮ → Settings → Secrets**; the app restarts with the new values. To wake a
sleeping app, open its URL and click **Yes, get this app back up!**.

**Check the log.** Open **Manage app** (bottom right of the app) to see the log. At
startup the app writes one line like:

```
Speech Coach startup: Community Cloud detected: True (app root /mount/src/speech-coach);
cloud mode: True; Python 3.14.x; secrets: no secrets file; sources: CLOUD_MODE=default, ...
```

`secrets: no secrets file` means nothing was saved under Secrets, and
`secrets: unreadable (...)` means the TOML has a syntax error. Values are never logged.

**Upgrading packages.** Change the pins in `requirements.txt`, run
`pip install -r requirements-dev.txt` locally, and run `pytest`. `tests/test_pinned_stack.py`
checks that what's installed matches the pins and that faster-whisper still decodes and
transcribes audio with them.

**Uploads.** The upload limit is 200 MB. A 3-minute phone video at 1080p can be larger,
so record at 720p or use an audio-only recording for the cloud app.

**Secrets.** Never commit `.streamlit/secrets.toml` or `.env` (both are git-ignored). The
app logs only where each setting came from, never its value.
