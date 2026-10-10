# Setup — macOS & Windows

Recall-AI has two parts. For development, run both on the same machine:

| Part | Folder | What it does | Weight |
|---|---|---|---|
| **Backend** | `backend/` | FastAPI service: selectable face recognition, VAD, transcription, persona (Claude), and browser client | Heavy (~2–3 GB with torch + models) |
| **Client** | `client/` | Optional native capture client; browser client is served by the backend | Light (no ML) |

> Until the open PRs merge, the client lives on `feat/capture-client` and the backend
> startup fix on `fix/backend-ml-compat`. Check out the branch you need, or wait for the merge.

---

## 1. Prerequisites

| | macOS | Windows |
|---|---|---|
| Python **3.11–3.13** (3.12 is the safest) | `brew install python@3.12`, or pyenv (see note) | [python.org installer](https://www.python.org/downloads/windows/) — tick **"Add python.exe to PATH"** |
| Git | `xcode-select --install` | [git-scm.com](https://git-scm.com/download/win) |
| C compiler (backend only) | Comes with Xcode Command Line Tools | **Microsoft C++ Build Tools** → workload **"Desktop development with C++"** — *or* use Docker (§3b) |
| Docker *(optional)* | Docker Desktop | Docker Desktop (WSL 2 backend) |

**pyenv on macOS:** run `brew install xz` *before* `pyenv install`. Otherwise Python is built
without `lzma` and the backend fails with `No module named '_lzma'`. Check with
`python -c "import lzma"`.

**Accounts** (backend only, when using cloud providers):
- **Anthropic API key** — for persona generation.
- **AWS account** with Rekognition access if using the default provider. Minimal IAM policy:
  ```json
  {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
    "Action": ["rekognition:CreateCollection", "rekognition:IndexFaces", "rekognition:SearchFacesByImage"],
    "Resource": "*"}]}
  ```

```bash
git clone https://github.com/drraghavendra/Recall-AI.git
cd Recall-AI
```

---

## 2. Credentials

The SDKs read credentials from the **shell environment** or `~/.aws`. They do **not** read
`backend/.env` — that file is only for `RECALL_*` settings.

**AWS** — either way works on both OSes:
```bash
aws configure            # AWS CLI: writes ~/.aws/credentials (recommended)
```
or set `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in the shell that runs the backend.
Use the same region everywhere (`RECALL_AWS_REGION`, default `us-east-1`).

**Anthropic** — set it in the shell that starts the backend:

```bash
# macOS (zsh/bash)
export ANTHROPIC_API_KEY="sk-ant-..."
```
```powershell
# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

---

## 3. Backend

### 3a. Native

**macOS**
```bash
cd backend
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[ml,dev]"
cp .env.example .env                      # optional: tune RECALL_* settings
.venv/bin/uvicorn recall.main:app --port 8000 --workers 1
```

**Windows (PowerShell)** — requires the C++ Build Tools from §1 for the audio extra:
```powershell
cd backend
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[ml,dev]"
copy .env.example .env                    # optional
.venv\Scripts\python -m uvicorn recall.main:app --port 8000 --workers 1
```
Calling `.venv\Scripts\python` directly avoids PowerShell's script-activation policy.

### 3b. Docker (any OS — the easiest route on Windows)
```bash
cd backend
docker build -t recall-backend .
docker run -p 8000:8000 -v recall-data:/data \
  -e ANTHROPIC_API_KEY -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e RECALL_AWS_REGION \
  recall-backend
```
`-e NAME` (without a value) passes through the variable from your shell. In PowerShell,
replace the trailing `\` with a backtick `` ` ``.

### Notes
- **`--workers 1` is required.** The face and conversation state lives in the process.
- **First start downloads models** (Whisper `base.en` from Hugging Face, plus Silero VAD) and
  creates the Rekognition collection. This can take a few minutes on a slow connection.
- **Check it's up:** `curl http://localhost:8000/health` — on Windows PowerShell use
  `curl.exe …` or `Invoke-RestMethod http://localhost:8000/health`.
- **Backend on another machine:** add `--host 0.0.0.0` and allow Python through the
  firewall prompt (macOS Firewall / Windows Defender → private networks).

---

### Choose a face recognition provider

The default provider is AWS Rekognition. To use the local InsightFace provider instead, install
the optional extra and set these variables in the same PowerShell window before launching:

```powershell
cd backend
.venv\Scripts\python -m pip install -e ".[ml,dev,local-face]"
$env:RECALL_FACE_BACKEND = "insightface"
$env:RECALL_LOCAL_FACE_DB_PATH = "recall_faces.sqlite3"
```

InsightFace uses the same `buffalo_l` CPU recognition model as the Whisper IMG face module. It
stores embeddings locally in SQLite. When the web client's consent toggle is enabled, Recall-AI
also stores one representative JPEG frame per recognized person in the backend database; the
person card lets you view and remove it. The first run may download model files. Check model
licensing before commercial use. Use face matching only with the people’s explicit consent. For a
no-cloud UI smoke run, `RECALL_FACE_BACKEND=mock` skips AWS face recognition.

### Browser client

With the backend running, open **http://localhost:8000** in a modern browser, allow camera and
microphone access, confirm consent, then choose **Start camera & mic**. The browser client sends
frames and audio to the selected backend; it does not save captured images. Use the Stop button
to end the stream.

## 4. Optional native client

**macOS**
```bash
cd client
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m recall_client run
```

**Windows (PowerShell)**
```powershell
cd client
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m recall_client run
```

Options (put `--server` **before** the subcommand):
```bash
python -m recall_client --server http://192.168.1.20:8000 run --camera 1 --window 5 --frames 1
```
Face sampling defaults to **1 frame every 5 seconds** (~720 Rekognition calls/hour). Use
`--window` (seconds) and `--frames` (frames per window) to change it. Press **q** or **Esc** in
the video window to quit. Only **one** capture client can connect
at a time.

### Camera & mic permissions

| | macOS | Windows |
|---|---|---|
| Camera | System Settings → Privacy & Security → **Camera** → enable the app that launches Python (Terminal, iTerm, VS Code…), then **restart that app** | Settings → Privacy & security → **Camera** → turn on **"Let desktop apps access your camera"** |
| Microphone | Same, under **Microphone** | Same, under **Microphone** → **"Let desktop apps access your microphone"** |
| Which mic is used | System default input (Sound settings) | Default recording device (Sound settings) |

A camera that can't be opened fails immediately with a message. A denied mic doesn't error —
it silently records nothing, so check permissions first if no transcripts appear.

---

## 5. Enroll your own voice (once)

This keeps *your* words out of other people's personas. With the backend running:
```bash
# macOS
.venv/bin/python -m recall_client enroll-voice --seconds 10
```
```powershell
# Windows
.venv\Scripts\python -m recall_client enroll-voice --seconds 10
```
Speak normally for the whole 10 seconds. Re-run it any time to replace the profile.

---

## 6. Smoke test checklist

1. `/health` returns `{"status": "ok", "face_backend": "…"}`.
2. Open `http://localhost:8000`, confirm consent, and start camera/mic capture.
3. Verify the browser displays the selected face provider and a live camera preview.
4. Point the camera at someone new → **"You're meeting them for the first time."**
5. Let them talk (≥ ~1.2s at a time) → the status shows **Talking with …**
6. Stop talking for ~6s → the conversation closes, and the persona panel fills in a few
   seconds later.

Run the tests (no camera, mic or cloud needed): `.venv/bin/pytest` on macOS or
`.venv\Scripts\python -m pytest` on Windows, in either `backend/` or `client/`.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Microsoft Visual C++ 14.0 or greater is required` (Windows, backend install) | `webrtcvad` compiles from source. Install the C++ Build Tools (§1), or use Docker (§3b). |
| `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools ≥ 81 removed it. `pip install "setuptools<81"` (pinned on `fix/backend-ml-compat`). |
| `No module named '_lzma'` (macOS + pyenv) | `brew install xz`, then `pyenv install --force <version>` and recreate the venv. |
| `AccessDeniedException` / `UnrecognizedClientException` from Rekognition | AWS credentials missing in the backend's shell, wrong region, or IAM policy missing the 3 actions in §1. |
| InsightFace provider fails to import | Install `.[ml,dev,local-face]` and confirm `RECALL_FACE_BACKEND=insightface`. |
| Persona never updates; logs show Claude `401` | `ANTHROPIC_API_KEY` isn't set in the shell that started the backend (`.env` doesn't count). |
| Client says "another capture client is connected" | Close the other client — only one stream at a time. |
| Camera won't open | Grant permission (§4), close Zoom/Teams/FaceTime, or try `--camera 1`. |
| Your own words show up in someone's persona | Your voice isn't enrolled, or the enrollment was noisy — redo §5 in a quiet room. |
| Transcripts lag further and further behind | The CPU can't keep up with Whisper. Set `RECALL_WHISPER_MODEL=tiny.en`. |
