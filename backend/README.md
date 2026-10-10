# Recall-AI backend

One FastAPI process implementing [DESIGN.md](../DESIGN.md): an always-on **read path**
(frames → faceId → persona) and a VAD-triggered **write path** (speech → conversation →
transcript ↔ face matching → persona update). Face recognition is selected with
`RECALL_FACE_BACKEND`: AWS Rekognition (default), local InsightFace, or Mock. The persona LLM is
Claude; VAD, transcription and the self/other voice check run locally.

## Run

```bash
cd backend
python3 -m venv .venv && .venv/bin/pip install -e ".[ml,dev]"
cp .env.example .env            # optional overrides
export ANTHROPIC_API_KEY=...    # or `ant auth login`
export AWS_PROFILE=...          # any standard boto3 credential source

.venv/bin/uvicorn recall.main:app --port 8000 --workers 1
```

Open [http://localhost:8000](http://localhost:8000) for the browser capture client. It streams
camera frames and microphone audio over the backend's WebSocket. Browsers generally require a
secure context for camera/microphone access; `localhost` is supported for local development.

For fully local face matching, install the optional model and select it before starting the
backend:

```bash
.venv/bin/pip install -e ".[ml,dev,local-face]"
export RECALL_FACE_BACKEND=insightface
export RECALL_LOCAL_FACE_DB_PATH=recall_faces.sqlite3
.venv/bin/uvicorn recall.main:app --port 8000 --workers 1
```

InsightFace uses the local `buffalo_l` CPU model and a SQLite embedding collection. With the
browser consent toggle enabled, the backend also saves one representative JPEG frame for each
focused person to its local `recall.db`; the original stream is not archived. You can review that
image in the person card and remove it there. Removing the image also pauses further image saving
until consent is enabled again. Face images and embeddings are sensitive data. Use consented
images only, and check the model/license terms before commercial use. `mock` is useful for API/UI
development without a camera or cloud face provider.

Speech-to-text is selected the same way, with `RECALL_STT_BACKEND`:

| Value | Runs | Install | Notes |
|---|---|---|---|
| `whisper` (default) | Local CPU | `[ml]` | faster-whisper, `RECALL_WHISPER_MODEL` (default `base.en`) |
| `parakeet` | Local CPU | `[parakeet]` | NVIDIA Parakeet TDT 0.6B via onnx-asr. `RECALL_PARAKEET_MODEL`: `nemo-parakeet-tdt-0.6b-v2` (English, default) or `-v3` (+24 European languages). ~620 MB int8 download on first start |
| `sarvam` | Sarvam API (paid) | `[sarvam]` | Needs `SARVAM_API_KEY` in the environment. Strong on 22 Indian languages and code-mixed speech; see `RECALL_SARVAM_*` in `.env.example` |

```bash
.venv/bin/pip install -e ".[ml,dev,parakeet]"
export RECALL_STT_BACKEND=parakeet
```

Each provider implements the `Transcriber` protocol in `audio/types.py` and is imported only when
selected, so adding another one is a new module plus a branch in `wiring._build_transcriber`.

`RECALL_FACE_BACKEND` / `RECALL_STT_BACKEND` set the providers at startup; users can also switch
either one live from the browser's **Face** / **Speech** menus (`PUT /providers`). The first switch
to a provider loads its model (minutes if it still has to download); it then stays cached, so
switching back is instant. If a provider can't start (missing AWS credentials, Sarvam key or
extra), the current one stays active and the error is shown. Each face provider keeps its own
identities, so people saved under one are new to another. Live switches last until restart.

`--workers 1` is required: the focused face, the open conversation and the persona locks
are in-process state. First start downloads the Whisper and Silero models and creates the
Rekognition collection.

Enroll your own voice once, so your words never end up in someone else's persona:

```bash
# ~10s of you talking, 16 kHz mono s16le
ffmpeg -f avfoundation -i ":0" -t 10 -ar 16000 -ac 1 -f s16le me.pcm
curl --data-binary @me.pcm -H "Content-Type: application/octet-stream" localhost:8000/self-voice
```

Tests cover the pure logic (tracker, conversation lifecycle, persona watermark) and make no
AWS/Claude calls. They don't need the heavy `[ml]` extra (torch, Whisper, Resemblyzer):

```bash
.venv/bin/pip install numpy pydantic pytest && .venv/bin/pytest
```

### Docker (Railway / Fly.io / Render / EC2)

```bash
docker build -t recall-backend .
docker run -p 8000:8000 -v recall-data:/data \
  -e ANTHROPIC_API_KEY -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e RECALL_AWS_REGION \
  recall-backend
```

Run exactly one instance, with a persistent volume at `/data` (SQLite). Host it in the same
AWS region as the Rekognition collection to keep face-match latency down. A hosted backend
means the Mac streams live audio/video over the network — keep a local run as the demo
fallback if venue Wi-Fi is unknown. Not suited to serverless hosts (e.g. Vercel): it needs a
long-lived WebSocket, in-process state and background tasks.

## API

| Endpoint | |
|---|---|
| `WS /ws` | Capture client. One at a time. |
| `GET /persona/{face_id}` | 404 = no persona yet → "You're meeting them for the first time" |
| `POST /self-voice` | Raw 16 kHz mono s16le PCM, ≥3s |
| `GET /health` | Backend status and the active face / speech-to-text providers. |
| `GET /providers` | Active providers plus the available options. |
| `PUT /providers` | `{"face_backend"?: ..., "stt_backend"?: ...}` switches live; 400 if a provider can't start. Pushes `providers_changed`. |
| `GET /` | Browser capture client. |

**Client → server** (JSON text frames):

```json
{"type": "frames", "ts": 1760090000.0, "frames": ["<base64 jpeg>", "..."]}
{"type": "audio",  "ts": 1760090000.0, "pcm": "<base64 16kHz mono s16le>"}
```

Send one `frames` message per sampling window (client default: 1 frame / 1s, set with
`--window` / `--frames`) and audio continuously
with no gaps, using the same clock (`time.time()`) for both `ts`.

**Server → client**: `face_update {ts, face_id, is_new, confidence, portrait_url}`,
`conversation_started` / `conversation_ended {conv_id, face_id, ts}`,
`transcript {ts, end_ts, text, speaker, face_id}`,
`persona_updated {face_id, persona}`.

Client read-path rule: on `face_update`, fetch `/persona/{face_id}` only when the face
changed; refetch on `persona_updated` for the face on screen, plus a 3–4 min cache TTL as
a safety net.

## Layout

| Module | Role | Swappable behind |
|---|---|---|
| `face/` | Sharpest frame(s) → selected provider search/enroll → debounced `FocusedFaceTracker` timeline | `FaceRecognizer` |
| `audio/` | Silero VAD utterances → selected speech-to-text (faster-whisper, Parakeet or Sarvam) + Resemblyzer self/other. Interfaces in `audio/types.py` (import-light); concrete models in `vad.py`, `transcriber.py`, `parakeet.py`, `sarvam.py`, `speaker.py` | `VoiceActivityDetector`, `Transcriber`, `SpeakerEncoder` |
| `matching/` | Conversation lifecycle: speech decides *when*, face timeline decides *who*, handoffs | — |
| `persona/` | Watermarked incremental updates via Claude structured output | `PersonaWriter` |
| `db.py` | SQLite schema + queries | — |
| `wiring.py` | Builds and starts everything; the only place concrete classes are chosen | — |

## Implementation notes vs. DESIGN.md

- Only the sharpest `RECALL_FRAMES_PER_WINDOW` frames (default 1) of each burst are sent to
  Rekognition, to keep cost and rate limits sane (~$1 per 1,000 calls). Both providers search
  the largest face, which is the focused-face heuristic. A new identity is enrolled from the
  first unmatched frame — a blurry frame can occasionally create a duplicate identity.
- **A conversation opens when a ≥`min_speech_duration` utterance completes**, back-dated to
  its onset, rather than mid-utterance. Same result, simpler, but the
  `conversation_started` event can lag by up to `RECALL_MAX_UTTERANCE_S` during a monologue.
- **The persona model is `claude-sonnet-5-5` at low effort**, updated about once a minute mid-conversation, with server-side refusal
  fallbacks enabled (`fallbacks="default"`). Override with `RECALL_ANTHROPIC_MODEL` /
  `RECALL_PERSONA_EFFORT`.
- **Someone else's speech is only attributed to the focused face if it passes two checks**: it is
  louder than `RECALL_MIN_OTHER_SPEECH_DBFS`, and (InsightFace only) the face's mouth moved while
  it was spoken. The browser client sends ~5 fps `lip_frames` while it hears sound for the second
  check. Rejected speech shows as background in the transcript and is never stored. Both thresholds
  are heuristics; tune them with the per-utterance `level` and `lip check` log lines.
- **Other known faces in frame** are listed in `face_update.others` (display only: only the focused
  face is enrolled or attributed speech). Needs a provider with `search_all` (InsightFace).
- **Transcription must keep up with real time** or the write path falls progressively
  behind (it never drops audio). `base.en` is fine on Apple Silicon; go smaller if not.
- A failed *final* persona update isn't retried (there's no later run for a closed
  conversation); it's logged.
