# Recall-AI backend

One FastAPI process implementing [DESIGN.md](../DESIGN.md): an always-on **read path**
(frames → faceId → persona) and a VAD-triggered **write path** (speech → conversation →
transcript ↔ face matching → persona update). Face recognition is selected with
`RECALL_FACE_BACKEND`: AWS Rekognition (default), local InsightFace, or Mock. The persona LLM is
Claude. Audio uses local Silero VAD and self/other voice checks; transcription defaults to local
Whisper, with optional Saaras v4 code-mix transcription and optional DeepFilterNet enhancement.

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

### Smartphone as the capture sensor

Open the client on the phone and choose **Start camera & mic**; mobile browsers are asked to use
the rear camera, mono audio, echo cancellation, noise suppression, and automatic gain control.
The phone sends its capture stream over Wi-Fi to the backend WebSocket, which does the processing.
`localhost` on a phone refers to the phone itself, so a phone connecting to a laptop needs the
laptop's LAN address and a trusted HTTPS certificate (camera/mic browser permissions are restricted
on plain HTTP LAN origins). Run Uvicorn bound to the LAN interface with that certificate, and open
the matching `https://<laptop-address>:8000` URL on the phone. Keep devices on a trusted network.

The current face pipeline still selects and follows one focused/largest face. It does not yet track
several identities in parallel or assign utterances by lip motion. MediaPipe face boxes alone do not
solve identity tracking or active-speaker diarization; those need temporal face IDs, aligned audio/video
timestamps, and evaluation on multi-speaker recordings.

### Optional Saaras code-mix transcription

Saaras sends each detected speech utterance to Sarvam's hosted API. Set the key in the environment
and enable the provider (local Whisper stays the default):

```powershell
$env:SARVAM_API_KEY = "your-key"
$env:RECALL_TRANSCRIBER_BACKEND = "sarvam"
$env:RECALL_SARVAM_MODEL = "saaras:v4"
$env:RECALL_SARVAM_MODE = "codemix"
$env:RECALL_SARVAM_LANGUAGE_CODE = "unknown"
$env:RECALL_SARVAM_KEYTERMS = '["Koramangala", "swalpa adjust maadi", "BMTC", "Indiranagar"]'
```

Restart the backend after setting these values. Keyterms are optional, limited to 50 entries and
64 characters each, and bias recognition rather than guaranteeing a result. REST transcription is
limited to audio files under 30 seconds; this app currently caps each utterance at 15 seconds.
Audio is transmitted to Sarvam when this provider is selected. Do not enable it for conversations
that must remain local.

### Optional DeepFilterNet enhancement

Install the enhancement extra and select it before running the server:

```powershell
python -m pip install -e ".[ml,dev,denoise]"
$env:RECALL_SPEECH_ENHANCER = "deepfilternet"
```

On Windows, building the DeepFilterNet Python extension may require Cargo/Rust and Microsoft C++
Build Tools. Leave `RECALL_SPEECH_ENHANCER=none` if those tools are not installed.

The enhancer runs on each VAD utterance before transcription; self-voice classification continues
to use the captured audio. It requires the DeepFilterNet model and PyTorch runtime. Noise suppression
can help with background noise, but it does not separate overlapping speakers; evaluate it with
recordings from the target rooms before relying on its transcripts.

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
| `GET /health` | Backend status and selected face provider. |
| `GET /` | Browser capture client. |

**Client → server** (JSON text frames):

```json
{"type": "frames", "ts": 1760090000.0, "frames": ["<base64 jpeg>", "..."]}
{"type": "audio",  "ts": 1760090000.0, "pcm": "<base64 16kHz mono s16le>"}
```

Send one `frames` message per sampling window (client default: 1 frame / 5s, set with
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
| `audio/` | Silero VAD utterances → faster-whisper text + Resemblyzer self/other. Interfaces in `audio/types.py` (import-light); concrete models in `vad.py`, `transcriber.py`, `speaker.py` | `VoiceActivityDetector`, `Transcriber`, `SpeakerEncoder` |
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
- **The persona model is `claude-opus-5-5` at low effort**, with server-side refusal
  fallbacks enabled (`fallbacks="default"`). Override with `RECALL_ANTHROPIC_MODEL` /
  `RECALL_PERSONA_EFFORT`.
- **Transcription must keep up with real time** or the write path falls progressively
  behind (it never drops audio). `base.en` is fine on Apple Silicon; go smaller if not.
- A failed *final* persona update isn't retried (there's no later run for a closed
  conversation); it's logged.
