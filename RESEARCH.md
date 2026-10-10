# Recall-AI — Research: How Others Build This

Project Scope: prior art for an ambient AI memory assistant (long-term vision: Meta Ray-Ban-style
glasses; hackathon scope: Mac webcam + mic prototype) covering face recognition,
conversation detection, transcript-to-person linking, persona/summary generation, and
recall features. Maps to GitHub issues [#1–#11](https://github.com/drraghavendra/Recall-AI/issues).

---

## A. Comparable products

| Product | Capture | Processing split | Storage | Consent/privacy |
|---|---|---|---|---|
| **Limitless Pendant** (formerly Rewind Pendant) | Beam-forming mics, audio only | Local record → sync (BT/WiFi) → cloud AI (Claude + GPT) for transcription/summary/search | Encrypted cloud ("Confidential Cloud"), auto-delete after 30 days unless saved | "Consent mode": only records new voices after verbal agreement |
| **Rewind.ai** (Mac app) | Screen (5–10 fps) + mic audio (16kHz) | **All on-device** — Whisper-small variant for STT, EasyOCR for screen text, CLIP-like embeddings, local Llama-3.1-8B for summarization on the Neural Engine (INT8) | Local vector DB (LanceDB) + local SQLite | No cloud at all — explicit "local-first" pitch |
| **Humane AI Pin** | Camera + mic + depth/motion sensors | On-device Snapdragon pre-processing, then split to Microsoft cloud for heavier inference | Humane.center cloud account | Physical "Trust Light" lights up whenever a sensor is active; no wake word, must be explicitly engaged |
| **Meta Ray-Ban Display / Live AI** | Glasses only carry camera + 5-mic array; **zero on-glasses AI** | All multimodal processing happens on the paired phone/cloud; glasses are just I/O | Phone app / Meta cloud | Live AI session = continuous capture while active; indicator light while recording |

**Takeaway for Recall-AI:** every product splits into the same four stages — **capture → local lightweight filtering (VAD/face-presence gating) → heavier inference (ASR, embeddings, LLM) → local or cloud store queried at recall time.** Rewind.ai is the closest architectural analogue to what we're prototyping (local-only, Mac, vector DB + SQLite), and is the best template to copy for the hackathon. Consent/indicator patterns (Limitless "consent mode", Humane "Trust Light") are worth a token nod in the demo (e.g. an on-screen "recording" indicator) even though full consent-flow isn't in scope for a demo.

Sources: [Limitless review](https://fast.io/resources/limitless-ai-review-2026/), [Fast Company on Limitless](https://www.fastcompany.com/91090562/limitless-wants-its-ai-wearable-to-be-your-memorys-assistant), [Rewind AI Mac tool specs](https://ucstrategies.com/news/rewind-ai-mac-memory-search-tool-specs-privacy-pricing-2026/), [RewindMCP (SQLite access library)](https://github.com/pedramamini/RewindMCP), [Humane AI Pin architecture](https://www.maginative.com/article/a-closer-look-at-ai-pin-humanes-vision-of-ambient-computing/), [Meta AI glasses help docs](https://www.meta.com/help/ai-glasses/894093646030348/).

---

## B. Face recognition (issue #7: create new face / find face)

Compared `face_recognition` (dlib), `DeepFace`, `InsightFace`, MediaPipe, OpenCV built-ins:

- **Accuracy:** InsightFace (ArcFace) tops LFW benchmarks (~99.8%); DeepFace/FaceNet512 ~98.4%; dlib ~96.8%. For a hackathon demo of 2–5 enrolled faces, all three are "accurate enough" — speed and ease of integration matter more than the last percentage point.
- **Speed on CPU (no GPU):** InsightFace's SCRFD detector hits ~25 FPS on CPU; dlib HOG ~8 FPS; dlib CNN only ~2 FPS. **DeepFace ships a built-in `DeepFace.stream()`** that opens the webcam and runs detection+recognition live — least code to working demo.
- **Recommendation: `DeepFace` with the `Facenet512` model + `retinaface`/`opencv` detector backend**, or `InsightFace` (`buffalo_s` lightweight pack) if frame rate feels sluggish. Both expose a single `represent()`/`embedding` call — no custom model training needed.

**Pipeline pattern (standard across all libraries/products):**
1. Detect face(s) in frame → crop + align.
2. Compute embedding vector (128-d for dlib/Facenet, 512-d for ArcFace).
3. Compare to all enrolled embeddings via **cosine similarity**.
4. If best match ≥ threshold → return existing `faceId`. Else → prompt "new face", generate a new `faceId`, store the embedding (average of first few frames for a cleaner enrollment centroid).
5. Typical cosine-similarity thresholds in the wild: **0.35–0.40 distance** (i.e. similarity ≥ ~0.6–0.65) — tune empirically with your own webcam/lighting rather than trusting a paper's number.

**Storage for hackathon scale (few people, thousands of frames):** a flat list of `{faceId, embedding, enrolled_at}` in SQLite or even a pickled numpy array is sufficient — no need for a real vector DB (FAISS/Chroma) until you have hundreds of identities.

**Multiple faces in one sampled frame — picking "the" focused face:** scope is simplified to 1:1 (one active conversation partner at a time, not a group). When a sample contains more than one detected face (a bystander walks through, someone's in the background), pick the **largest bounding box / most-centered face** as the focused one — cheap, needs no extra model, and works well for a device pointed/worn by the user since you naturally orient toward who you're engaged with. True active-speaker detection (lip-movement tracking across a burst of frames, correlated with audio) would be more accurate but needs continuous frames rather than a single sample and real-time landmark tracking — out of scope for the hackathon, noted here as the upgrade path if the largest/centered heuristic misfires in practice.

Sources: [DeepFace GitHub](https://github.com/serengil/deepface), [Face recognition API comparison 2026](https://mixpeek.com/curated-lists/best-face-recognition-apis), [InsightFace vs face_recognition](https://www.libhunt.com/compare-insightface-vs-face_recognition), [Face recognition pipeline w/ cosine threshold tuning](https://github.com/NajatelOtmani/Face_recognition), [CompreFace threshold doc](https://github.com/lordunix/CompreFace/blob/master/docs/Face-Recognition-Similarity-Threshold.md).

---

## C. Conversation start/end detection (issue #11)

Two design options were considered.

### Option 1 — continuous VAD + face-tracking fusion

- **Voice activity detection (VAD):** `webrtcvad` is the old baseline (edge/IoT grade); **Silero VAD has become the de facto open-source standard** — <2MB model, sub-millisecond per chunk, >95% accuracy, runs real-time on CPU with no GPU. Use Silero VAD, not webrtcvad, for 2026.
- **Speaker diarization** (who's talking, for multi-person scenes): `pyannote.audio` is the strongest open toolkit (handles overlapping speech) but **wants a GPU to run efficiently** — likely too heavy for a hackathon unless the demo is 1:1 (you + one other person), in which case diarization is barely needed since face-presence already tells you *who*.
- **Fusion pattern:** run Silero VAD on the mic stream (cheap, CPU) + face presence/tracking on the camera stream (cheap, but requires processing *every* frame). Emit `conversation_started(faceId)` when (a face is present AND continuously tracked) AND (speech is active) persist for > N seconds (e.g. 2–3s debounce). Emit `conversation_ended(faceId)` when either signal drops out for > M seconds (e.g. 5–8s debounce).
- **Drawback:** requires running face recognition on every frame (continuous video processing) just to maintain the "presence" signal, and ties transcription's start/stop to the fused signal — which means audio captured while no face is recognized (e.g. talking to yourself, phone call, person turned away) never gets transcribed at all, losing exactly the data issues #2/#3 (recall item location / recall current task) depend on.

### Option 2 — periodic face sampling + always-on audio, joined by timestamp (recommended)

This was proposed directly for this project and is the better fit for the hackathon:

- **Face side:** don't process every video frame. Sample a frame every ~1–2s, run face match on it (picking the single **focused face** per §B if more than one is detected), and maintain `current_face: faceId | None` — simplified to a scalar for the 1:1 hackathon scope rather than a set. Apply **hysteresis/debounce on samples**: require the new face to be observed for N consecutive samples (or a grace window, e.g. "no face for >5s") before declaring it changed — a single bad sample (blink, turned head, bad lighting) must not flip the state. On a confirmed change, write a `conversation_started` / `conversation_ended` event to a face-timeline log, each with a precise timestamp.
- **Audio side:** capture and transcribe **continuously and independently of face state** — gate transcription with VAD only as a compute-saving filter ("don't bother transcribing silence"), never as the thing that decides whether a conversation is happening. This produces one continuous timestamped transcript stream regardless of who (if anyone) is recognized on camera.
- **Join, not live-cut:** attribute transcript segments to a faceId as a **post-hoc join by timestamp** against the face-timeline log (at write time or query time), rather than trying to literally start/stop the transcription engine in sync with face-change events. Segments with no face in the timeline default to "self/unknown" — which is exactly the case issues #2/#3 need (self-talk, no conversation partner). This join also needs a second signal — see §D's self-vs-other speaker check — since the face timeline alone can't tell your own voice apart from the other person's within a conversation window.
- **Trade-offs to accept:** boundary precision between two people's transcripts is bounded by the sampling interval (±1–2s of audio near a handoff may land on the wrong side). The scalar `current_face` means genuine group conversations (3+ people) aren't represented — if that's needed later, issue #11's own example event (`"...with faceId-1234 and faceId-457"`) shows the set-based version to fall back to.
- **Why this wins for the hackathon:** far less compute (sampling vs. every-frame face recognition), no GPU-hungry diarization, and — most importantly — never loses transcript data just because no face was recognized.

**Recommendation: build Option 2.**

> **Refined in [DESIGN.md §7](DESIGN.md#7-conversation-lifecycle--vad-decides-when-face-decides-who):** VAD is *not* just an efficiency filter. Face presence alone would open "conversations" with anyone silently standing in frame. Final split: **VAD speech activity decides when** a conversation opens/closes; **the focused face decides who** it's with. Audio is still captured and transcribed continuously regardless of face state.

Sources: [VAD complete 2026 guide](https://picovoice.ai/blog/complete-guide-voice-activity-detection-vad/), [pyannote.audio vs Silero VAD](https://olud.ai/vs/pyannote-audio-vs-silero-vad.html), [pyannote GPU requirement note & VAD tuning](https://www.pyannote.ai/blog/tune-vad-crosstalk-speaker-diarization).

---

## D. Linking transcripts to faceId + persona generation (issues #8, #9, #10)

- **Transcription:** `faster-whisper` (CTranslate2, 4× faster than stock Whisper, INT8/FP16) is the standard local choice, **but on Apple Silicon it can't use the Metal/Neural Engine** — CPU only. **`whisper.cpp` with Core ML acceleration is 2–4× faster on M-series Macs** and is the better pick for this hackathon's hardware. `WhisperLive`/`whisper_streaming` show the standard streaming pattern: sliding-window chunks gated by VAD, ~500–800ms latency.
- **Linking to a person — why face-tagging alone isn't sufficient:** tagging every transcript segment produced while a conversation is "open" with the other person's `faceId` sounds enough, but a conversation has *two* voices in it — yours and theirs. Doing only that would feed your own words into the other person's persona summary (issue #9), which is wrong. This needs one more lightweight component on top of face-tagging: a **binary self-vs-other speaker check**, not full diarization:
  1. **Enroll your own voice once** (a few seconds of audio → one voice embedding), the audio equivalent of face enrollment.
  2. For every transcribed speech segment, compare its embedding to your enrolled one. Match → `speaker: self`. No match → `speaker: other`, and attach whichever face was the focused face (§B/§C) at that timestamp.
  3. Bucket transcripts by `(speaker, face_id)`: everything `self` → your own transcript (this is also exactly what feeds issues #2/#3's "self-talk" recall, with no face required at all); everything `other` + a given faceId → that person's transcript. Over a session this produces multiple named transcripts (self, Person A, Person B, ...) sequentially as the focused face changes — not simultaneous overlapping speech, consistent with the 1:1 scope.
  - **Model pick: `Resemblyzer`** — CPU-only, simple API, good enough for a same/different-speaker decision. Lighter than `SpeechBrain`'s ECAPA-TDNN (more accurate, heavier) and nowhere near as heavy as `pyannote.audio` (already excluded) since this is a 2-class check against one enrolled reference, not open-set multi-speaker diarization.
- **Persona/summary generation — two standard patterns from the memory-architecture literature:**
  1. **Rolling summarization:** after each conversation, call an LLM to produce "summary of (previous persona summary + this session's transcript)" — keeps a single evolving paragraph per faceId instead of re-reading full history every time. This is the cheapest and matches issue #9/#10 directly.
  2. **Session-embedding RAG:** embed each conversation session, store per-faceId, and at query/display time retrieve the top-k most similar/recent sessions and summarize on demand. More scalable, more work — not needed for a hackathon with a handful of conversations per person.
- **Recommendation:** pattern 1 (rolling summary) stored as a plain text field per `faceId` in SQLite, regenerated via one LLM call per conversation-end event. Use Claude or GPT via API — identical to what Limitless does under the hood.

Sources: [whisper.cpp Core ML on Apple Silicon](https://www.digitalapplied.com/blog/local-speech-to-text-whisper-self-hosted-transcription-2026), [faster-whisper](https://localaimaster.com/blog/faster-whisper-guide), [WhisperLive](https://github.com/collabora/WhisperLive), [long-term conversational memory w/ rolling summaries](https://arxiv.org/pdf/2402.17753), [personal memory RAG session-embedding pattern](https://arxiv.org/pdf/2608.27809).

---

## E. Recall features (issues #1, #2, #3) + alerts (#4)

- **Rewind.ai's retrieval pattern** is the clearest template: dense embedding retrieval (cosine similarity, ~0.8 threshold) over a local vector store, rerank, then hand top results to a small local LLM to produce the final answer. This generalizes directly to "where did I put X" and "what was I just doing":
  - Log discrete **events** (object mentions, locations, task starts/stops) with a timestamp + embedding into one table.
  - "Where did I leave X" → embed the query, retrieve nearest event mentioning that object, return its logged context/time.
  - "What was I just doing" → simplest version doesn't even need embeddings — just query the most recent event(s) by timestamp.
- **Reminders (#1):** standard pattern, no research needed beyond "store {text, trigger_time/trigger_condition}" and surface via notification — this is the one piece with no interesting prior art, it's just a scheduled-task feature.
- **Storage for hackathon scale:** **SQLite + a local embeddings column (or sqlite-vec extension)** is the repeatedly-recommended "dependency-free" stack for single-user, local-first memory apps — no need for Chroma/FAISS/LanceDB infra for a demo with minutes-to-hours of data.
- **Alerts/notifications (#4):** for a Mac demo, native OS notifications (e.g. `osascript`/`terminal-notifier` or a simple in-app banner) are sufficient — no separate research needed.

Sources: [Rewind architecture (OCR+embeddings+local LLM)](https://ucstrategies.com/news/rewind-ai-mac-memory-search-tool-specs-privacy-pricing-2026/), [SQLite + vector search dependency-free stack](https://dev.to/hypernexus/sqlite-vector-search-the-dependency-free-ai-memory-stack-28n3), [local-first AI memory architecture](https://hackernoon.com/the-architecture-of-local-first-ai-memory-no-cloud-no-keys-no-read-time-llms).

---

## F. Client/service split — and why it dodges macOS compatibility risk

The device (this Mac) should run a **minimal, dependency-light client**: capture only.
Everything model-heavy runs in a **separate backend service** — which does not need to run
on macOS at all. That second fact is the actual payoff: instead of fighting a library's
build/compatibility story on Apple Silicon, the fix is "run it in the service" (a plain
Linux process — can live on this same Mac as a separate process for now, or move to any
other box later with no client code change).

**Known macOS/Apple-Silicon risk points, pushed to the service for exactly this reason:**
- **DeepFace** pulls in TensorFlow; getting GPU/Metal acceleration working needs the
  `tensorflow-macos` + `tensorflow-metal` plugin pair, which has a history of version-pinning
  headaches. CPU-only TensorFlow installs fine, but it's still a heavier, more fragile
  dependency than the client should carry.
- **InsightFace** (onnxruntime) is actually fairly safe on Apple Silicon (CPU, no GPU
  needed) — but it's still a face-recognition model stack with its own install footprint,
  better kept off the capture device regardless.
- **whisper.cpp with Core ML acceleration** requires a C++ build step plus a Core ML model
  conversion (Xcode command-line tools, `coremltools`) — real setup friction on first run.
  **faster-whisper** avoids that build but is CPU-only on Apple Silicon (confirmed: CTranslate2
  has no Metal/MPS backend) — either way it's model weights + an inference runtime that
  doesn't belong on a "minimal" device.
- **pyannote.audio** (already excluded for being GPU-hungry) would be an even harder sell
  on the client — another reason it stays out of scope entirely rather than becoming a
  client dependency.
- By contrast, **OpenCV** (`cv2.VideoCapture(0)`, AVFoundation backend, automatic) and
  **`sounddevice`** (PortAudio-based mic capture) are both lightweight, well-supported on
  macOS/Apple Silicon, and carry no ML runtime — safe to be the *only* two non-trivial
  dependencies on the device.

**Resulting split:**

| | Client (this Mac) | Service (anywhere) |
|---|---|---|
| Role | Capture + encode + transport + render | Everything model-heavy + storage + decisions |
| Camera | `opencv-python`, sample a frame every ~1–2s, JPEG-encode, send | Face detect + embed + match (DeepFace/InsightFace), face-set debounce |
| Mic | `sounddevice`, stream raw/PCM audio chunks continuously, send | Silero VAD (silence filter) + transcription (whisper.cpp/faster-whisper, or swap for a cloud STT API to dodge local build steps entirely) + timestamp join to conversations |
| LLM | — | Persona/summary generation, recall-query answering |
| Storage | — | SQLite (faces, face_samples, conversations, transcript_segments, personas, reminders, events) |
| UI | Renders whatever the service pushes back (face overlay + persona text + recall answers + alerts) | Decides *when* to alert; client just displays it |
| Dependencies | `opencv-python`, `sounddevice`, a basic HTTP/WebSocket client — nothing ML | Everything else in this doc |

**Transport:** one WebSocket connection client→service carrying two message types —
periodic JPEG frame samples and continuous audio chunks — with the service pushing
JSON events back down the same socket (recognized face + persona text, recall answers,
alerts). Simple enough for a hackathon; no message broker needed. For local development
the service is just a second process on the same Mac (`localhost`); nothing about the
client code changes if it's later pointed at a different host.

Sources: [OpenCV AVFoundation backend on macOS](https://forum.opencv.org/t/warning-deprecated-method-on-mac-os/16504), [OpenCV macOS camera issue discussion](https://github.com/opencv/opencv/issues/23368).

---

## Bottom-line stack recommendation for the hackathon

| Piece | Runs on | Pick | Why |
|---|---|---|---|
| Video capture | Client | `opencv-python` (AVFoundation backend, automatic) | Zero-config on Mac, no ML dependency |
| Audio capture | Client | `sounddevice` | Simple, no ML dependency |
| Face recognition | Service | `DeepFace` (Facenet512) or `InsightFace` (buffalo_s) | One-call embedding + stream helper, no training — kept off-device to avoid TensorFlow/onnxruntime footprint on the client |
| Conversation segmentation | Service | Periodic face sampling (~1–2s), focused-face heuristic (largest/centered), debounced scalar face change | Far cheaper than per-frame tracking; simplified to 1:1 |
| VAD | Service | `Silero VAD` — drives conversation open/close *and* skips transcribing silence | CPU-only, sub-ms; speech is what makes a conversation, face only labels it (DESIGN.md §7) |
| Full diarization | — | **Skip** — not needed | Avoids GPU dependency; a binary self/other check (below) covers what this project actually needs |
| Self vs. other speaker ID | Service | `Resemblyzer`, one enrolled self-embedding, same/different check per segment | Lightweight 2-class check, not open-set diarization — needed so your own words don't end up in the other person's persona |
| Transcription | Service | `whisper.cpp` (Core ML) or `faster-whisper`, or a cloud STT API | Keeps the build/runtime complexity off the client entirely |
| Persona/summary | Service | One LLM call per conversation-end, rolling-summary pattern | Matches issues #9/#10 directly, cheapest to build |
| Storage | Service | SQLite (+ plain numpy/pickle for embeddings) | No vector-DB infra needed at this scale |
| Alerts/UI | Service decides, Client renders | WebSocket push → banner, or native macOS notification | Fast to wire up, good demo visibility |
