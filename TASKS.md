# Recall-AI — Task Breakdown

Hackathon demo target: **Mac webcam + mic**, split into a **minimal client** (capture only,
no ML dependencies) and a **backend service** (everything model-heavy). The service doesn't
need to run on macOS — for now it's a second process on this same Mac (`localhost`), but
nothing about the client changes if it's later pointed at a different host. See
[RESEARCH.md §F](RESEARCH.md#f-clientservice-split--and-why-it-dodges-macos-compatibility-risk)
for why this split exists (mainly: it keeps TensorFlow/onnxruntime/whisper.cpp build
headaches off the device entirely). Tasks are tagged **[Client]** or **[Service]**.

Tasks are ordered as a dependency chain — each stage builds on the previous one and should
be demo-able on its own before moving to the next. Each task references the GitHub issue it
implements: [drraghavendra/Recall-AI/issues](https://github.com/drraghavendra/Recall-AI/issues).

---

## Stage 0 — Project skeleton
- [ ] **[Client]** Python 3.11+ venv, minimal deps only: `opencv-python`, `sounddevice`,
      `websockets` (or `requests` if polling instead of streaming). No ML libraries here.
- [ ] **[Service]** Separate Python 3.11+ venv, `requirements.txt`: `fastapi`/`uvicorn`
      (or a plain `websockets` server), `deepface` or `insightface`, `silero-vad` (torch
      hub), `whisper.cpp` bindings or `faster-whisper` (or a cloud STT API client),
      `resemblyzer` (self/other speaker check), `sqlite3` (stdlib), an LLM API client
      (Claude/GPT).
- [ ] **[Service]** Define the WebSocket protocol: client→service messages
      `{type: "frame", ts, jpeg_bytes}` (every ~1–2s) and `{type: "audio", ts, pcm_bytes}`
      (continuous chunks); service→client messages `{type: "face_update", ...}`,
      `{type: "persona", face_id, summary}`, `{type: "recall_answer", ...}`,
      `{type: "alert", ...}`.
- [ ] **[Service]** SQLite schema: `faces(face_id, embedding, enrolled_at, name_guess)`,
      `self_voice(embedding, enrolled_at)` (single row — your enrolled voice),
      `face_samples(id, ts, face_id)` (periodic sampling log, scalar `face_id` — simplified
      to 1:1, nullable when no face is focused),
      `conversations(conv_id, face_id, started_at, ended_at, persona_watermark)` (scalar `face_id`; watermark = last segment fed to the persona LLM),
      `transcript_segments(id, ts_start, ts_end, text, speaker, face_id)` (continuous,
      captured independent of face state; `speaker` is `"self"` or `"other"` from the
      Resemblyzer check, `face_id` set only when `speaker="other"` and a face was focused
      at that time — null otherwise),
      `personas(face_id, summary, updated_at)`,
      `reminders(id, text, trigger_at, done)`, `events(id, kind, payload, ts)` (generic
      log for "where did I put X" / "what was I doing").
- [ ] **[Service]** One-time self-voice enrollment step: record a few seconds of the
      user's own voice, compute its Resemblyzer embedding, store in `self_voice`. Needed
      before Stage 3's speaker check can work.
- [ ] **[Client]+[Service]** Smoke test: client connects over WebSocket to `localhost`,
      sends one dummy frame + one dummy audio chunk, service echoes an ack.

## Stage 1 — Face recognition from sampled frames (issue #7)
- [ ] **[Client]** Open webcam with OpenCV (`cv2.VideoCapture(0)`), display live feed
      locally. On a timer (~1–2s interval, not every frame), JPEG-encode one frame and
      send it to the service over the WebSocket.
- [ ] **[Service]** On each received frame, run face detection (DeepFace or InsightFace)
      — may return zero, one, or multiple detected faces.
- [ ] **[Service]** If multiple faces detected, pick the **focused face**: largest
      bounding box / most centered in frame. Compute its embedding; compare against
      `faces` table via cosine similarity.
- [ ] **[Service]** **Find face:** similarity ≥ threshold (start at ~0.6, tune by eye) →
      existing `face_id`. **Create new face:** below threshold → new `face_id`, store
      embedding, optional name label.
- [ ] **[Service]** Write each sample's result (timestamp + the single focused `face_id`,
      or null if none) to `face_samples`; push a `face_update` message back to the client.
- [ ] **[Client]** Draw the most recent `face_update` as an overlay on the live feed.
- [ ] Demo checkpoint: client shows live overlay correctly distinguishing 2+ enrolled
      faces from "unknown"; all recognition work visibly happens service-side (e.g. log it
      there), client only captures/sends/renders.

## Stage 2 — Conversation lifecycle: VAD decides when, face decides who (issue #11)
See [DESIGN.md §7](DESIGN.md#7-conversation-lifecycle--vad-decides-when-face-decides-who).
- [ ] **[Client]** Continuously capture mic audio (`sounddevice`) and stream raw/PCM
      chunks to the service over the WebSocket — no local filtering, keeps the client free
      of VAD/ML dependencies.
- [ ] **[Service]** Run Silero VAD (streaming) on received audio → `speech_start` /
      `speech_end` events. Config block: `min_speech_duration` (~1–1.5s),
      `min_silence_duration` (~5–8s), VAD probability threshold (default 0.5).
- [ ] **[Service]** Maintain the debounced `current_face: face_id | None` from
      `face_samples` (who). A change only counts once confirmed across ≥2 consecutive
      windows.
- [ ] **[Service]** Keep a rolling ~3s audio buffer so speech onset isn't lost while
      waiting for `min_speech_duration` to confirm.
- [ ] **[Service]** **Open** a `conversations` row when speech is sustained past
      `min_speech_duration`; `face_id` = `current_face` at that moment (null → self
      session); `started_at` back-dated to the real speech onset. Opening is what starts
      face↔transcript matching — the match module is idle otherwise.
- [ ] **[Service]** **Handoff**: while open, if a *different* non-null face is confirmed,
      close at that timestamp and open a new row with the new `face_id`. A face merely
      leaving frame does nothing.
- [ ] **[Service]** **Close** when silence exceeds `min_silence_duration`; `ended_at` = last
      speech timestamp.
- [ ] **[Service]** Log transitions (`"Started convo with faceId-1234"` / `"...ended"`);
      push to the client for the demo UI.
- [ ] Demo checkpoint: stand silently in frame → **no** conversation; start talking →
      opens with your face_id; pause 3s → stays open; cough alone in an empty room → no
      conversation; second person takes over the frame and talks → handoff; everyone
      stops talking → closes ~5–8s later.

## Stage 3 — Transcription, split self vs. other, joined to faceId (issue #8)
- [ ] **[Service]** Only transcribe audio inside VAD speech regions (reuses Stage 2's
      VAD — no separate pass).
- [ ] **[Service]** Transcribe speech continuously (whisper.cpp, faster-whisper, or a
      cloud STT API), **independent of whether any conversation is open** — produces
      timestamped text regardless of face state, so self-talk/no-one-in-frame audio is
      still captured.
- [ ] **[Service]** For each transcribed segment, compute its Resemblyzer embedding and
      compare to the enrolled `self_voice` embedding (Stage 0). Match → `speaker="self"`.
      No match → `speaker="other"`, and look up whichever `conversations` row (if any) was
      open across the segment's `[ts_start, ts_end]` window to set `face_id`. Write the row
      to `transcript_segments` with both fields set (`face_id` null for `self` segments or
      unmatched `other` segments with no open conversation).
- [ ] Demo checkpoint: talk while alone → segment tagged `self`, `face_id` null; a known
      face enters and *they* talk → segment tagged `other` + that `face_id`; *you* talk
      during the same conversation → segment tagged `self`, not attributed to them; switch
      conversation partner → new `other` segments tagged with the new `face_id`.

## Stage 4 — Persona & summary generation (issue #9)
See [DESIGN.md §7, "Persona freshness during long conversations"](DESIGN.md#persona-freshness-during-long-conversations).
- [ ] **[Service]** Persona update routine: gather the conversation's `speaker="other"`
      segments **after its `persona_watermark`** (excluding your own `self` segments), call
      an LLM (Claude/GPT) with the existing persona (if any) + that transcript → updated
      persona; upsert into `personas`; advance the watermark; push `persona_updated`.
- [ ] **[Service]** Run it (a) every `persona_update_interval` (~5 min) while a
      conversation is open, skipped if fewer than `persona_min_new_segments` new other
      segments since the last run; (b) once more on `conversation_ended`.
- [ ] Demo checkpoint: two short conversations with the same person → persona reflects
      both, nothing you said; one long conversation (interval set to ~1 min for testing) →
      persona updates mid-conversation, and the final update doesn't duplicate facts from
      earlier checkpoints.

## Stage 5 — Display person summary: the always-on read path (issue #10)
Independent of conversations/VAD — can be built right after Stage 1 using hand-seeded
`personas` rows, before Stage 4 generates real ones.
- [ ] **[Service]** `get_persona(face_id)` read endpoint; push `persona_updated(face_id)`
      whenever the write path (Stage 4) changes a persona.
- [ ] **[Client]** On each `face_update`: if `face_id` changed since the last burst, fetch
      the persona; if none exists (new face, or seen but never talked to), show "You're
      meeting them for the first time" — key this off *no persona*, not `is_new`, which is
      only true on the first burst. If unchanged, keep the cached one — no fetch every 2s.
      Refetch on `persona_updated` for the face on screen (this is what replaces the
      first-time message after the first conversation ends, and what surfaces
      mid-conversation checkpoints).
- [ ] **[Client]** Cache TTL of 3–4 min on the on-screen persona as a safety net — refetch
      even without a push, in case one was missed (dropped/reconnected WebSocket).
- [ ] **[Client]** Thin UI (OpenCV overlay or a small local web page): live feed,
      recognized name/ID, persona panel — shown whenever a known face is in frame, talking
      or not.
- [ ] Demo checkpoint: point camera at an enrolled person who stays silent → persona
      shows immediately; after a conversation with them ends, the panel updates on its own.

## Stage 6 — Recall features
- [ ] **[Service]** **Reminders** (issue #1): simple `reminders` table + background
      check loop; push due reminders as an `alert` message.
- [ ] **[Service]** **Recall item location** (issue #2): log "object mention + location"
      style events (can be manually triggered via a client UI text box for the demo, e.g.
      "left keys on kitchen table"); query by keyword/embedding at recall time ("where are
      my keys?").
- [ ] **[Service]** **Recall current task** (issue #3): log task-start/task-switch events
      (manual trigger for demo); "what was I just doing" returns the most recent task
      event by timestamp.
- [ ] **[Client]** Simple input box to type/ask recall questions, sent to the service;
      render the `recall_answer` that comes back.
- [ ] Demo checkpoint: ask each recall question via the client UI, get correct answer back
      from the service.

## Stage 7 — Alerts & notifications (issue #4)
- [ ] **[Service]** Decide *when* to alert (due reminder, recall answer ready, etc.) and
      push an `alert` message.
- [ ] **[Client]** Render it as a visible banner (UI banner is enough for demo; native
      macOS notification, e.g. via `osascript`, as a stretch goal — note this would be the
      one piece that's inherently client-side since it's a macOS-specific notification).

## Stage 8 — Polish / stability (issue #5: glitches and bugs)
- [ ] Pass over the full pipeline end-to-end: client capture → service face ID →
      conversation event → transcript → persona → client UI, fix rough edges found during
      dry runs.
- [ ] Specifically test the client/service boundary: what happens on a dropped WebSocket
      connection, a slow service response, or the service restarting mid-demo — decide
      whether to buffer/retry on the client or just accept visible glitches for a hackathon
      demo and note it as a known limitation.
- [ ] Note known limitations explicitly in README (face mismatch under bad lighting, no
      diarization for 3+ simultaneous speakers, network dependency between client/service,
      etc.) rather than silently failing in the demo.

## Stage 9 — Market research (issue #6)
- [ ] Separate from technical build: write up competitive landscape (Limitless, Humane,
      Rewind, Friend, Meta Ray-Ban) — pricing, target user, what Recall-AI's "recall what
      you're missing" angle offers that they don't. Can run in parallel with any
      engineering stage above, doesn't block the pipeline.

---

## Suggested build order for a short timebox
1. Stage 0 — get the client/service WebSocket skeleton talking to itself before building
   any real feature on top; this boundary is new risk versus a monolithic app, worth
   proving early.
2. Stage 1 + Stage 5 (face rec + read path, with hand-seeded personas) — the whole
   "see someone, see who they are" story works end-to-end before any audio exists.
3. Stage 2 (VAD conversation events) — the trigger for everything on the write path.
4. Stage 3 + 4 together (transcript → persona) — natural pair, same event trigger; once
   done, swap the hand-seeded personas for generated ones.
5. Stage 6/7 (recall + alerts) last — they're the most "canned"/scriptable for a demo and
   least dependent on the perception pipeline being perfect.
