# Recall-AI capture client (Mac)

Minimal client from [DESIGN.md](../DESIGN.md): captures the camera and mic, streams them to
the [backend](../backend/README.md), and overlays who's in front of you. No ML runs here.

## Install

```bash
cd client
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

**macOS permissions:** the app that launches Python (Terminal, iTerm, VS Code, ...) needs
**Camera** and **Microphone** access in System Settings → Privacy & Security. Grant it, then
restart that app. A denied camera fails fast with a message. A denied mic records silence.

## Run

Start the backend first, then:

```bash
.venv/bin/python -m recall_client run                         # localhost:8000
.venv/bin/python -m recall_client --server http://10.0.0.5:8000 run --frames 8 --window 2
```

`--server` goes *before* the subcommand. **q** or **Esc** in the video window quits.

One-time: enroll your own voice, so your words stay out of other people's personas:

```bash
.venv/bin/python -m recall_client enroll-voice --seconds 10
```

## What it does

- **Camera:** every `--window` seconds (default 2), grabs `--frames` (default 6) frames spread
  across the window, downsizes them to 640px JPEGs, and sends one `frames` message.
- **Mic:** 16 kHz mono s16le in 100ms blocks, sent continuously. Timestamps are anchored by
  sample count so chunks stay gapless (re-anchored after an input overflow).
- **Overlay:** red REC indicator, server status if not connected, "Talking with … / Idle",
  and the persona panel. When a face has no persona yet, the panel says "You're meeting them
  for the first time".
- **Persona cache:** the persona is fetched only when the face in view changes. It refreshes
  instantly on `persona_updated`, with a 3.5 min TTL safety-net refetch (`--persona-ttl`).
- **Connection:** reconnects with backoff (1s → 10s). Audio/frames produced while
  disconnected are dropped (the server starts a fresh stream per connection). If another
  capture client is already connected, the overlay shows "server: another capture client is
  connected".

## Layout

| Module | |
|---|---|
| `state.py` | Read-path view state + persona cache (pure, thread-safe) |
| `sampler.py`, `timing.py`, `messages.py`, `config.py` | Pure: frame windows, audio timestamps, protocol messages |
| `camera.py`, `audio.py`, `api.py`, `connection.py` | IO: OpenCV, PortAudio, HTTP, WebSocket thread |
| `ui.py` | Drawing only (OpenCV, main thread) |
| `app.py` | Wires it together; routes server events into `state` |

Tests (no camera, mic or server needed): `.venv/bin/pytest`
