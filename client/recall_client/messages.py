"""Builders for the client -> server WebSocket messages (see backend/README.md)."""

import base64
import json


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def frames_message(ts: float, jpegs: list[bytes]) -> str:
    return json.dumps({"type": "frames", "ts": ts, "frames": [_b64(jpeg) for jpeg in jpegs]})


def audio_message(ts: float, pcm: bytes) -> str:
    return json.dumps({"type": "audio", "ts": ts, "pcm": _b64(pcm)})
