import base64
import json

import pytest

from recall_client.config import ws_url_for
from recall_client.messages import audio_message, frames_message
from recall_client.sampler import FrameWindow


def test_frames_message_matches_backend_protocol():
    message = json.loads(frames_message(12.5, [b"\xff\xd8one", b"\xff\xd8two"]))
    assert message["type"] == "frames"
    assert message["ts"] == 12.5
    assert [base64.b64decode(f) for f in message["frames"]] == [b"\xff\xd8one", b"\xff\xd8two"]


def test_audio_message_matches_backend_protocol():
    message = json.loads(audio_message(3.0, b"\x01\x00\x02\x00"))
    assert message == {"type": "audio", "ts": 3.0, "pcm": base64.b64encode(b"\x01\x00\x02\x00").decode()}


def run_window(window, times):
    grabbed, messages = [], []

    def grab():
        grabbed.append(len(grabbed))
        return f"frame{len(grabbed) - 1}".encode()

    for t in times:
        message = window.step(t, grab)
        if message:
            messages.append(json.loads(message))
    return grabbed, messages


def test_window_samples_n_frames_spread_across_the_window():
    window = FrameWindow(window_s=2.0, frames_per_window=4)
    camera_ticks = [i / 30 for i in range(61)]  # 30 fps for just over 2s
    grabbed, messages = run_window(window, camera_ticks)

    assert len(messages) == 1
    assert messages[0]["ts"] == 0.0
    assert len(messages[0]["frames"]) == 4
    assert len(grabbed) == 5  # 4 in the first window + the first frame of the next


def test_window_ts_is_the_window_start_and_never_exceeds_n_frames():
    window = FrameWindow(window_s=2.0, frames_per_window=3)
    ticks = [100.0 + i * 0.1 for i in range(41)]
    _, messages = run_window(window, ticks)
    assert [m["ts"] for m in messages] == pytest.approx([100.0, 102.0])
    assert all(len(m["frames"]) == 3 for m in messages)


def test_ws_url_is_derived_from_server_url():
    assert ws_url_for("http://localhost:8000") == "ws://localhost:8000/ws"
    assert ws_url_for("https://recall.example.com/") == "wss://recall.example.com/ws"
    with pytest.raises(ValueError):
        ws_url_for("localhost:8000")
