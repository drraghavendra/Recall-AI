import asyncio
import logging
import time

import cv2

from recall_client.api import ApiError, RecallApi
from recall_client.audio import MicStream
from recall_client.camera import encode_jpeg
from recall_client.config import ClientConfig
from recall_client.connection import ServerConnection
from recall_client.messages import audio_message
from recall_client.sampler import FrameWindow
from recall_client.state import ViewState
from recall_client.ui import draw

log = logging.getLogger(__name__)

WINDOW_TITLE = "Recall-AI"
QUIT_KEYS = {ord("q"), 27}


class EventHandler:
    """Routes server events into ViewState; runs on the connection's event loop."""

    def __init__(self, state: ViewState, api: RecallApi):
        self._state = state
        self._api = api
        self._fetches: set[asyncio.Task] = set()

    def __call__(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "face_update":
            face_to_fetch = self._state.face_update(event.get("face_id"))
            if face_to_fetch is not None:
                task = asyncio.get_running_loop().create_task(self._fetch(face_to_fetch))
                self._fetches.add(task)
                task.add_done_callback(self._fetches.discard)
        elif kind == "persona_updated":
            self._state.persona_updated(event["face_id"], event["persona"])
        elif kind == "conversation_started":
            self._state.conversation_started(event.get("face_id"))
        elif kind == "conversation_ended":
            self._state.conversation_ended()

    async def _fetch(self, face_id: str) -> None:
        try:
            persona = await asyncio.to_thread(self._api.get_persona, face_id)
        except ApiError as e:
            log.warning("persona fetch failed: %s", e)
            self._state.fetch_failed(face_id)
            return
        self._state.persona_fetched(face_id, persona)


def run(config: ClientConfig) -> None:
    camera = cv2.VideoCapture(config.camera_index)
    if not camera.isOpened():
        raise SystemExit(
            "Could not open the camera. On macOS, allow camera access for your terminal/IDE in "
            "System Settings -> Privacy & Security -> Camera, then restart it."
        )

    state = ViewState(config.persona_ttl_s)
    connection = ServerConnection(config.ws_url, EventHandler(state, RecallApi(config.server_url)))
    mic = MicStream(
        config.sample_rate,
        config.audio_block_s,
        on_chunk=lambda ts, pcm: connection.send(audio_message(ts, pcm)),
    )
    frames = FrameWindow(config.window_s, config.frames_per_window)

    connection.start()
    mic.start()
    log.info("capturing; press q or Esc in the video window to quit")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                log.warning("camera read failed")
                time.sleep(0.05)
                continue
            message = frames.step(
                time.time(), lambda: encode_jpeg(frame, config.jpeg_width, config.jpeg_quality)
            )
            if message is not None:
                connection.send(message)
            cv2.imshow(WINDOW_TITLE, draw(frame, state.snapshot(), connection.status))
            if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                break
    finally:
        mic.stop()
        connection.stop()
        camera.release()
        cv2.destroyAllWindows()
