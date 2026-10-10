import asyncio
import logging
import uuid
from collections import Counter

from recall.db import Database
from recall.events import Publish
from recall.face.frames import select_sharpest
from recall.face.recognizer import FaceRecognizer, SearchResult
from recall.face.tracker import FocusedFaceTracker

log = logging.getLogger(__name__)


class FaceService:
    """Always-on read path: frame window -> focused faceId -> face_update event + tracker timeline."""

    def __init__(
        self,
        recognizer: FaceRecognizer,
        tracker: FocusedFaceTracker,
        db: Database,
        publish: Publish,
        frames_per_window: int,
        max_pending_windows: int = 2,
    ):
        self._recognizer = recognizer
        self._tracker = tracker
        self._db = db
        self._publish = publish
        self._frames_per_window = frames_per_window
        self._queue: asyncio.Queue[tuple[float, list[bytes]]] = asyncio.Queue(maxsize=max_pending_windows)

    def submit(self, ts: float, jpegs: list[bytes]) -> None:
        # Stale windows are worthless for a live overlay — drop the oldest instead of falling behind.
        if self._queue.full():
            self._queue.get_nowait()
            log.warning("face recognition is falling behind; dropped a frame window")
        self._queue.put_nowait((ts, jpegs))

    async def run(self) -> None:
        while True:
            ts, jpegs = await self._queue.get()
            try:
                await self._handle_window(ts, jpegs)
            except Exception:
                log.exception("face window at %.3f failed", ts)

    async def _handle_window(self, ts: float, jpegs: list[bytes]) -> None:
        chosen = await asyncio.to_thread(select_sharpest, jpegs, self._frames_per_window)
        results = [await asyncio.to_thread(self._recognizer.search, image) for image in chosen]
        face_id, is_new, confidence = await self._resolve(chosen, results)

        self._db.insert_face_sample(ts, face_id, confidence)
        self._tracker.observe(ts, face_id)
        self._publish(
            {"type": "face_update", "ts": ts, "face_id": face_id, "is_new": is_new, "confidence": confidence}
        )

    async def _resolve(
        self, images: list[bytes], results: list[SearchResult]
    ) -> tuple[str | None, bool, float]:
        matched = [r.face_id for r in results if r.face_id]
        if matched:
            face_id, votes = Counter(matched).most_common(1)[0]
            return face_id, False, votes / len(results)

        unmatched = [image for image, r in zip(images, results) if r.face_found]
        if not unmatched:
            return None, False, 0.0

        face_id = uuid.uuid4().hex
        provider_face_id = await asyncio.to_thread(self._recognizer.enroll, unmatched[0], face_id)
        if provider_face_id is None:
            return None, False, 0.0
        self._db.insert_face(face_id, provider_face_id)
        log.info("enrolled new face %s", face_id)
        return face_id, True, len(unmatched) / len(results)
