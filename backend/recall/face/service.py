import asyncio
import logging
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
        self._queue: asyncio.Queue[tuple[float, list[bytes], bool]] = asyncio.Queue(maxsize=max_pending_windows)

    def set_recognizer(self, recognizer: FaceRecognizer) -> None:
        """Swap the provider at runtime; a window already in progress finishes on the old one."""
        self._recognizer = recognizer

    def submit(self, ts: float, jpegs: list[bytes], store_portrait: bool = False) -> None:
        # Stale windows are worthless for a live overlay — drop the oldest instead of falling behind.
        if self._queue.full():
            self._queue.get_nowait()
            log.warning("face recognition is falling behind; dropped a frame window")
        self._queue.put_nowait((ts, jpegs, store_portrait))

    async def run(self) -> None:
        while True:
            ts, jpegs, store_portrait = await self._queue.get()
            try:
                await self._handle_window(ts, jpegs, store_portrait)
            except Exception:
                log.exception("face window at %.3f failed", ts)

    async def _handle_window(self, ts: float, jpegs: list[bytes], store_portrait: bool = False) -> None:
        chosen = await asyncio.to_thread(select_sharpest, jpegs, self._frames_per_window)
        results, others = await self._search(chosen)
        face_id, is_new, confidence = await self._resolve(chosen, results)

        if store_portrait and face_id and chosen and self._db.get_face_portrait(face_id) is None:
            self._db.save_face_portrait(face_id, chosen[0])

        self._db.insert_face_sample(ts, face_id, confidence)
        self._tracker.observe(ts, face_id)
        self._publish(
            {
                "type": "face_update",
                "ts": ts,
                "face_id": face_id,
                "is_new": is_new,
                "confidence": confidence,
                "portrait_url": f"/faces/{face_id}/portrait" if face_id and self._db.get_face_portrait(face_id) else None,
                "others": [other for other in others if other["face_id"] != face_id],
            }
        )

    async def _search(self, images: list[bytes]) -> tuple[list[SearchResult], list[dict]]:
        """The focused (largest) face's result per image, plus known faces also in view in the last image.

        Only the focused face is ever enrolled or attributed speech; other known faces are display-only.
        """
        search_all = getattr(self._recognizer, "search_all", None)
        if search_all is None:
            return [await asyncio.to_thread(self._recognizer.search, image) for image in images], []
        hits_per_image = [await asyncio.to_thread(search_all, image) for image in images]
        results = [hits[0].result if hits else SearchResult(False, None, 0.0) for hits in hits_per_image]
        others: dict[str, dict] = {}
        for hit in hits_per_image[-1][1:] if hits_per_image else []:
            if hit.result.face_id and hit.result.face_id not in others:
                others[hit.result.face_id] = {
                    "face_id": hit.result.face_id,
                    "similarity": round(hit.result.similarity, 3),
                    "box": [round(v, 4) for v in hit.box],
                }
        return results, list(others.values())

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

        face_id = self._db.next_face_id()
        provider_face_id = await asyncio.to_thread(self._recognizer.enroll, unmatched[0], face_id)
        if provider_face_id is None:
            return None, False, 0.0
        self._db.insert_face(face_id, provider_face_id)
        log.info("enrolled new face %s", face_id)
        return face_id, True, len(unmatched) / len(results)
