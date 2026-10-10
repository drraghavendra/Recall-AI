import asyncio
import logging
from collections import Counter

from recall.db import Database
from recall.events import Publish
from recall.face.frames import portrait_candidate, select_sharpest
from recall.face.recognizer import FaceBox, FaceRecognizer, SearchResult
from recall.face.tracker import FocusedFaceTracker, PresenceTracker

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
        presence: PresenceTracker | None = None,
    ):
        self._recognizer = recognizer
        self._tracker = tracker
        self._db = db
        self._publish = publish
        self._frames_per_window = frames_per_window
        self._presence = presence or PresenceTracker(confirm_windows=2, absent_windows=3)
        # Where each recognised face was last seen (window ts, box), so lip frames can be matched to people.
        self._boxes: dict[str, tuple[float, FaceBox]] = {}
        self._latest_ts = 0.0
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
        results, others, focused_box = await self._search(chosen)
        face_id, is_new, confidence = await self._resolve(chosen, results)

        self._latest_ts = max(self._latest_ts, ts)
        if face_id and focused_box:
            self._boxes[face_id] = (ts, focused_box)
        for other in others:
            self._boxes[other["face_id"]] = (ts, tuple(other["box"]))
        if store_portrait and face_id and chosen:
            await self._update_portrait(face_id, chosen[-1], focused_box)

        self._db.insert_face_sample(ts, face_id, confidence)
        self._tracker.observe(ts, face_id)
        seen = {other["face_id"] for other in others} | ({face_id} if face_id else set())
        self._presence.observe(seen)
        self._publish(
            {
                "type": "face_update",
                "ts": ts,
                "face_id": face_id,
                "is_new": is_new,
                "confidence": confidence,
                "portrait_url": self._portrait_url(face_id),
                "others": [other for other in others if other["face_id"] != face_id],
                # Debounced views for the UI: one bad frame doesn't change these.
                "focused": self._tracker.current,
                "present": sorted(self._presence.present),
            }
        )

    def known_boxes(self, max_age_s: float = 2.5) -> dict[str, FaceBox]:
        """Recognised faces seen in the last couple of windows, by where they were in the frame."""
        return {face_id: box for face_id, (ts, box) in self._boxes.items() if ts >= self._latest_ts - max_age_s}

    def _portrait_url(self, face_id: str | None) -> str | None:
        info = self._db.portrait_info(face_id) if face_id else None
        # The version changes whenever a better photo replaces the old one, so clients reload it.
        return None if info is None else f"/faces/{face_id}/portrait?v={info[1]:.0f}"

    async def _update_portrait(self, face_id: str, image: bytes, box: FaceBox | None) -> None:
        """Keep the best photo seen so far: replace it when a clearly better one (bigger, sharper face) comes along."""
        candidate = await asyncio.to_thread(portrait_candidate, image, box)
        if candidate is None:
            return
        jpeg, score = candidate
        current = self._db.portrait_info(face_id)
        if current is None or score > current[0] * 1.15:
            self._db.save_face_portrait(face_id, jpeg, score)

    async def _search(self, images: list[bytes]) -> tuple[list[SearchResult], list[dict], FaceBox | None]:
        """The focused (largest) face's result per image, plus known faces also in view in the last image.

        Only the focused face is ever enrolled or attributed speech; other known faces are display-only.
        """
        search_all = getattr(self._recognizer, "search_all", None)
        if search_all is None:
            return [await asyncio.to_thread(self._recognizer.search, image) for image in images], [], None
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
        last = hits_per_image[-1] if hits_per_image else []
        return results, list(others.values()), last[0].box if last else None

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
