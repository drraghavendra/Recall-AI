"""Is the focused face actually talking? Mouth openness from short frame bursts sent while sound is heard.

The client sends ~5 fps `lip_frames` while its mic picks up sound. For each face in the frame we take
its 68 facial landmarks and measure how far the inner lips are apart relative to the
mouth's width. Talking makes that ratio rise and fall; a listening face holds it roughly steady.
Being a ratio of landmark distances, it ignores lighting, camera noise, face size and head motion.
"""

import asyncio
import bisect
import logging
from collections.abc import Callable

import numpy as np

log = logging.getLogger(__name__)

# 68-point (iBUG) indices: outer mouth corners, and the three top/bottom inner-lip pairs.
MOUTH_CORNERS = (48, 54)
INNER_LIP_PAIRS = ((61, 67), (62, 66), (63, 65))

# jpeg -> every usable face in it as (box as fractions of the frame, 68 landmarks).
DetectMouths = Callable[[bytes], list[tuple[tuple[float, float, float, float], np.ndarray]]]
# Recognised faces seen in the last moments: face_id -> box, from the 1-per-second recognition frames.
KnownBoxes = Callable[[], dict[str, tuple[float, float, float, float]]]


def mouth_openness(landmarks: np.ndarray) -> float | None:
    points = np.asarray(landmarks, dtype=np.float32)[:, :2]
    width = float(np.linalg.norm(points[MOUTH_CORNERS[0]] - points[MOUTH_CORNERS[1]]))
    if width < 4:
        return None
    gap = np.mean([np.linalg.norm(points[top] - points[bottom]) for top, bottom in INNER_LIP_PAIRS])
    return float(gap / width)


class LipActivityTracker:
    """Mouth-openness timelines for every recognised face, to tell which one is talking."""

    def __init__(self, min_openness_std: float, min_frames: int = 4, keep_s: float = 120.0):
        self._min_openness_std = min_openness_std
        self._min_frames = min_frames
        self._keep_s = keep_s
        self._times: dict[str, list[float]] = {}
        self._openness: dict[str, list[float]] = {}

    def observe(self, face_id: str, ts: float, openness: float | None) -> None:
        """One lip frame's mouth openness for `face_id`, or None if its mouth wasn't usable."""
        if openness is None:
            return
        times = self._times.setdefault(face_id, [])
        values = self._openness.setdefault(face_id, [])
        index = bisect.bisect(times, ts)  # frames can arrive slightly out of order
        times.insert(index, ts)
        values.insert(index, openness)
        while times and times[0] < times[-1] - self._keep_s:
            times.pop(0)
            values.pop(0)

    def _spread(self, face_id: str, start: float, end: float, pad_s: float) -> float | None:
        times = self._times.get(face_id)
        if not times:
            return None
        lo, hi = bisect.bisect_left(times, start - pad_s), bisect.bisect_right(times, end + pad_s)
        window = self._openness[face_id][lo:hi]
        return float(np.std(window)) if len(window) >= self._min_frames else None

    def speaker_during(self, start: float, end: float, pad_s: float = 0.3) -> tuple[str | None, bool]:
        """Who was talking during [start, end]: (face_id, True) for the face whose mouth moved most,
        (None, True) if we watched faces and none of their mouths moved, (None, False) if too few frames."""
        spreads = {face_id: spread for face_id in self._times if (spread := self._spread(face_id, start, end, pad_s)) is not None}
        if not spreads:
            return None, False
        speaker, spread = max(spreads.items(), key=lambda item: item[1])
        log.info(
            "lip check %.1f-%.1fs: %s (need %.3f)",
            start, end, ", ".join(f"{face} {value:.3f}" for face, value in sorted(spreads.items())), self._min_openness_std,
        )
        return (speaker, True) if spread >= self._min_openness_std else (None, True)

    def face_talking(self, face_id: str, start: float, end: float, pad_s: float = 0.3) -> bool | None:
        """Did `face_id`'s mouth move during [start, end]? None if too few of its frames to tell."""
        spread = self._spread(face_id, start, end, pad_s)
        if spread is None:
            return None
        log.info("lip check %.1f-%.1fs: %s %.3f (need %.3f)", start, end, face_id, spread, self._min_openness_std)
        return spread >= self._min_openness_std

    def talking_during(self, start: float, end: float, pad_s: float = 0.3) -> bool | None:
        """Was anyone in view talking? True/False, or None if too few frames to tell."""
        speaker, decided = self.speaker_during(start, end, pad_s)
        return None if not decided else speaker is not None


def _match_faces(
    detected: list[tuple[tuple[float, float, float, float], np.ndarray]],
    known: dict[str, tuple[float, float, float, float]],
    max_distance: float = 0.12,
) -> list[tuple[str, np.ndarray]]:
    """Pair faces in a lip frame with recognised faces by nearest box centre (each used once)."""
    centre = lambda box: (box[0] + box[2] / 2, box[1] + box[3] / 2)
    pairs = []
    for i, (box, _) in enumerate(detected):
        for face_id, known_box in known.items():
            (ax, ay), (bx, by) = centre(box), centre(known_box)
            distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if distance <= max_distance:
                pairs.append((distance, i, face_id))
    matched, used_faces, used_ids = [], set(), set()
    for _, i, face_id in sorted(pairs):
        if i not in used_faces and face_id not in used_ids:
            used_faces.add(i)
            used_ids.add(face_id)
            matched.append((face_id, detected[i][1]))
    return matched


class LipService:
    """Feeds `lip_frames` through the provider's landmark detector into a LipActivityTracker.

    Every face in the frame is measured and matched to a recognised person by position, so when two
    people are in view we can tell which one is actually talking.
    """

    def __init__(self, detect: DetectMouths, known_boxes: KnownBoxes, tracker: LipActivityTracker, max_pending: int = 4):
        self._detect = detect
        self._known_boxes = known_boxes
        self.tracker = tracker
        self._queue: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue(maxsize=max_pending)

    def submit(self, ts: float, jpeg: bytes) -> None:
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait((ts, jpeg))

    async def run(self) -> None:
        while True:
            ts, jpeg = await self._queue.get()
            try:
                detected = await asyncio.to_thread(self._detect, jpeg)
                for face_id, landmarks in _match_faces(detected, self._known_boxes()):
                    self.tracker.observe(face_id, ts, mouth_openness(landmarks))
            except Exception:
                log.exception("lip frame at %.3f failed", ts)
