"""Is the focused face actually talking? Mouth openness from short frame bursts sent while sound is heard.

The client sends ~5 fps `lip_frames` while its mic picks up sound. For each frame we take the focused
(largest) face's 68 facial landmarks and measure how far the inner lips are apart relative to the
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

# (jpeg, previous face centre) -> the focused face's 68 landmarks as a (68, 2+) array and its centre, or None.
DetectMouth = Callable[[bytes, tuple[float, float] | None], tuple[np.ndarray, tuple[float, float]] | None]


def mouth_openness(landmarks: np.ndarray) -> float | None:
    points = np.asarray(landmarks, dtype=np.float32)[:, :2]
    width = float(np.linalg.norm(points[MOUTH_CORNERS[0]] - points[MOUTH_CORNERS[1]]))
    if width < 4:
        return None
    gap = np.mean([np.linalg.norm(points[top] - points[bottom]) for top, bottom in INNER_LIP_PAIRS])
    return float(gap / width)


class LipActivityTracker:
    def __init__(self, min_openness_std: float, min_frames: int = 4, keep_s: float = 120.0):
        self._min_openness_std = min_openness_std
        self._min_frames = min_frames
        self._keep_s = keep_s
        self._times: list[float] = []
        self._openness: list[float] = []

    def observe(self, ts: float, openness: float | None) -> None:
        """One lip frame's mouth openness for the focused face, or None if no usable face."""
        if openness is None:
            return
        index = bisect.bisect(self._times, ts)  # frames can arrive slightly out of order
        self._times.insert(index, ts)
        self._openness.insert(index, openness)
        while self._times and self._times[0] < self._times[-1] - self._keep_s:
            self._times.pop(0)
            self._openness.pop(0)

    def talking_during(self, start: float, end: float, pad_s: float = 0.3) -> bool | None:
        """True/False if the focused face's mouth moved during [start, end]; None if too few frames to tell."""
        lo = bisect.bisect_left(self._times, start - pad_s)
        hi = bisect.bisect_right(self._times, end + pad_s)
        window = self._openness[lo:hi]
        if len(window) < self._min_frames:
            return None
        spread = float(np.std(window))
        log.info(
            "lip check %.1f-%.1fs: openness std %.3f over %d frames (range %.2f-%.2f, need %.3f)",
            start, end, spread, len(window), min(window), max(window), self._min_openness_std,
        )
        return spread >= self._min_openness_std


class LipService:
    """Feeds `lip_frames` through the provider's landmark detector into a LipActivityTracker."""

    def __init__(self, detect: DetectMouth, tracker: LipActivityTracker, max_pending: int = 4):
        self._detect = detect
        self.tracker = tracker
        self._queue: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue(maxsize=max_pending)
        self._last_centre: tuple[float, float] | None = None

    def submit(self, ts: float, jpeg: bytes) -> None:
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait((ts, jpeg))

    async def run(self) -> None:
        while True:
            ts, jpeg = await self._queue.get()
            try:
                found = await asyncio.to_thread(self._detect, jpeg, self._last_centre)
                if found is None:
                    self._last_centre = None
                    continue
                landmarks, self._last_centre = found
                self.tracker.observe(ts, mouth_openness(landmarks))
            except Exception:
                log.exception("lip frame at %.3f failed", ts)
