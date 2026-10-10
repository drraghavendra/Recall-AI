"""Is the focused face actually talking? Mouth movement from short frame bursts sent while sound is heard.

The client sends ~5 fps `lip_frames` while its mic picks up sound. For each frame we crop the focused
(largest) face's mouth and, as a reference, its nose — a rigid region that moves with the head but not
with speech — and compare both to the previous frame. A mouth that changes much more than the nose is
talking; head motion and lighting changes move both.
"""

import asyncio
import bisect
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

PATCH_SIZE = (32, 20)  # (width, height) every crop is resized to, so face size doesn't matter

# Returns (BGR image, 5x2 keypoints of the largest face: eyes, nose, mouth corners) or None.
DetectLandmarks = Callable[[bytes], tuple[np.ndarray, np.ndarray] | None]


def _crop(gray: np.ndarray, center: np.ndarray, width: float, height: float) -> np.ndarray | None:
    import cv2

    x0, y0 = int(center[0] - width / 2), int(center[1] - height / 2)
    x1, y1 = int(center[0] + width / 2), int(center[1] + height / 2)
    if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0] or x1 - x0 < 4 or y1 - y0 < 4:
        return None
    patch = cv2.resize(gray[y0:y1, x0:x1], PATCH_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    return patch - patch.mean()  # ignore overall brightness changes


def mouth_and_reference(image: np.ndarray, kps: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    left_eye, right_eye, nose, mouth_left, mouth_right = np.asarray(kps, dtype=np.float32)
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    if eye_distance < 8:
        return None
    mouth = _crop(gray, (mouth_left + mouth_right) / 2, 1.1 * eye_distance, 0.7 * eye_distance)
    reference = _crop(gray, nose, 0.9 * eye_distance, 0.45 * eye_distance)
    if mouth is None or reference is None:
        return None
    return mouth, reference


@dataclass(frozen=True)
class MouthMotion:
    ts: float
    mouth: float
    reference: float


class LipActivityTracker:
    def __init__(
        self,
        min_motion: float,
        motion_ratio: float,
        min_talking_fraction: float,
        min_pairs: int = 3,
        max_gap_s: float = 0.6,
        keep_s: float = 120.0,
    ):
        self._min_motion = min_motion
        self._motion_ratio = motion_ratio
        self._min_talking_fraction = min_talking_fraction
        self._min_pairs = min_pairs
        self._max_gap_s = max_gap_s
        self._keep_s = keep_s
        self._previous: tuple[float, np.ndarray, np.ndarray] | None = None
        self._times: list[float] = []
        self._motions: list[MouthMotion] = []

    def observe(self, ts: float, patches: tuple[np.ndarray, np.ndarray] | None) -> None:
        """One frame's mouth/reference patches for the focused face, or None if no usable face."""
        previous, self._previous = self._previous, None if patches is None else (ts, *patches)
        if patches is None or previous is None or not 0 < ts - previous[0] <= self._max_gap_s:
            return
        mouth = float(np.mean(np.abs(patches[0] - previous[1])))
        reference = float(np.mean(np.abs(patches[1] - previous[2])))
        self._times.append(ts)
        self._motions.append(MouthMotion(ts, mouth, reference))
        while self._times and self._times[0] < ts - self._keep_s:
            self._times.pop(0)
            self._motions.pop(0)

    def _moving(self, motion: MouthMotion) -> bool:
        return motion.mouth >= self._min_motion and motion.mouth >= self._motion_ratio * motion.reference

    def talking_during(self, start: float, end: float, pad_s: float = 0.3) -> bool | None:
        """True/False if the focused face's mouth moved during [start, end]; None if too few frames to tell."""
        lo = bisect.bisect_left(self._times, start - pad_s)
        hi = bisect.bisect_right(self._times, end + pad_s)
        window = self._motions[lo:hi]
        if len(window) < self._min_pairs:
            return None
        fraction = sum(self._moving(m) for m in window) / len(window)
        log.info(
            "lip check %.1f-%.1fs: mouth moving in %.0f%% of %d frame pairs (mean mouth %.1f vs nose %.1f)",
            start, end, fraction * 100, len(window),
            np.mean([m.mouth for m in window]), np.mean([m.reference for m in window]),
        )
        return fraction >= self._min_talking_fraction


class LipService:
    """Feeds `lip_frames` through the provider's landmark detector into a LipActivityTracker."""

    def __init__(self, detect: DetectLandmarks, tracker: LipActivityTracker, max_pending: int = 4):
        self._detect = detect
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
                patches = await asyncio.to_thread(self._patches, jpeg)
                self.tracker.observe(ts, patches)
            except Exception:
                log.exception("lip frame at %.3f failed", ts)

    def _patches(self, jpeg: bytes) -> tuple[np.ndarray, np.ndarray] | None:
        found = self._detect(jpeg)
        return None if found is None else mouth_and_reference(*found)
