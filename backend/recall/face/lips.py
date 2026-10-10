"""Is the focused face actually talking? Mouth openness from short frame bursts sent while sound is heard.

The client sends ~10 fps `lip_frames` while its mic picks up sound. For each face in the frame we take
its 68 facial landmarks and measure how far the inner lips are apart relative to the
mouth's width. Talking makes that ratio rise and fall; a listening face holds it roughly steady.
Being a ratio of landmark distances, it ignores lighting, camera noise, face size and head motion.

A line is that face's only if its mouth was moving for much of the time there was sound (scored in
1 s slices, so a smile or an "mm" during someone else's long speech doesn't count), and, when we have
the line's loudness envelope, if the mouth opened and closed in step with the sound.
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

    def __init__(
        self,
        min_openness_std: float,
        min_frames: int = 4,
        keep_s: float = 120.0,
        min_talking_fraction: float = 0.4,
        min_sync: float | None = 0.1,
        slice_s: float = 1.0,
    ):
        self._min_openness_std = min_openness_std
        self._min_frames = min_frames
        self._min_talking_fraction = min_talking_fraction
        self._min_sync = min_sync
        self._slice_s = slice_s
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

    def face_talking(
        self,
        face_id: str,
        start: float,
        end: float,
        envelope: tuple[float, ...] | None = None,
        hop_s: float = 0.05,
        pad_s: float = 0.3,
    ) -> bool | None:
        """Was `face_id` the one talking during [start, end]? None if too few of its frames to tell.

        `envelope` is the line's loudness every `hop_s` from `start`; without it only movement is checked.
        """
        times = self._times.get(face_id) or []
        lo, hi = bisect.bisect_left(times, start - pad_s), bisect.bisect_right(times, end + pad_s)
        if hi - lo < self._min_frames:
            return None
        ts = np.asarray(times[lo:hi])
        openness = np.asarray(self._openness[face_id][lo:hi])
        loud = None if not envelope else _sound_at(np.asarray(envelope), start, hop_s)

        fraction = self._talking_fraction(ts, openness, start, end, loud)
        sync = _sync(ts, openness, loud) if loud is not None else None
        talking = fraction >= self._min_talking_fraction and (
            self._min_sync is None or sync is None or sync >= self._min_sync
        )
        log.info(
            "lip check %.1f-%.1fs (%.1fs): %s moving %.0f%% of the sound (need %.0f%%), sync %s (need %s) -> %s",
            start, end, end - start, face_id, fraction * 100, self._min_talking_fraction * 100,
            "n/a" if sync is None else f"{sync:.2f}", "n/a" if self._min_sync is None else f"{self._min_sync:.2f}",
            "talking" if talking else "still",
        )
        return talking

    def _talking_fraction(self, ts, openness, start, end, loud) -> float:
        """Share of the 1 s slices with sound (and enough frames) in which the mouth was moving."""
        considered = active = 0
        edge = start
        while edge < end:
            right = min(edge + self._slice_s, end)
            if right - edge >= self._slice_s / 2 or considered == 0:
                in_slice = (ts >= edge) & (ts < right)
                has_sound = loud is None or loud(edge, right)
                if in_slice.sum() >= 3 and has_sound:
                    considered += 1
                    active += float(np.std(openness[in_slice])) >= self._min_openness_std
            edge = right
        if considered == 0:  # line too short to slice: fall back to the whole line
            return 1.0 if float(np.std(openness)) >= self._min_openness_std else 0.0
        return active / considered

    def talking_during(self, start: float, end: float, pad_s: float = 0.3) -> bool | None:
        """Was anyone in view talking? True/False, or None if too few frames to tell."""
        speaker, decided = self.speaker_during(start, end, pad_s)
        return None if not decided else speaker is not None


class _Sound:
    """The line's loudness envelope as a function of time."""

    def __init__(self, envelope: np.ndarray, start: float, hop_s: float):
        # Smooth over ~150 ms: syllables, not individual pitch periods.
        kernel = np.ones(3) / 3
        self.values = np.convolve(envelope, kernel, mode="same") if len(envelope) >= 3 else envelope
        self.times = start + (np.arange(len(envelope)) + 0.5) * hop_s
        self._threshold = 0.5 * float(np.percentile(self.values, 90)) if len(self.values) else 0.0

    def __call__(self, left: float, right: float) -> bool:
        """Was there real sound (not just the gaps between words) in [left, right)?"""
        window = self.values[(self.times >= left) & (self.times < right)]
        return bool(window.size) and float(window.max()) >= self._threshold

    def at(self, ts: np.ndarray) -> np.ndarray:
        return np.interp(ts, self.times, self.values)


def _sound_at(envelope: np.ndarray, start: float, hop_s: float) -> "_Sound | None":
    return _Sound(envelope, start, hop_s) if len(envelope) >= 4 else None


def _sync(ts: np.ndarray, openness: np.ndarray, sound: _Sound, max_lag_s: float = 0.3, min_frames: int = 8) -> float | None:
    """How well mouth openness follows the loudness (best Pearson correlation within ±`max_lag_s`,
    allowing for the camera and mic clocks being slightly apart). None if too few frames inside the sound."""
    inside = (ts >= sound.times[0]) & (ts <= sound.times[-1])
    if inside.sum() < min_frames or float(np.std(openness[inside])) < 1e-6:
        return None
    best = None
    for lag in np.arange(-max_lag_s, max_lag_s + 1e-9, 0.05):
        loudness = sound.at(ts[inside] + lag)
        if float(np.std(loudness)) < 1e-9:
            continue
        r = float(np.corrcoef(openness[inside], loudness)[0, 1])
        best = r if best is None else max(best, r)
    return best


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
