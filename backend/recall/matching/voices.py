"""Speaker identity: a voiceprint per face, learnt from the lines credited to that face.

Every accepted line from someone else adds its voice embedding to the credited face's samples. A
face's voiceprint is the mean of its samples that agree with each other, so the odd background line
that slipped in early doesn't skew it. Once a face has enough samples, a new line said while that face
is in focus is checked against it:
- the voice matches the focused face: theirs;
- it clearly matches another known face better, and that face is in view: credited to them instead
  (never to someone who isn't there: their voiceprint may have learnt someone else's voice early on,
  e.g. the wearer's own before "Learn my voice");
- it matches nobody: background (someone off camera or far away).
Until a face has a voiceprint, its lines are accepted as before (and teach it).
"""

import logging
from dataclasses import dataclass

import numpy as np

from recall.db import Database

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceJudgement:
    """`face_id`: whose voice it is (None if unknown); `background`: it's nobody we should credit."""

    face_id: str | None
    background: bool = False


def _normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def voiceprint(samples: list[np.ndarray], agree: float = 0.6) -> np.ndarray:
    """Mean of the samples that agree with the overall mean (drops outliers, if they're a minority)."""
    stacked = np.stack(samples)
    centre = _normalise(stacked.mean(axis=0))
    agreeing = stacked[stacked @ centre >= agree]
    if len(agreeing) * 2 >= len(stacked):
        centre = _normalise(agreeing.mean(axis=0))
    return centre


class VoiceBook:
    def __init__(
        self,
        db: Database,
        same_threshold: float = 0.65,
        other_threshold: float = 0.75,
        min_samples: int = 3,
        min_seconds: float = 1.0,
        keep: int = 20,
    ):
        self._db = db
        self._same = same_threshold
        self._other = other_threshold
        self._min_samples = min_samples
        self._min_seconds = min_seconds
        self._keep = keep
        self._samples: dict[str, list[np.ndarray]] = {
            face_id: [np.frombuffer(blob, dtype=np.float32) for blob in blobs]
            for face_id, blobs in db.voice_samples().items()
        }
        self._prints: dict[str, np.ndarray] = {}
        for face_id in self._samples:
            self._refresh(face_id)

    def _refresh(self, face_id: str) -> None:
        samples = self._samples.get(face_id, [])
        if len(samples) >= self._min_samples:
            self._prints[face_id] = voiceprint(samples)

    def learn(self, face_id: str, voice: np.ndarray, seconds: float) -> None:
        if seconds < self._min_seconds:
            return  # too short for a reliable embedding
        voice = np.asarray(voice, dtype=np.float32)
        samples = self._samples.setdefault(face_id, [])
        samples.append(voice)
        del samples[: -self._keep]
        self._db.add_voice_sample(face_id, voice.tobytes(), seconds, self._keep)
        self._refresh(face_id)

    def judge(
        self, focused: str | None, voice: np.ndarray | None, seconds: float, in_view: set[str] | None = None
    ) -> VoiceJudgement:
        """Whose voice is this line, given who was in focus and who else is in view (None: anyone)?
        Too short or no voiceprints to compare: no opinion (the focused face)."""
        if voice is None or seconds < self._min_seconds:
            return VoiceJudgement(focused)
        candidates = None if in_view is None else set(in_view) | ({focused} if focused else set())
        scores = {
            face_id: float(np.dot(voice, vp))
            for face_id, vp in self._prints.items()
            if candidates is None or face_id in candidates
        }
        if not scores:
            return VoiceJudgement(focused)
        best, best_score = max(scores.items(), key=lambda item: item[1])
        focused_score = scores.get(focused) if focused else None
        log.info(
            "voice check (%.1fs, focus %s, in view %s): %s",
            seconds, focused, sorted(candidates) if candidates is not None else "any",
            ", ".join(f"{face} {score:.2f}" for face, score in sorted(scores.items())),
        )
        if best != focused and best_score >= self._other and (focused_score is None or best_score > focused_score + 0.05):
            return VoiceJudgement(best)  # clearly someone else in view
        if focused_score is None:
            return VoiceJudgement(focused)  # no voiceprint for the face in focus yet: can't tell
        if focused_score >= self._same:
            return VoiceJudgement(focused)
        return VoiceJudgement(None, background=True)
