import logging
from dataclasses import dataclass

from recall.audio.types import Utterance
from recall.face.lips import LipActivityTracker

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Verdict:
    """Why an utterance is background (`reason`), or which face's lips say spoke it (`face_id`), if known."""

    reason: str | None = None
    face_id: str | None = None


class AttributionGate:
    """Decides who, if anyone in view, said someone else's utterance.

    Two checks, each skipped when it has no data: speech quieter than `min_level_dbfs` is someone
    further away; and when we watched mouths during the utterance and none moved, it's no one in view
    (background). With `lips_pick_speaker`, it also names the face whose mouth moved most; otherwise
    the caller credits the focused face.
    """

    def __init__(self, min_level_dbfs: float | None, lips: LipActivityTracker | None, lips_pick_speaker: bool = True):
        self._min_level_dbfs = min_level_dbfs
        self._lips = lips
        self._lips_pick_speaker = lips_pick_speaker

    def __call__(self, utterance: Utterance) -> Verdict:
        level = utterance.level_dbfs
        if level is not None:
            log.info("utterance %.1f-%.1fs level %.1f dBFS", utterance.start, utterance.end, level)
        if self._min_level_dbfs is not None and level is not None and level < self._min_level_dbfs:
            return Verdict(reason="too_quiet")
        if self._lips is None:
            return Verdict()
        speaker, decided = self._lips.speaker_during(utterance.start, utterance.end)
        if decided and speaker is None:
            return Verdict(reason="lips_still")
        return Verdict(face_id=speaker if self._lips_pick_speaker else None)
