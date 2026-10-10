import logging

from recall.audio.types import Utterance
from recall.face.lips import LipActivityTracker

log = logging.getLogger(__name__)


class AttributionGate:
    """Decides whether someone else's utterance can belong to the face in focus.

    Two checks, each skipped when it has no data: speech quieter than `min_level_dbfs` is someone
    further away, and speech during which the focused face's mouth did not move is someone else.
    Returns the rejection reason, or None if the utterance may be attributed.
    """

    def __init__(self, min_level_dbfs: float | None, lips: LipActivityTracker | None):
        self._min_level_dbfs = min_level_dbfs
        self._lips = lips

    def __call__(self, utterance: Utterance) -> str | None:
        level = utterance.level_dbfs
        if level is not None:
            log.info("utterance %.1f-%.1fs level %.1f dBFS", utterance.start, utterance.end, level)
        if self._min_level_dbfs is not None and level is not None and level < self._min_level_dbfs:
            return "too_quiet"
        if self._lips is not None and self._lips.talking_during(utterance.start, utterance.end) is False:
            return "lips_still"
        return None
