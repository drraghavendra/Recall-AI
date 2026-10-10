import logging
from collections.abc import Callable
from dataclasses import dataclass

from recall.audio.types import Utterance
from recall.face.lips import LipActivityTracker

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Verdict:
    """Why an utterance is background (`reason`), or which face said it (`face_id`), if known."""

    reason: str | None = None
    face_id: str | None = None


class AttributionGate:
    """Decides who, if anyone in view, said someone else's utterance.

    By default the speaker is the focused face (nearest the centre: who you're looking at), and the
    line is kept only if that face's mouth moved while it was said; otherwise it's background. With
    `lips_pick_speaker`, the speaker is instead whichever watched face's mouth moved most. Lines are
    kept when there are too few lip frames to tell. Speech quieter than `min_level_dbfs` (if set) is
    background too.
    """

    def __init__(
        self,
        min_level_dbfs: float | None,
        lips: LipActivityTracker | None,
        lips_pick_speaker: bool = False,
        focused_during: Callable[[float, float], str | None] | None = None,
    ):
        self._min_level_dbfs = min_level_dbfs
        self._lips = lips
        self._lips_pick_speaker = lips_pick_speaker
        self._focused_during = focused_during

    def __call__(self, utterance: Utterance) -> Verdict:
        level = utterance.level_dbfs
        if level is not None:
            log.info("utterance %.1f-%.1fs level %.1f dBFS", utterance.start, utterance.end, level)
        if self._min_level_dbfs is not None and level is not None and level < self._min_level_dbfs:
            return Verdict(reason="too_quiet")
        if self._lips is None:
            return Verdict()
        if self._lips_pick_speaker:
            speaker, decided = self._lips.speaker_during(utterance.start, utterance.end)
            return Verdict(reason="lips_still") if decided and speaker is None else Verdict(face_id=speaker)
        focused = self._focused_during(utterance.start, utterance.end) if self._focused_during else None
        if focused is None:
            return Verdict()
        if self._lips.face_talking(focused, utterance.start, utterance.end) is False:
            return Verdict(reason="lips_still")
        return Verdict(face_id=focused)
