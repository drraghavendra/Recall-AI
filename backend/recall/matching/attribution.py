import logging
from collections.abc import Callable
from dataclasses import dataclass

from recall.audio.level import ENVELOPE_HOP_S
from recall.audio.types import Utterance
from recall.face.lips import LipActivityTracker
from recall.matching.voices import VoiceBook

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
    background too. With `voices`, the line's voice is then checked against known voiceprints: it may be
    re-credited to whoever's voice it is, or be background if it's not the focused face's voice.
    """

    def __init__(
        self,
        min_level_dbfs: float | None,
        lips: LipActivityTracker | None,
        lips_pick_speaker: bool = False,
        focused_during: Callable[[float, float], str | None] | None = None,
        voices: VoiceBook | None = None,
    ):
        self._min_level_dbfs = min_level_dbfs
        self._lips = lips
        self._lips_pick_speaker = lips_pick_speaker
        self._focused_during = focused_during
        self._voices = voices

    def __call__(self, utterance: Utterance) -> Verdict:
        level = utterance.level_dbfs
        if level is not None:
            log.info("utterance %.1f-%.1fs level %.1f dBFS", utterance.start, utterance.end, level)
        if self._min_level_dbfs is not None and level is not None and level < self._min_level_dbfs:
            return Verdict(reason="too_quiet")
        verdict = self._by_lips(utterance)
        if self._voices is None or verdict.reason:
            return verdict
        focused = verdict.face_id or (self._focused_during(utterance.start, utterance.end) if self._focused_during else None)
        seconds = utterance.end - utterance.start
        judgement = self._voices.judge(focused, utterance.voice, seconds)
        if judgement.background:
            return Verdict(reason="voice_unknown")
        if judgement.face_id and utterance.voice is not None:
            self._voices.learn(judgement.face_id, utterance.voice, seconds)
        return Verdict(face_id=judgement.face_id)

    def _by_lips(self, utterance: Utterance) -> Verdict:
        if self._lips is None:
            return Verdict()
        if self._lips_pick_speaker:
            speaker, decided = self._lips.speaker_during(utterance.start, utterance.end)
            return Verdict(reason="lips_still") if decided and speaker is None else Verdict(face_id=speaker)
        focused = self._focused_during(utterance.start, utterance.end) if self._focused_during else None
        if focused is None:
            return Verdict()
        if self._lips.face_talking(focused, utterance.start, utterance.end, utterance.envelope, ENVELOPE_HOP_S) is False:
            return Verdict(reason="lips_still")
        return Verdict(face_id=focused)
