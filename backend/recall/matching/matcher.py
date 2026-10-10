import logging
from collections.abc import Callable
from dataclasses import dataclass

from recall.audio.types import Utterance
from recall.db import Database
from recall.events import Publish
from recall.face.tracker import FocusedFaceTracker
from recall.matching.attribution import Verdict

log = logging.getLogger(__name__)


@dataclass
class Conversation:
    conv_id: int
    face_id: str | None
    started_at: float
    last_speech_end: float
    ended_at: float | None = None
    segment_count: int = 0


class ConversationMatcher:
    """Write path entry point (DESIGN.md §7): speech decides *when*, the face timeline decides *who*.

    Driven entirely by the audio pipeline's clock: a conversation opens when an utterance of at least
    `min_speech_s` arrives (back-dated to its onset), closes after `min_silence_s` of silence, and is
    handed off when a different face is confirmed. Closed conversations are finalized (persona update)
    only once no in-progress utterance could still belong to them.

    Someone else's speech that `gate` rejects (too quiet, or the focused face's lips weren't moving) is
    background: it is shown as such but never stored, attributed, or allowed to open or extend a
    conversation. Because the audio pipeline emits at least every `max_utterance_s`, speech still in
    progress after that long can't be ours, so continuous background chatter doesn't hold a
    conversation open forever.
    """

    def __init__(
        self,
        db: Database,
        tracker: FocusedFaceTracker,
        min_speech_s: float,
        min_silence_s: float,
        publish: Publish,
        on_finalized: Callable[[Conversation], None],
        gate: Callable[[Utterance], Verdict] | None = None,
        max_utterance_s: float | None = None,
        on_line: Callable[[], None] | None = None,
    ):
        self._db = db
        self._tracker = tracker
        self._min_speech_s = min_speech_s
        self._min_silence_s = min_silence_s
        self._publish = publish
        self._on_finalized = on_finalized
        self._gate = gate
        self._max_utterance_s = max_utterance_s
        self._on_line = on_line
        self._open: Conversation | None = None
        self._closing: list[Conversation] = []

    @property
    def open_conversation(self) -> Conversation | None:
        return self._open

    def on_utterance(self, utterance: Utterance) -> None:
        verdict = self._gate(utterance) if utterance.speaker == "other" and self._gate is not None else Verdict()
        if verdict.reason:
            log.info("background speech (%s): %r", verdict.reason, utterance.text)
            self._publish_transcript(utterance, "background", None, verdict.reason)
            return
        midpoint = (utterance.start + utterance.end) / 2
        self._apply_handoff(midpoint)
        conv = self._conversation_at(midpoint)
        if conv is None:
            if self._open is not None or utterance.end - utterance.start < self._min_speech_s:
                return
            conv = self._start(self._tracker.face_at(utterance.start), utterance.start)

        face_id = None
        if utterance.speaker == "other":
            # Whose lips moved while it was said; else the face in focus for most of the line; else the
            # conversation's face.
            face_id = verdict.face_id or self._db.majority_face(utterance.start, utterance.end) or conv.face_id
        self._db.insert_segment(
            conv.conv_id, utterance.start, utterance.end, utterance.text, utterance.speaker, face_id
        )
        self._publish_transcript(utterance, utterance.speaker, face_id)
        conv.segment_count += 1
        if utterance.speaker == "other" and self._on_line is not None:
            self._on_line()
        if conv is self._open:
            conv.last_speech_end = max(conv.last_speech_end, utterance.end)

    def on_audio_time(self, ts: float, in_speech: bool) -> None:
        self._apply_handoff(ts)
        open_conv = self._open
        if open_conv and self._settled(open_conv.last_speech_end + self._min_silence_s, ts, in_speech):
            self._end(open_conv, open_conv.last_speech_end)
        for conv in [c for c in self._closing if self._settled(c.ended_at, ts, in_speech)]:
            self._finalize(conv)

    def _settled(self, since: float, ts: float, in_speech: bool) -> bool:
        """Has `since` passed, with no speech in progress that could still be attributed before it?"""
        if ts < since:
            return False
        return not in_speech or (self._max_utterance_s is not None and ts - since >= self._max_utterance_s)

    def _publish_transcript(self, utterance: Utterance, speaker: str, face_id: str | None, reason: str | None = None):
        event = {
            "type": "transcript",
            "ts": utterance.start,
            "end_ts": utterance.end,
            "text": utterance.text,
            "speaker": speaker,
            "face_id": face_id,
        }
        if reason:
            event["reason"] = reason
        self._publish(event)

    def on_stream_end(self) -> None:
        if self._open:
            self._end(self._open, self._open.last_speech_end)
        for conv in list(self._closing):
            self._finalize(conv)

    def _finalize(self, conv: Conversation) -> None:
        self._closing.remove(conv)
        if conv.face_id is not None and conv.segment_count:
            self._on_finalized(conv)

    def _apply_handoff(self, ts: float) -> None:
        conv = self._open
        if conv is None:
            return
        change_ts, face_id = self._tracker.change_at(ts)
        if face_id is None or face_id == conv.face_id:
            return
        boundary = max(change_ts, conv.started_at)
        self._end(conv, boundary)
        self._start(face_id, boundary)

    def _conversation_at(self, ts: float) -> Conversation | None:
        if self._open and ts >= self._open.started_at:
            return self._open
        for conv in self._closing:
            if conv.started_at <= ts <= conv.ended_at:
                return conv
        return None

    def _start(self, face_id: str | None, started_at: float) -> Conversation:
        conv_id = self._db.open_conversation(face_id, started_at)
        conv = Conversation(conv_id, face_id, started_at, last_speech_end=started_at)
        self._open = conv
        log.info("conversation %d started with %s", conv_id, face_id or "self")
        self._publish({"type": "conversation_started", "conv_id": conv_id, "face_id": face_id, "ts": started_at})
        return conv

    def _end(self, conv: Conversation, ended_at: float) -> None:
        conv.ended_at = ended_at
        self._db.close_conversation(conv.conv_id, ended_at)
        self._open = None
        self._closing.append(conv)
        log.info("conversation %d ended", conv.conv_id)
        self._publish(
            {"type": "conversation_ended", "conv_id": conv.conv_id, "face_id": conv.face_id, "ts": ended_at}
        )
