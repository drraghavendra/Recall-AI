import numpy as np
import pytest

from recall.audio.level import speech_level_dbfs
from recall.audio.types import Utterance
from recall.db import Database
from recall.face.lips import LipActivityTracker, mouth_openness
from recall.face.tracker import FocusedFaceTracker
from recall.matching.attribution import AttributionGate, Verdict
from recall.matching.matcher import ConversationMatcher

RATE = 16000


def tone(amplitude: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_level_ignores_leading_and_trailing_silence():
    padded = np.concatenate([np.zeros(RATE // 2, np.float32), tone(0.1), np.zeros(RATE // 2, np.float32)])
    assert speech_level_dbfs(padded, RATE) == pytest.approx(speech_level_dbfs(tone(0.1), RATE), abs=1.0)
    assert speech_level_dbfs(tone(0.1), RATE) - speech_level_dbfs(tone(0.01), RATE) == pytest.approx(20, abs=0.5)


def face_with_mouth_open(gap: float) -> np.ndarray:
    landmarks = np.zeros((68, 3), np.float32)
    landmarks[48, :2], landmarks[54, :2] = (0, 0), (40, 0)  # 40px wide mouth
    for top, bottom in ((61, 67), (62, 66), (63, 65)):
        landmarks[top, :2], landmarks[bottom, :2] = (20, -gap / 2), (20, gap / 2)
    return landmarks


def lips() -> LipActivityTracker:
    return LipActivityTracker(min_openness_std=0.03)


def feed(tracker, start, gaps, face_id="alice"):
    for i, gap in enumerate(gaps):
        tracker.observe(face_id, start + i * 0.2, mouth_openness(face_with_mouth_open(gap)))


def test_openness_is_inner_lip_gap_over_mouth_width():
    assert mouth_openness(face_with_mouth_open(10)) == pytest.approx(0.25)


def test_lip_check_sees_a_mouth_opening_and_closing_as_talking():
    tracker = lips()
    feed(tracker, 0.0, [2, 12, 4, 14, 1, 10, 3, 12])
    assert tracker.talking_during(0.2, 1.4) is True


def test_lip_check_sees_a_steady_mouth_as_not_talking():
    tracker = lips()
    feed(tracker, 0.0, [2, 2.5, 2, 1.8, 2.2, 2, 2.4, 2])
    assert tracker.talking_during(0.2, 1.4) is False


def test_lip_check_abstains_without_enough_frames():
    tracker = lips()
    feed(tracker, 0.0, [2, 12, 4, 14])
    assert tracker.talking_during(30.0, 32.0) is None


def test_with_two_faces_the_line_goes_to_the_one_whose_mouth_moved():
    tracker = lips()
    feed(tracker, 0.0, [2, 2.2, 2, 2.1, 2, 2.3, 2, 2.1], face_id="alice")  # listening
    feed(tracker, 0.0, [2, 12, 4, 14, 1, 10, 3, 12], face_id="bob")  # talking
    assert tracker.speaker_during(0.2, 1.4) == ("bob", True)
    gate = AttributionGate(min_level_dbfs=None, lips=tracker, lips_pick_speaker=True)
    assert gate(Utterance(0.2, 1.4, "hi", "other")) == Verdict(face_id="bob")


def test_by_default_the_centre_face_must_be_the_one_talking():
    tracker = lips()
    feed(tracker, 0.0, [2, 2.2, 2, 2.1, 2, 2.3, 2, 2.1], face_id="alice")  # centre face, listening
    feed(tracker, 0.0, [2, 12, 4, 14, 1, 10, 3, 12], face_id="bob")  # someone off to the side, talking
    gate = AttributionGate(None, tracker, lips_pick_speaker=False, focused_during=lambda s, e: "alice")
    assert gate(Utterance(0.2, 1.4, "hi", "other")) == Verdict(reason="lips_still")

    gate = AttributionGate(None, tracker, lips_pick_speaker=False, focused_during=lambda s, e: "bob")
    assert gate(Utterance(0.2, 1.4, "hi", "other")) == Verdict(face_id="bob")

    gate = AttributionGate(None, tracker, lips_pick_speaker=False, focused_during=lambda s, e: "carol")
    assert gate(Utterance(0.2, 1.4, "hi", "other")) == Verdict(face_id="carol")  # no lip frames: keep


def test_lip_frame_faces_are_matched_to_recognised_people_by_position():
    from recall.face.lips import _match_faces

    detected = [((0.60, 0.30, 0.10, 0.20), "right-face"), ((0.10, 0.30, 0.10, 0.20), "left-face")]
    known = {"alice": (0.11, 0.31, 0.10, 0.20), "bob": (0.59, 0.29, 0.10, 0.20), "carol": (0.9, 0.9, 0.05, 0.05)}
    assert sorted(_match_faces(detected, known)) == [("alice", "left-face"), ("bob", "right-face")]


def test_gate_rejects_quiet_speech_and_allows_unknown_levels():
    gate = AttributionGate(min_level_dbfs=-42.0, lips=None)
    assert gate(Utterance(0, 2, "hi", "other", level_dbfs=-55.0)) == Verdict(reason="too_quiet")
    assert gate(Utterance(0, 2, "hi", "other", level_dbfs=-30.0)) == Verdict()
    assert gate(Utterance(0, 2, "hi", "other")) == Verdict()


@pytest.fixture
def env():
    db = Database(":memory:")
    tracker = FocusedFaceTracker(confirm_windows=1, absent_confirm_windows=1)
    events: list[dict] = []
    finalized: list = []
    gate = AttributionGate(min_level_dbfs=-42.0, lips=None)
    matcher = ConversationMatcher(
        db, tracker, 1.0, 5.0, events.append, finalized.append, gate=gate, max_utterance_s=15.0
    )
    yield db, tracker, matcher, events, finalized
    db.close()


def test_background_speech_never_opens_a_conversation_or_reaches_the_db(env):
    db, tracker, matcher, events, _ = env
    tracker.observe(0.0, "alice")
    matcher.on_utterance(Utterance(1.0, 4.0, "someone across the room", "other", level_dbfs=-55.0))
    assert matcher.open_conversation is None
    assert [(e["type"], e["speaker"], e["face_id"]) for e in events] == [("transcript", "background", None)]
    assert db._conn.execute("SELECT COUNT(*) FROM transcript_segments").fetchone()[0] == 0


def test_continuous_background_chatter_does_not_hold_a_conversation_open(env):
    _, tracker, matcher, _, finalized = env
    tracker.observe(0.0, "alice")
    matcher.on_utterance(Utterance(1.0, 3.0, "hello", "other", level_dbfs=-30.0))
    conv = matcher.open_conversation

    matcher.on_audio_time(10.0, in_speech=True)  # chatter, but it could still be alice's long utterance
    assert matcher.open_conversation is conv
    matcher.on_audio_time(23.0, in_speech=True)  # 3s + 5s silence + 15s max utterance: can't be hers
    assert matcher.open_conversation is None
    matcher.on_audio_time(40.0, in_speech=True)
    assert finalized == [conv]
