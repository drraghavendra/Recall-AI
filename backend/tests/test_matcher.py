import pytest

from recall.audio.types import Utterance
from recall.db import Database
from recall.face.tracker import FocusedFaceTracker
from recall.matching.matcher import ConversationMatcher


@pytest.fixture
def env():
    db = Database(":memory:")
    tracker = FocusedFaceTracker(confirm_windows=1, absent_confirm_windows=1)
    events: list[dict] = []
    finalized: list = []
    matcher = ConversationMatcher(db, tracker, 1.0, 5.0, events.append, finalized.append)
    yield db, tracker, matcher, events, finalized
    db.close()


def utterance(start, end, speaker="other", text="hello"):
    return Utterance(start, end, text, speaker)


def segment_rows(db):
    rows = db._conn.execute("SELECT conv_id, speaker, face_id FROM transcript_segments ORDER BY id")
    return [tuple(row) for row in rows]


def test_short_utterance_with_no_conversation_is_noise(env):
    db, _, matcher, events, _ = env
    matcher.on_utterance(utterance(0.0, 0.5))
    matcher.on_audio_time(10.0, in_speech=False)
    assert matcher.open_conversation is None
    assert events == []
    assert segment_rows(db) == []


def test_silent_face_in_frame_opens_nothing(env):
    _, tracker, matcher, events, _ = env
    tracker.observe(0.0, "alice")
    matcher.on_audio_time(30.0, in_speech=False)
    assert events == []


def test_conversation_is_labeled_by_face_at_onset_and_closed_by_silence(env):
    _, tracker, matcher, events, finalized = env
    tracker.observe(0.0, "alice")
    matcher.on_utterance(utterance(1.0, 3.0))
    conv = matcher.open_conversation
    assert (conv.face_id, conv.started_at) == ("alice", 1.0)

    matcher.on_audio_time(5.0, in_speech=False)
    assert matcher.open_conversation is conv

    matcher.on_audio_time(8.5, in_speech=False)
    assert matcher.open_conversation is None
    assert [e["type"] for e in events] == ["conversation_started", "conversation_ended"]
    assert events[1]["ts"] == 3.0
    assert finalized == [conv]


def test_ongoing_speech_keeps_conversation_open(env):
    _, tracker, matcher, _, _ = env
    tracker.observe(0.0, "alice")
    matcher.on_utterance(utterance(1.0, 3.0))
    matcher.on_audio_time(20.0, in_speech=True)
    assert matcher.open_conversation is not None


def test_own_speech_is_never_attributed_to_the_other_person(env):
    db, tracker, matcher, _, _ = env
    tracker.observe(0.0, "alice")
    matcher.on_utterance(utterance(1.0, 3.0, "other"))
    matcher.on_utterance(utterance(3.5, 5.0, "self"))
    conv_id = matcher.open_conversation.conv_id
    assert segment_rows(db) == [(conv_id, "other", "alice"), (conv_id, "self", None)]


def test_face_leaving_frame_keeps_the_conversation(env):
    _, tracker, matcher, _, _ = env
    tracker.observe(0.0, "alice")
    matcher.on_utterance(utterance(1.0, 3.0))
    conv = matcher.open_conversation
    tracker.observe(3.5, None)
    matcher.on_utterance(utterance(4.0, 6.0))
    assert matcher.open_conversation is conv


def test_different_face_hands_off_to_a_new_conversation(env):
    db, tracker, matcher, _, finalized = env
    tracker.observe(0.0, "alice")
    matcher.on_utterance(utterance(1.0, 3.0))
    first = matcher.open_conversation

    tracker.observe(4.0, "bob")
    matcher.on_utterance(utterance(5.0, 7.0))
    second = matcher.open_conversation
    assert second is not first
    assert (second.face_id, second.started_at, first.ended_at) == ("bob", 4.0, 4.0)
    assert segment_rows(db) == [(first.conv_id, "other", "alice"), (second.conv_id, "other", "bob")]

    matcher.on_audio_time(7.5, in_speech=False)
    assert finalized == [first]


def test_self_session_is_never_sent_for_persona(env):
    _, _, matcher, events, finalized = env
    matcher.on_utterance(utterance(0.0, 2.0, "self"))
    matcher.on_audio_time(10.0, in_speech=False)
    assert events[0]["face_id"] is None
    assert finalized == []
