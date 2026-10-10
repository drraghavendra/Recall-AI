import numpy as np

from recall.audio.types import Utterance
from recall.db import Database
from recall.matching.attribution import AttributionGate, Verdict
from recall.matching.voices import VoiceBook, voiceprint

rng = np.random.default_rng(0)
DIM = 256


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


ALICE, BOB, STRANGER = (unit(rng.normal(size=DIM)) for _ in range(3))


def said_by(person, wobble=0.25):
    """A new embedding of `person`'s voice: close to their voice, not identical."""
    return unit(person + wobble * unit(rng.normal(size=DIM)))


def book(db=None):
    return VoiceBook(db or Database(":memory:"), same_threshold=0.65, other_threshold=0.75, min_samples=3)


def teach(voices, face_id, person, n=3):
    for _ in range(n):
        voices.learn(face_id, said_by(person), 2.0)


def test_no_voiceprint_yet_keeps_the_focused_face():
    voices = book()
    assert voices.judge("alice", said_by(STRANGER), 2.0).face_id == "alice"


def test_the_focused_faces_own_voice_is_theirs():
    voices = book()
    teach(voices, "alice", ALICE)
    assert voices.judge("alice", said_by(ALICE), 2.0).face_id == "alice"


def test_an_unknown_voice_while_a_known_face_is_in_focus_is_background():
    voices = book()
    teach(voices, "alice", ALICE)
    assert voices.judge("alice", said_by(STRANGER), 2.0).background


def test_another_known_persons_voice_goes_to_them():
    voices = book()
    teach(voices, "alice", ALICE)
    teach(voices, "bob", BOB)
    assert voices.judge("alice", said_by(BOB), 2.0).face_id == "bob"
    assert voices.judge(None, said_by(BOB), 2.0).face_id == "bob"  # even with no face in view


def test_short_lines_get_no_opinion_and_teach_nothing():
    voices = book()
    teach(voices, "alice", ALICE)
    assert voices.judge("alice", said_by(STRANGER), 0.5).face_id == "alice"
    voices.learn("bob", said_by(BOB), 0.5)
    assert voices.judge("alice", said_by(BOB), 2.0).background


def test_a_background_line_learnt_early_does_not_skew_the_voiceprint():
    samples = [said_by(ALICE) for _ in range(4)] + [said_by(STRANGER)]
    assert float(voiceprint(samples) @ ALICE) > 0.9


def test_voiceprints_survive_a_restart(tmp_path):
    db = Database(str(tmp_path / "recall.db"))
    teach(book(db), "alice", ALICE)
    assert book(db).judge("alice", said_by(STRANGER), 2.0).background


def test_gate_rejects_unknown_voice_and_learns_accepted_ones():
    voices = book()
    gate = AttributionGate(None, None, focused_during=lambda s, e: "alice", voices=voices)
    for _ in range(3):
        assert gate(Utterance(0, 2, "hi", "other", voice=said_by(ALICE))) == Verdict(face_id="alice")
    assert gate(Utterance(0, 2, "hey", "other", voice=said_by(STRANGER))) == Verdict(reason="voice_unknown")
    assert gate(Utterance(0, 2, "me again", "other", voice=said_by(ALICE))) == Verdict(face_id="alice")


def test_a_voice_matching_someone_who_has_left_stays_with_the_face_in_view():
    # person_001's voiceprint learnt the wearer's voice while they were in focus; then the wearer turns
    # the camera to themselves (person_002). Their lines must not go to person_001, who isn't there.
    voices = book()
    teach(voices, "person_001", ALICE)
    assert voices.judge("person_002", said_by(ALICE), 2.0, in_view={"person_002"}).face_id == "person_002"
    # While both are in view, a clear match still goes to the right one.
    assert voices.judge("person_002", said_by(ALICE), 2.0, in_view={"person_001", "person_002"}).face_id == "person_001"


def wearer_gate(voices, in_view):
    return AttributionGate(
        None, None, focused_during=lambda s, e: "me", voices=voices, in_view=lambda: set(in_view), wearer=lambda: "me"
    )


def test_someone_elses_voice_while_only_the_user_is_in_view_is_background():
    gate = wearer_gate(book(), in_view={"me"})
    assert gate(Utterance(0, 2, "background chatter", "other", voice=said_by(STRANGER))) == Verdict(reason="not_in_view")


def test_with_the_user_in_focus_another_face_in_view_gets_the_line():
    voices = book()
    gate = wearer_gate(voices, in_view={"me", "bob"})
    assert gate(Utterance(0, 2, "hi", "other", voice=said_by(BOB))) == Verdict(face_id="bob")
    # Once bob's voice is known, a different voice is no longer his.
    teach(voices, "bob", BOB)
    assert gate(Utterance(0, 2, "hey", "other", voice=said_by(STRANGER))) == Verdict(reason="voice_unknown")


def test_the_users_own_words_go_to_their_card_when_marked():
    from recall.face.tracker import FocusedFaceTracker
    from recall.matching.matcher import ConversationMatcher

    db = Database(":memory:")
    db.set_wearer_face("me")
    tracker = FocusedFaceTracker(confirm_windows=1, absent_confirm_windows=1)
    tracker.observe(0.0, "me")
    matcher = ConversationMatcher(db, tracker, 1.0, 5.0, lambda _e: None, lambda _c: None)
    matcher.on_utterance(Utterance(0.5, 2.5, "I'm a teacher", "self"))
    assert [segment.text for segment in db.pending_lines()["me"]] == ["I'm a teacher"]


def test_marking_the_users_face_credits_their_earlier_lines():
    db = Database(":memory:")
    conv = db.open_conversation("me", 0.0)
    db.insert_segment(conv, 0, 1, "I'm a teacher", "self", None)
    db.insert_segment(conv, 1, 2, "nice", "other", "bob")
    assert db.credit_self_lines("me") == 1
    assert [s.text for s in db.pending_lines()["me"]] == ["I'm a teacher"]
