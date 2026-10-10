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
