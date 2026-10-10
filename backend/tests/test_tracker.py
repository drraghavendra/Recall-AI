from recall.face.tracker import FocusedFaceTracker


def test_new_face_needs_consecutive_confirmations():
    tracker = FocusedFaceTracker(confirm_windows=2, absent_confirm_windows=3)
    assert not tracker.observe(0.0, "alice")
    assert tracker.current is None
    assert tracker.observe(2.0, "alice")
    assert tracker.current == "alice"


def test_single_misread_does_not_flip_the_face():
    tracker = FocusedFaceTracker(confirm_windows=2, absent_confirm_windows=3)
    tracker.observe(0.0, "alice")
    tracker.observe(2.0, "alice")
    tracker.observe(4.0, "bob")
    tracker.observe(6.0, "alice")
    tracker.observe(8.0, "bob")
    assert tracker.current == "alice"


def test_absence_needs_its_own_longer_confirmation():
    tracker = FocusedFaceTracker(confirm_windows=1, absent_confirm_windows=3)
    tracker.observe(0.0, "alice")
    tracker.observe(2.0, None)
    tracker.observe(4.0, None)
    assert tracker.current == "alice"
    assert tracker.observe(6.0, None)
    assert tracker.current is None


def test_timeline_is_stamped_when_the_new_face_was_first_seen():
    tracker = FocusedFaceTracker(confirm_windows=2, absent_confirm_windows=3)
    tracker.observe(0.0, "alice")
    tracker.observe(2.0, "alice")
    tracker.observe(10.0, "bob")
    tracker.observe(12.0, "bob")
    assert tracker.face_at(-1.0) is None
    assert tracker.face_at(5.0) == "alice"
    assert tracker.face_at(10.0) == "bob"
    assert tracker.change_at(11.0) == (10.0, "bob")
