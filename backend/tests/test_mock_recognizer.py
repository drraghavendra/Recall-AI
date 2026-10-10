from recall.face.mock import MockRecognizer


def test_reports_only_the_configured_ids_and_always_matches():
    recognizer = MockRecognizer(face_count=3, hold_windows=1, seed=7)
    results = [recognizer.search(b"") for _ in range(200)]
    assert all(r.face_found and r.similarity > 0 for r in results)
    assert {r.face_id for r in results} == {"1", "2", "3"}


def test_holds_the_same_person_for_hold_windows_searches():
    recognizer = MockRecognizer(face_count=3, hold_windows=4, seed=1)
    ids = [recognizer.search(b"").face_id for _ in range(12)]
    assert ids[0:4] == [ids[0]] * 4
    assert ids[4:8] == [ids[4]] * 4
    assert ids[8:12] == [ids[8]] * 4
