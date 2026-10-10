from recall.db import Database


def test_face_ids_are_sequential_and_portrait_round_trips_in_local_database():
    db = Database(":memory:")
    try:
        assert db.next_face_id() == "person_001"
        db.insert_face("person_001", "provider-id-1")
        assert db.next_face_id() == "person_002"

        db.save_face_portrait("person_001", b"first-jpeg", score=0.1)
        db.save_face_portrait("person_001", b"better-jpeg", score=0.3)
        assert db.get_face_portrait("person_001") == b"better-jpeg"
        assert db.portrait_info("person_001")[0] == 0.3
        assert db.delete_face_portrait("person_001") is True
        assert db.get_face_portrait("person_001") is None
        assert db.delete_face_portrait("person_001") is False
    finally:
        db.close()


def test_portrait_is_a_face_crop_and_rejects_dark_frames():
    import cv2
    import numpy as np

    from recall.face.frames import portrait_candidate

    frame = np.full((720, 1280, 3), 140, np.uint8)
    cv2.circle(frame, (640, 360), 120, (90, 120, 200), -1)
    cv2.putText(frame, "edges", (560, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (10, 10, 10), 3)
    jpeg = cv2.imencode(".jpg", frame)[1].tobytes()
    near = portrait_candidate(jpeg, (0.45, 0.3, 0.1, 0.35))
    far = portrait_candidate(jpeg, (0.48, 0.45, 0.03, 0.06))
    edge = portrait_candidate(jpeg, (0.0, 0.0, 0.1, 0.35))

    crop = cv2.imdecode(np.frombuffer(near[0], np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape[:2] == (320, 320)
    assert cv2.imdecode(np.frombuffer(edge[0], np.uint8), cv2.IMREAD_COLOR).shape[:2] == (320, 320)
    assert near[1] > far[1]
    dark = cv2.imencode(".jpg", np.full((720, 1280, 3), 12, np.uint8))[1].tobytes()
    assert portrait_candidate(dark, (0.45, 0.3, 0.1, 0.35)) is None

