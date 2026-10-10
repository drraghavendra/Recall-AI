from recall.db import Database


def test_face_ids_are_sequential_and_portrait_round_trips_in_local_database():
    db = Database(":memory:")
    try:
        assert db.next_face_id() == "person_001"
        db.insert_face("person_001", "provider-id-1")
        assert db.next_face_id() == "person_002"

        db.save_face_portrait("person_001", b"first-jpeg")
        db.save_face_portrait("person_001", b"replacement-is-ignored")
        assert db.get_face_portrait("person_001") == b"first-jpeg"
        assert db.delete_face_portrait("person_001") is True
        assert db.get_face_portrait("person_001") is None
        assert db.delete_face_portrait("person_001") is False
    finally:
        db.close()
