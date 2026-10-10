import sys
from types import SimpleNamespace

import numpy as np

from recall.face.insightface import InsightFaceRecognizer


class FakeFaceApp:
    def __init__(self, embedding):
        self.embedding = embedding

    def get(self, _image):
        if self.embedding is None:
            return []
        return [SimpleNamespace(bbox=np.array([0, 0, 10, 10]), normed_embedding=self.embedding)]


def fake_cv2(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(imdecode=lambda *_args: object(), IMREAD_COLOR=1),
    )


def test_insightface_recognizer_enrolls_and_reuses_saved_identity(tmp_path, monkeypatch):
    fake_cv2(monkeypatch)
    image = b"jpeg"
    recognizer = InsightFaceRecognizer(str(tmp_path / "faces.sqlite3"), face_app=FakeFaceApp([1.0, 0.0]))

    assert recognizer.enroll(image, "person_001") == "person_001"
    result = recognizer.search(image)

    assert result.face_found is True
    assert result.face_id == "person_001"
    assert result.similarity == 1.0


def test_insightface_recognizer_does_not_match_below_threshold(tmp_path, monkeypatch):
    fake_cv2(monkeypatch)
    recognizer = InsightFaceRecognizer(
        str(tmp_path / "faces.sqlite3"), match_threshold=0.8, face_app=FakeFaceApp([0.0, 1.0])
    )
    recognizer.enroll(b"jpeg", "person_001")
    recognizer._face_app = FakeFaceApp([1.0, 0.0])

    result = recognizer.search(b"jpeg")

    assert result.face_found is True
    assert result.face_id is None


def test_insightface_recognizer_reports_no_face(tmp_path, monkeypatch):
    fake_cv2(monkeypatch)
    recognizer = InsightFaceRecognizer(str(tmp_path / "faces.sqlite3"), face_app=FakeFaceApp(None))

    result = recognizer.search(b"jpeg")

    assert result.face_found is False
    assert result.face_id is None
