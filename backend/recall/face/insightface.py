"""Local InsightFace implementation of the face-recognizer provider interface."""

import sqlite3
import time
from contextlib import closing
from pathlib import Path
from threading import RLock

import numpy as np

from recall.face.recognizer import SearchResult


class InsightFaceRecognizer:
    """Detect the largest face and match normalized embeddings in a local SQLite collection."""

    def __init__(
        self,
        db_path: str,
        match_threshold: float = 0.55,
        new_face_threshold: float = 0.35,
        min_det_score: float = 0.6,
        face_app=None,
    ):
        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError("InsightFace match threshold must be between 0 and 1")
        self._db_path = db_path
        self._match_threshold = match_threshold
        self._new_face_threshold = min(new_face_threshold, match_threshold)
        self._min_det_score = min_det_score
        self._lock = RLock()
        if face_app is None:
            try:
                from insightface.app import FaceAnalysis
            except ImportError as exc:
                raise RuntimeError(
                    "InsightFace is not installed. Install the backend's [local-face] extra."
                ) from exc
            face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            face_app.prepare(ctx_id=-1, det_size=(640, 640))
        self._face_app = face_app
        if db_path != ":memory:":
            Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS face_embeddings ("
                    "face_id TEXT PRIMARY KEY, embedding BLOB NOT NULL, enrolled_at REAL NOT NULL)"
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _embedding(self, image_bytes: bytes) -> np.ndarray | None:
        if not image_bytes:
            return None
        import cv2

        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        faces = self._face_app.get(image)
        if not faces:
            return None
        # Match the existing focused-face behavior: use the largest detected face.
        face = max(
            faces,
            key=lambda item: max(0.0, float(item.bbox[2] - item.bbox[0]))
            * max(0.0, float(item.bbox[3] - item.bbox[1])),
        )
        # A hand over the face or a blurry detection gives a junk embedding; better to see "no face" than guess.
        if float(getattr(face, "det_score", 1.0)) < self._min_det_score:
            return None
        embedding = np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm == 0:
            return None
        return embedding / norm

    @staticmethod
    def _similarity(left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != right.shape:
            return -1.0
        return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))

    def search(self, image: bytes) -> SearchResult:
        embedding = self._embedding(image)
        if embedding is None:
            return SearchResult(face_found=False, face_id=None, similarity=0.0)
        best_id: str | None = None
        best_score = -1.0
        with self._lock, closing(self._connect()) as conn:
            for face_id, raw in conn.execute("SELECT face_id, embedding FROM face_embeddings"):
                saved = np.frombuffer(raw, dtype=np.float32)
                score = self._similarity(embedding, saved)
                if score > best_score:
                    best_id, best_score = face_id, score
        if best_id is not None and best_score >= self._match_threshold:
            return SearchResult(face_found=True, face_id=best_id, similarity=best_score)
        if best_id is not None and best_score >= self._new_face_threshold:
            # Between "same person" and "stranger": likely the known person partly occluded or turned away.
            # Report no usable face so the caller neither switches identity nor enrolls a duplicate.
            return SearchResult(face_found=False, face_id=None, similarity=best_score)
        return SearchResult(face_found=True, face_id=None, similarity=max(0.0, best_score))

    def enroll(self, image: bytes, face_id: str) -> str | None:
        embedding = self._embedding(image)
        if embedding is None:
            return None
        with self._lock, closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    "INSERT INTO face_embeddings(face_id, embedding, enrolled_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(face_id) DO UPDATE SET embedding=excluded.embedding, enrolled_at=excluded.enrolled_at",
                    (face_id, embedding.tobytes(), time.time()),
                )
        return face_id
