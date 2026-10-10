"""Local InsightFace implementation of the face-recognizer provider interface."""

import sqlite3
import time
from contextlib import closing
from pathlib import Path
from threading import RLock

import numpy as np

from recall.face.recognizer import FaceBox, FaceHit, SearchResult


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

    @staticmethod
    def _decode(image_bytes: bytes):
        if not image_bytes:
            return None
        import cv2

        return cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    @staticmethod
    def _area(bbox) -> float:
        return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))

    def _faces(self, image_bytes: bytes) -> list[tuple[FaceBox, np.ndarray | None]]:
        """Every detected face, largest first, with its normalized embedding (None if unusable)."""
        image = self._decode(image_bytes)
        if image is None:
            return []
        height, width = getattr(image, "shape", (1, 1))[:2]
        faces = sorted(self._face_app.get(image), key=lambda face: self._area(face.bbox), reverse=True)
        found = []
        for face in faces:
            x0, y0, x1, y1 = (float(v) for v in face.bbox[:4])
            box = (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height)
            found.append((box, self._normalized_embedding(face)))
        return found

    def _normalized_embedding(self, face) -> np.ndarray | None:
        # A hand over the face or a blurry detection gives a junk embedding; better to see "no face" than guess.
        if float(getattr(face, "det_score", 1.0)) < self._min_det_score:
            return None
        embedding = np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm == 0:
            return None
        return embedding / norm

    def _embedding(self, image_bytes: bytes) -> np.ndarray | None:
        # Match the existing focused-face behavior: use the largest detected face.
        faces = self._faces(image_bytes)
        return faces[0][1] if faces else None

    @staticmethod
    def _similarity(left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != right.shape:
            return -1.0
        return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))

    def _saved_embeddings(self) -> list[tuple[str, np.ndarray]]:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute("SELECT face_id, embedding FROM face_embeddings").fetchall()
        return [(face_id, np.frombuffer(raw, dtype=np.float32)) for face_id, raw in rows]

    def _match(self, embedding: np.ndarray | None, saved: list[tuple[str, np.ndarray]]) -> SearchResult:
        if embedding is None:
            return SearchResult(face_found=False, face_id=None, similarity=0.0)
        best_id: str | None = None
        best_score = -1.0
        for face_id, vector in saved:
            score = self._similarity(embedding, vector)
            if score > best_score:
                best_id, best_score = face_id, score
        if best_id is not None and best_score >= self._match_threshold:
            return SearchResult(face_found=True, face_id=best_id, similarity=best_score)
        if best_id is not None and best_score >= self._new_face_threshold:
            # Between "same person" and "stranger": likely the known person partly occluded or turned away.
            # Report no usable face so the caller neither switches identity nor enrolls a duplicate.
            return SearchResult(face_found=False, face_id=None, similarity=best_score)
        return SearchResult(face_found=True, face_id=None, similarity=max(0.0, best_score))

    def search(self, image: bytes) -> SearchResult:
        return self._match(self._embedding(image), self._saved_embeddings())

    def search_all(self, image: bytes) -> list[FaceHit]:
        faces = self._faces(image)
        saved = self._saved_embeddings() if faces else []
        return [FaceHit(self._match(embedding, saved), box) for box, embedding in faces]

    def mouth_landmarks(
        self, image_bytes: bytes, near: tuple[float, float] | None = None
    ) -> tuple[np.ndarray, tuple[float, float]] | None:
        """For lip frames: detection plus the 68-point landmark model only (no recognition).

        Picks the largest face, except that among faces nearly as large it sticks with the one closest
        to `near` (the previous frame's face centre), so two similar-sized faces don't alternate.
        Returns the landmarks and the chosen face's centre.
        """
        image = self._decode(image_bytes)
        if image is None:
            return None
        bboxes, kpss = self._face_app.det_model.detect(image, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return None
        areas = [self._area(box) for box in bboxes]
        candidates = [i for i in range(len(bboxes)) if areas[i] >= 0.7 * max(areas)]
        centre = lambda i: (float(bboxes[i][0] + bboxes[i][2]) / 2, float(bboxes[i][1] + bboxes[i][3]) / 2)
        if near is None:
            largest = max(candidates, key=lambda i: areas[i])
        else:
            largest = min(candidates, key=lambda i: (centre(i)[0] - near[0]) ** 2 + (centre(i)[1] - near[1]) ** 2)
        if float(bboxes[largest][4]) < self._min_det_score:
            return None
        from insightface.app.common import Face

        face = Face(bbox=bboxes[largest][:4], kps=None if kpss is None else kpss[largest], det_score=bboxes[largest][4])
        self._face_app.models["landmark_3d_68"].get(image, face)
        return face.landmark_3d_68, centre(largest)

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
