"""Local InsightFace implementation of the face-recognizer provider interface."""

import sqlite3
import time
from contextlib import closing
from pathlib import Path
from threading import RLock

import numpy as np

from recall.face.recognizer import FaceBox, FaceHit, SearchResult


def focused_first(faces: list, focus: str) -> list:
    """Move the focused face to the front of `faces` (each item starts with its box, largest face first).

    "center": the face nearest the middle of the frame, among faces at least half as tall as the
    largest — on glasses you look at the person you're talking to, and a small face in the background
    can't win just by being central. "largest": the biggest face.
    """
    if focus != "center" or len(faces) < 2:
        return faces
    tallest = max(item[0][3] for item in faces)
    candidates = [i for i, item in enumerate(faces) if item[0][3] >= 0.5 * tallest]

    def distance(i: int) -> float:
        x, y, w, h = faces[i][0]
        return ((x + w / 2 - 0.5) ** 2 + (y + h / 2 - 0.5) ** 2) ** 0.5

    chosen = min(candidates, key=distance)
    return [faces[chosen]] + [item for i, item in enumerate(faces) if i != chosen]


class InsightFaceRecognizer:
    """Detect faces and match normalized embeddings against a local SQLite gallery.

    Each person keeps up to `gallery_size` embeddings: the enrolment one plus confident-but-imperfect
    matches (a new angle or light), so later sightings from those angles still match. Faces smaller
    than `min_enroll_face_height` (fraction of frame height) are matched but never enrolled: distant
    background faces gave poor embeddings and created stray people.
    """

    def __init__(
        self,
        db_path: str,
        match_threshold: float = 0.55,
        new_face_threshold: float = 0.35,
        min_det_score: float = 0.6,
        face_app=None,
        min_enroll_face_height: float = 0.15,
        gallery_size: int = 5,
        gallery_add_below: float = 0.72,
        focus: str = "center",
    ):
        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError("InsightFace match threshold must be between 0 and 1")
        self._db_path = db_path
        self._match_threshold = match_threshold
        self._new_face_threshold = min(new_face_threshold, match_threshold)
        self._min_det_score = min_det_score
        self._min_enroll_face_height = min_enroll_face_height
        self._gallery_size = gallery_size
        self._gallery_add_below = gallery_add_below
        self._focus = focus
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
                    "CREATE TABLE IF NOT EXISTS face_gallery ("
                    "face_id TEXT NOT NULL, embedding BLOB NOT NULL, added_at REAL NOT NULL)"
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_gallery_face ON face_gallery(face_id)")
                # Earlier builds kept one embedding per person in face_embeddings.
                old = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'face_embeddings'"
                ).fetchone()
                if old:
                    conn.execute(
                        "INSERT INTO face_gallery (face_id, embedding, added_at) "
                        "SELECT face_id, embedding, enrolled_at FROM face_embeddings"
                    )
                    conn.execute("DROP TABLE face_embeddings")

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
        """Every detected face, focused face first, with its normalized embedding (None if unusable)."""
        image = self._decode(image_bytes)
        if image is None:
            return []
        height, width = getattr(image, "shape", (1, 1))[:2]
        found = []
        for face in sorted(self._face_app.get(image), key=lambda face: self._area(face.bbox), reverse=True):
            x0, y0, x1, y1 = (float(v) for v in face.bbox[:4])
            box = (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height)
            found.append((box, self._normalized_embedding(face)))
        return focused_first(found, self._focus)

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
            rows = conn.execute("SELECT face_id, embedding FROM face_gallery").fetchall()
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

    def _add_to_gallery(self, face_id: str, embedding: np.ndarray) -> None:
        with self._lock, closing(self._connect()) as conn:
            with conn:
                count = conn.execute("SELECT COUNT(*) FROM face_gallery WHERE face_id = ?", (face_id,)).fetchone()[0]
                if count < self._gallery_size:
                    conn.execute(
                        "INSERT INTO face_gallery (face_id, embedding, added_at) VALUES (?, ?, ?)",
                        (face_id, embedding.tobytes(), time.time()),
                    )

    def _learn(self, result: SearchResult, box: FaceBox, embedding: np.ndarray | None) -> None:
        """A clear, confident match that still looks somewhat different: keep it as another view of them."""
        if (
            embedding is not None
            and result.face_id
            and self._match_threshold <= result.similarity < self._gallery_add_below
            and box[3] >= self._min_enroll_face_height
        ):
            self._add_to_gallery(result.face_id, embedding)

    def search(self, image: bytes) -> SearchResult:
        hits = self.search_all(image)
        return hits[0].result if hits else SearchResult(face_found=False, face_id=None, similarity=0.0)

    def search_all(self, image: bytes) -> list[FaceHit]:
        faces = self._faces(image)
        saved = self._saved_embeddings() if faces else []
        hits = []
        for box, embedding in faces:
            result = self._match(embedding, saved)
            self._learn(result, box, embedding)
            hits.append(FaceHit(result, box))
        return hits

    def mouth_landmarks_all(self, image_bytes: bytes, min_height: float = 0.06) -> list[tuple[FaceBox, np.ndarray]]:
        """For lip frames: every usable face's box and 68 landmarks (detection + landmarks only, no recognition)."""
        image = self._decode(image_bytes)
        if image is None:
            return []
        height, width = image.shape[:2]
        bboxes, kpss = self._face_app.det_model.detect(image, max_num=0, metric="default")
        if bboxes is None:
            return []
        from insightface.app.common import Face

        found = []
        for i, bbox in enumerate(bboxes):
            x0, y0, x1, y1, score = (float(v) for v in bbox[:5])
            box = (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height)
            if score < self._min_det_score or box[3] < min_height:
                continue
            face = Face(bbox=bbox[:4], kps=None if kpss is None else kpss[i], det_score=score)
            self._face_app.models["landmark_3d_68"].get(image, face)
            found.append((box, face.landmark_3d_68))
        return found

    def enroll(self, image: bytes, face_id: str) -> str | None:
        faces = self._faces(image)
        if not faces:
            return None
        box, embedding = faces[0]
        if embedding is None or box[3] < self._min_enroll_face_height:
            return None
        with self._lock, closing(self._connect()) as conn:
            with conn:
                conn.execute("DELETE FROM face_gallery WHERE face_id = ?", (face_id,))
        self._add_to_gallery(face_id, embedding)
        return face_id
