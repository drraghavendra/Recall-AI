import random

from recall.face.recognizer import SearchResult


class MockRecognizer:
    """Pretends every frame shows one of `face_count` known people ("1", "2", ...), no AWS needed.

    Sticks with a randomly chosen person for `hold_windows` searches before picking again, so
    conversations last long enough to test personas; `hold_windows=1` re-rolls every frame.
    """

    def __init__(self, face_count: int, hold_windows: int, seed: int | None = None):
        if face_count < 1 or hold_windows < 1:
            raise ValueError("face_count and hold_windows must be >= 1")
        self._face_ids = [str(i) for i in range(1, face_count + 1)]
        self._hold_windows = hold_windows
        self._rng = random.Random(seed)
        self._current = self._rng.choice(self._face_ids)
        self._remaining = hold_windows

    def search(self, image: bytes) -> SearchResult:
        if self._remaining == 0:
            self._current = self._rng.choice(self._face_ids)
            self._remaining = self._hold_windows
        self._remaining -= 1
        return SearchResult(face_found=True, face_id=self._current, similarity=99.0)

    def enroll(self, image: bytes, face_id: str) -> str | None:
        # Unreachable in practice: search always matches one of the fixed ids.
        return None
