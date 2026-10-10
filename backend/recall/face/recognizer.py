from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    face_found: bool
    face_id: str | None
    similarity: float


class FaceRecognizer(Protocol):
    """Matches/enrolls the largest face in an image. Methods block; call them off the event loop."""

    def search(self, image: bytes) -> SearchResult: ...

    def enroll(self, image: bytes, face_id: str) -> str | None:
        """Index the face under `face_id`. Returns the provider's face id, or None if no usable face."""
        ...
