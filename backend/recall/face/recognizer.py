from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    face_found: bool
    face_id: str | None
    similarity: float


# (x, y, width, height) as fractions of the image size
FaceBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class FaceHit:
    result: SearchResult
    box: FaceBox


class FaceRecognizer(Protocol):
    """Matches/enrolls the largest face in an image. Methods block; call them off the event loop."""

    def search(self, image: bytes) -> SearchResult: ...

    def enroll(self, image: bytes, face_id: str) -> str | None:
        """Index the face under `face_id`. Returns the provider's face id, or None if no usable face."""
        ...


class MultiFaceRecognizer(FaceRecognizer, Protocol):
    """Optional: providers that can match every face in the frame, not just the largest."""

    def search_all(self, image: bytes) -> list[FaceHit]:
        """Every detected face, largest (= focused) first."""
        ...
