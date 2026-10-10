import cv2
import numpy as np


def _sharpness(jpeg: bytes) -> float | None:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def select_sharpest(jpegs: list[bytes], k: int) -> list[bytes]:
    """Pick the k least-blurry decodable frames, so only those are sent to the paid recognizer."""
    scored = [(score, jpeg) for jpeg in jpegs if (score := _sharpness(jpeg)) is not None]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [jpeg for _, jpeg in scored[:k]]
