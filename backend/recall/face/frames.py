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


def portrait_candidate(jpeg: bytes, box: tuple[float, float, float, float] | None) -> tuple[bytes, float] | None:
    """A square, face-centred crop for the person's profile photo, and a quality score (higher is better).

    The score favours a large face (close to the camera), a sharp crop and good exposure; too-dark crops
    (e.g. a camera still adjusting its exposure) return None. Without a face box, the whole frame is used.
    """
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    face_fraction = 0.05
    if box is not None:
        x, y, w, h = box[0] * width, box[1] * height, box[2] * width, box[3] * height
        # Head and a little shoulder, not the whole room; slide the square to stay inside the frame.
        side = int(min(max(w, h) * 1.7, width, height))
        if side < 24:
            return None
        cx, cy = x + w / 2, y + h / 2 - 0.08 * h
        x0 = int(min(max(0, cx - side / 2), width - side))
        y0 = int(min(max(0, cy - side / 2), height - side))
        image = image[y0 : y0 + side, x0 : x0 + side]
        face_fraction = box[2] * box[3]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    exposure = min(1.0, max(0.0, (brightness - 45) / 55)) * min(1.0, max(0.0, (235 - brightness) / 40))
    if exposure < 0.3:
        return None
    sharpness = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 120)
    side = 320
    crop = cv2.resize(image, (side, side) if box is not None else (side * width // height, side), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    score = face_fraction * (0.3 + 0.7 * sharpness) * exposure
    return encoded.tobytes(), score
