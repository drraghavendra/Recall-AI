import numpy as np


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """Signed 16-bit little-endian PCM -> float32 in [-1, 1]."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
