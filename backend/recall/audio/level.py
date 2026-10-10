import numpy as np


def speech_level_dbfs(audio: np.ndarray, sample_rate: int, frame_s: float = 0.03) -> float:
    """Loudness of the voiced part of a clip, in dBFS: mean RMS of its loudest half of 30 ms frames.

    Using only the loudest frames keeps the VAD's leading/trailing padding from dragging the level down.
    """
    n = max(1, int(sample_rate * frame_s))
    usable = len(audio) // n * n
    frames = audio[:usable].reshape(-1, n) if usable else audio.reshape(1, -1)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    loudest = np.sort(rms)[len(rms) // 2 :]
    return float(20 * np.log10(max(float(np.mean(loudest)), 1e-9)))
