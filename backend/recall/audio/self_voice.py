import numpy as np

from recall.audio.types import Speaker, SpeakerEncoder


class SelfVoiceClassifier:
    """Binary self-vs-other check against the user's enrolled voice. Unenrolled -> everything is 'other'."""

    def __init__(self, encoder: SpeakerEncoder, threshold: float, profile: np.ndarray | None):
        self.encoder = encoder
        self._threshold = threshold
        self._profile = profile

    def set_profile(self, embedding: np.ndarray) -> None:
        self._profile = embedding

    def classify(self, audio: np.ndarray) -> Speaker:
        profile = self._profile
        if profile is None:
            return "other"
        similarity = float(np.dot(self.encoder.embed(audio), profile))
        return "self" if similarity >= self._threshold else "other"
