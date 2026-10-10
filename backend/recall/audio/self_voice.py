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
        return self.classify_and_embed(audio, embed=False)[0]

    def classify_and_embed(self, audio: np.ndarray, embed: bool = True) -> tuple[Speaker, np.ndarray | None]:
        """Self or other, plus the clip's voice embedding (computed once; None if not needed)."""
        profile = self._profile
        if profile is None and not embed:
            return "other", None
        embedding = self.encoder.embed(audio)
        if profile is None:
            return "other", embedding
        return ("self" if float(np.dot(embedding, profile)) >= self._threshold else "other"), embedding
