import numpy as np
from faster_whisper import WhisperModel


class FasterWhisperTranscriber:
    def __init__(self, model_size: str):
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(audio, beam_size=1, condition_on_previous_text=False)
        return " ".join(segment.text.strip() for segment in segments).strip()
