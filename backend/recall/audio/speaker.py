import threading

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav


class ResemblyzerEncoder:
    def __init__(self, sample_rate: int):
        self._sample_rate = sample_rate
        self._encoder = VoiceEncoder(device="cpu", verbose=False)
        # Shared by the audio pipeline and the enrollment endpoint, which run in different threads.
        self._lock = threading.Lock()

    def embed(self, audio: np.ndarray) -> np.ndarray:
        wav = preprocess_wav(audio, source_sr=self._sample_rate, trim_silence=False)
        with self._lock:
            return self._encoder.embed_utterance(wav)
