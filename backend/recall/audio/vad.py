import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from recall.audio.types import VadEvent


class SileroVad:
    """Streaming Silero VAD. Event sample indices count from the last reset."""

    frame_size = 512

    def __init__(self, threshold: float, min_silence_ms: int, speech_pad_ms: int, sample_rate: int):
        self._iterator = VADIterator(
            load_silero_vad(),
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def process(self, frame: np.ndarray) -> VadEvent | None:
        result = self._iterator(torch.from_numpy(frame))
        if not result:
            return None
        if "start" in result:
            return VadEvent("start", int(result["start"]))
        return VadEvent("end", int(result["end"]))

    def reset(self) -> None:
        self._iterator.reset_states()
