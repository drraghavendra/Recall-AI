from collections.abc import Callable
import time

import numpy as np
import sounddevice as sd

from recall_client.timing import ChunkClock


class MicStream:
    """Continuous 16 kHz mono s16le capture. `on_chunk(ts, pcm)` runs on PortAudio's thread — keep it non-blocking."""

    def __init__(self, sample_rate: int, block_s: float, on_chunk: Callable[[float, bytes], None]):
        self._clock = ChunkClock(sample_rate)
        self._on_chunk = on_chunk
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=int(sample_rate * block_s),
            callback=self._callback,
        )

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags) -> None:
        if status.input_overflow:
            self._clock.resync()
        ts = self._clock.stamp(frames, time.time())
        self._on_chunk(ts, indata[:, 0].astype("<i2").tobytes())


def record(seconds: float, sample_rate: int) -> bytes:
    """Blocking recording, for one-off voice enrollment."""
    samples = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    return samples[:, 0].astype("<i2").tobytes()
