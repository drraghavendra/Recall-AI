class ChunkClock:
    """Wall-clock timestamps for gapless audio chunks.

    The first chunk is anchored at `now - its duration` (the callback fires once the block is
    full); later chunks are anchored by sample count, matching how the backend derives time.
    Call `resync()` after dropped samples (input overflow) to re-anchor on the next chunk.
    """

    def __init__(self, sample_rate: int):
        self._sample_rate = sample_rate
        self._start: float | None = None
        self._samples = 0

    def stamp(self, n_samples: int, now: float) -> float:
        if self._start is None:
            self._start = now - n_samples / self._sample_rate
            self._samples = 0
        ts = self._start + self._samples / self._sample_rate
        self._samples += n_samples
        return ts

    def resync(self) -> None:
        self._start = None
