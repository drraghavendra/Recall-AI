import asyncio
import logging

import numpy as np

from recall.audio.pcm import pcm16_to_float
from recall.audio.self_voice import SelfVoiceClassifier
from recall.audio.types import (
    Speaker,
    Transcriber,
    Utterance,
    UtteranceListener,
    VadEvent,
    VoiceActivityDetector,
)

log = logging.getLogger(__name__)


class AudioPipeline:
    """Continuous mic stream -> VAD utterances -> (transcript, self/other) -> listener.

    Chunks are processed strictly in order on one task, so the listener sees every utterance before
    any later `on_audio_time` — conversation decisions never race ahead of transcription.
    Timestamps are derived from the first chunk's `ts` plus the sample count, which assumes the
    client sends audio without gaps; `reset()` starts a new stream (e.g. on client reconnect).
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        transcriber: Transcriber,
        speaker: SelfVoiceClassifier,
        listener: UtteranceListener,
        sample_rate: int,
        max_utterance_s: float,
        preroll_s: float = 1.0,
    ):
        self._vad = vad
        self._transcriber = transcriber
        self._speaker = speaker
        self._listener = listener
        self._sample_rate = sample_rate
        self._max_utterance = int(max_utterance_s * sample_rate)
        self._preroll = int(preroll_s * sample_rate)
        self._queue: asyncio.Queue[tuple[float, bytes] | None] = asyncio.Queue()
        self._clear_stream()

    def set_transcriber(self, transcriber: Transcriber) -> None:
        """Swap the provider at runtime; an utterance already being transcribed finishes on the old one."""
        self._transcriber = transcriber

    def submit(self, ts: float, pcm: bytes) -> None:
        self._queue.put_nowait((ts, pcm))

    def reset(self) -> None:
        self._queue.put_nowait(None)

    async def run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    await self._end_stream()
                else:
                    await self._process(*item)
            except Exception:
                log.exception("audio pipeline failed on a chunk; resetting stream")
                self._clear_stream()

    def _clear_stream(self) -> None:
        self._stream_start: float | None = None
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_offset = 0
        self._fed = 0
        self._speech_start: int | None = None
        self._vad.reset()

    async def _end_stream(self) -> None:
        if self._speech_start is not None:
            await self._emit(self._speech_start, self._fed)
        if self._stream_start is not None:
            self._listener.on_stream_end()
        self._clear_stream()

    async def _process(self, ts: float, pcm: bytes) -> None:
        if self._stream_start is None:
            self._stream_start = ts
        self._buffer = np.concatenate([self._buffer, pcm16_to_float(pcm)])

        frame_size = self._vad.frame_size
        while self._fed + frame_size <= self._buffer_offset + len(self._buffer):
            offset = self._fed - self._buffer_offset
            event = self._vad.process(self._buffer[offset : offset + frame_size])
            self._fed += frame_size
            if event is not None:
                await self._on_vad_event(event)
            elif self._speech_start is not None and self._fed - self._speech_start >= self._max_utterance:
                await self._emit(self._speech_start, self._fed)
                self._speech_start = self._fed

        self._trim()
        self._listener.on_audio_time(self._ts(self._fed), in_speech=self._speech_start is not None)

    async def _on_vad_event(self, event: VadEvent) -> None:
        if event.kind == "start":
            self._speech_start = max(event.sample, self._buffer_offset)
        elif self._speech_start is not None:
            await self._emit(self._speech_start, min(event.sample, self._fed))
            self._speech_start = None

    async def _emit(self, start: int, end: int) -> None:
        audio = self._buffer[start - self._buffer_offset : end - self._buffer_offset]
        if audio.size == 0:
            return
        text, speaker = await asyncio.to_thread(self._analyze, audio)
        if any(ch.isalnum() for ch in text):  # Whisper emits e.g. ". . . ." on noise
            self._listener.on_utterance(Utterance(self._ts(start), self._ts(end), text, speaker))

    def _analyze(self, audio: np.ndarray) -> tuple[str, Speaker]:
        return self._transcriber.transcribe(audio), self._speaker.classify(audio)

    def _trim(self) -> None:
        keep_from = (self._speech_start if self._speech_start is not None else self._fed) - self._preroll
        drop = keep_from - self._buffer_offset
        if drop > 0:
            self._buffer = self._buffer[drop:]
            self._buffer_offset += drop

    def _ts(self, sample: int) -> float:
        return self._stream_start + sample / self._sample_rate
