"""Audio interfaces and data types. Deliberately import-light: no torch / whisper / resemblyzer."""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

Speaker = Literal["self", "other"]


@dataclass(frozen=True)
class VadEvent:
    kind: Literal["start", "end"]
    sample: int


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    text: str
    speaker: Speaker
    level_dbfs: float | None = None


class VoiceActivityDetector(Protocol):
    frame_size: int

    def process(self, frame: np.ndarray) -> VadEvent | None:
        """Feed exactly `frame_size` float32 samples; returns an event when speech starts/ends."""
        ...

    def reset(self) -> None: ...


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray) -> str:
        """16 kHz mono float32 -> text. Blocks; call off the event loop."""
        ...


class SpeakerEncoder(Protocol):
    def embed(self, audio: np.ndarray) -> np.ndarray:
        """16 kHz mono float32 -> L2-normalised voice embedding. Blocks; call off the event loop."""
        ...


class UtteranceListener(Protocol):
    def on_utterance(self, utterance: Utterance) -> None: ...

    def on_audio_time(self, ts: float, in_speech: bool) -> None:
        """Called after every chunk with the stream time processed so far."""
        ...

    def on_stream_end(self) -> None:
        """The audio stream stopped (e.g. client disconnected); no more audio will arrive for it."""
        ...
