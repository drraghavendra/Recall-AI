import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np

from recall.audio.pipeline import AudioPipeline
from recall.audio.self_voice import SelfVoiceClassifier
from recall.audio.speaker import ResemblyzerEncoder
from recall.audio.transcriber import FasterWhisperTranscriber
from recall.audio.vad import SileroVad
from recall.config import Settings
from recall.db import Database
from recall.events import EventHub
from recall.face.rekognition import RekognitionRecognizer
from recall.face.service import FaceService
from recall.face.tracker import FocusedFaceTracker
from recall.matching.matcher import ConversationMatcher
from recall.persona.llm import ClaudePersonaWriter
from recall.persona.service import PersonaService

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    db: Database
    hub: EventHub
    face: FaceService
    audio: AudioPipeline
    matcher: ConversationMatcher
    persona: PersonaService
    self_voice: SelfVoiceClassifier
    capture_client_connected: bool = False
    _tasks: list[asyncio.Task] = field(default_factory=list)

    def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self.face.run(), name="face"),
            asyncio.create_task(self.audio.run(), name="audio"),
            asyncio.create_task(
                self.persona.checkpoint_loop(lambda: self.matcher.open_conversation), name="persona-checkpoints"
            ),
        ]
        for task in self._tasks:
            task.add_done_callback(_log_unexpected_exit)

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.db.close()


def _log_unexpected_exit(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("background task %s crashed", task.get_name(), exc_info=task.exception())


def build_services(settings: Settings) -> Services:
    db = Database(settings.db_path)
    hub = EventHub()
    tracker = FocusedFaceTracker(settings.face_confirm_windows, settings.face_absent_confirm_windows)

    recognizer = RekognitionRecognizer(
        settings.rekognition_collection_id, settings.aws_region, settings.face_match_threshold
    )
    recognizer.ensure_collection()
    face = FaceService(recognizer, tracker, db, hub.publish, settings.frames_per_window)

    persona = PersonaService(
        db,
        hub.publish,
        ClaudePersonaWriter(settings.anthropic_model, settings.persona_effort),
        settings.persona_update_interval_s,
        settings.persona_min_new_segments,
    )
    matcher = ConversationMatcher(
        db,
        tracker,
        settings.min_speech_duration_s,
        settings.min_silence_duration_s,
        hub.publish,
        persona.on_conversation_finalized,
    )

    stored_profile = db.get_self_voice()
    self_voice = SelfVoiceClassifier(
        ResemblyzerEncoder(settings.sample_rate),
        settings.self_voice_threshold,
        None if stored_profile is None else np.frombuffer(stored_profile, dtype=np.float32),
    )
    if stored_profile is None:
        log.warning("no self voice enrolled yet: all speech will be treated as 'other' (POST /self-voice)")

    audio = AudioPipeline(
        SileroVad(settings.vad_threshold, settings.vad_min_silence_ms, settings.vad_speech_pad_ms, settings.sample_rate),
        FasterWhisperTranscriber(settings.whisper_model),
        self_voice,
        matcher,
        settings.sample_rate,
        settings.max_utterance_s,
    )
    return Services(settings, db, hub, face, audio, matcher, persona, self_voice)
