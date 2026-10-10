import asyncio
import logging
import os
from dataclasses import dataclass, field

import numpy as np

from recall.audio.pipeline import AudioPipeline
from recall.audio.self_voice import SelfVoiceClassifier
from recall.audio.speaker import ResemblyzerEncoder
from recall.audio.types import Transcriber
from recall.audio.vad import SileroVad
from recall.config import Settings
from recall.db import Database
from recall.events import EventHub
from recall.face.lips import LipActivityTracker, LipService
from recall.face.mock import MockRecognizer
from recall.face.recognizer import FaceRecognizer
from recall.face.rekognition import RekognitionRecognizer
from recall.face.service import FaceService
from recall.face.tracker import FocusedFaceTracker, PresenceTracker
from recall.matching.attribution import AttributionGate
from recall.matching.matcher import ConversationMatcher
from recall.matching.voices import VoiceBook
from recall.persona.llm import ClaudePersonaWriter
from recall.persona.service import PersonaService
from recall.providers import ProviderSwitcher

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
    providers: ProviderSwitcher
    lips: LipService | None = None
    voices: VoiceBook | None = None
    capture_client_connected: bool = False
    _tasks: list[asyncio.Task] = field(default_factory=list)

    def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self.face.run(), name="face"),
            asyncio.create_task(self.audio.run(), name="audio"),
            asyncio.create_task(self.persona.run(), name="persona-cycle"),
        ]
        if self.lips is not None:
            self._tasks.append(asyncio.create_task(self.lips.run(), name="lips"))
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


def _build_recognizer(settings: Settings) -> FaceRecognizer:
    if settings.face_backend == "mock":
        log.warning(
            "face backend is MOCK: reporting one of ids 1..%d, switching every %d windows",
            settings.mock_face_count,
            settings.mock_face_hold_windows,
        )
        return MockRecognizer(settings.mock_face_count, settings.mock_face_hold_windows)
    if settings.face_backend == "insightface":
        # Import only when selected so Rekognition/mock installs do not need the ML extra.
        from recall.face.insightface import InsightFaceRecognizer

        return InsightFaceRecognizer(
            settings.local_face_db_path,
            settings.local_face_match_threshold,
            settings.local_face_new_face_threshold,
            settings.local_face_min_det_score,
            focus=settings.focus_rule,
        )
    recognizer = RekognitionRecognizer(
        settings.rekognition_collection_id, settings.aws_region, settings.face_match_threshold
    )
    recognizer.ensure_collection()
    return recognizer


def _build_transcriber(settings: Settings) -> Transcriber:
    # Each provider is imported only when selected, so an install needs just that provider's extra.
    if settings.stt_backend == "parakeet":
        from recall.audio.parakeet import ParakeetTranscriber

        log.info("speech-to-text: Parakeet %s (%s)", settings.parakeet_model, settings.parakeet_quantization or "fp32")
        return ParakeetTranscriber(settings.parakeet_model, settings.parakeet_quantization)
    if settings.stt_backend == "sarvam":
        from recall.audio.sarvam import SarvamTranscriber

        log.info("speech-to-text: Sarvam API %s (language %s)", settings.sarvam_model, settings.sarvam_language_code)
        return SarvamTranscriber(
            os.environ.get("SARVAM_API_KEY", ""),
            settings.sarvam_model,
            settings.sarvam_language_code,
            settings.sarvam_mode,
            settings.sarvam_timeout_s,
            settings.sample_rate,
        )
    from recall.audio.transcriber import FasterWhisperTranscriber

    log.info("speech-to-text: faster-whisper %s", settings.whisper_model)
    return FasterWhisperTranscriber(settings.whisper_model)


def build_services(settings: Settings) -> Services:
    db = Database(settings.db_path)
    hub = EventHub()
    tracker = FocusedFaceTracker(settings.face_confirm_windows, settings.face_absent_confirm_windows)

    recognizer = _build_recognizer(settings)
    presence = PresenceTracker(settings.face_confirm_windows, settings.face_absent_confirm_windows)
    face = FaceService(recognizer, tracker, db, hub.publish, settings.frames_per_window, presence=presence)

    lips = None
    mouth_landmarks_all = getattr(recognizer, "mouth_landmarks_all", None)
    if settings.lip_check and mouth_landmarks_all is not None:
        lips = LipService(mouth_landmarks_all, face.known_boxes, LipActivityTracker(
                settings.lip_min_openness_std,
                min_talking_fraction=settings.lip_min_talking_fraction,
                min_sync=settings.lip_min_sync,
            ))

    persona = PersonaService(
        db,
        hub.publish,
        ClaudePersonaWriter(settings.anthropic_model, settings.persona_effort),
        settings.persona_cycle_s,
        settings.persona_batch_s,
    )
    voices = None
    if settings.voice_id:
        voices = VoiceBook(
            db,
            settings.voice_same_threshold,
            settings.voice_other_threshold,
            settings.voice_min_samples,
            settings.voice_min_seconds,
        )
    matcher = ConversationMatcher(
        db,
        tracker,
        settings.min_speech_duration_s,
        settings.min_silence_duration_s,
        hub.publish,
        lambda _conv: persona.request_run(),  # a conversation just ended: don't wait for the cycle
        gate=AttributionGate(
            settings.min_other_speech_dbfs,
            None if lips is None else lips.tracker,
            settings.lip_picks_speaker,
            # The focused face for most of the line, else whoever was focused when it started.
            lambda start, end: db.majority_face(start, end) or tracker.face_at(start),
            voices,
            # Who's in view: debounced presence, plus anyone recognised in the last few seconds (the line
            # is judged a moment after it was said).
            lambda: presence.present | set(face.known_boxes(max_age_s=8.0)),
        ),
        max_utterance_s=settings.max_utterance_s,
        on_line=persona.line_added,
    )

    stored_profile = db.get_self_voice()
    self_voice = SelfVoiceClassifier(
        ResemblyzerEncoder(settings.sample_rate),
        settings.self_voice_threshold,
        None if stored_profile is None else np.frombuffer(stored_profile, dtype=np.float32),
    )
    if stored_profile is None:
        log.warning("no self voice enrolled yet: all speech will be treated as 'other' (POST /self-voice)")

    transcriber = _build_transcriber(settings)
    audio = AudioPipeline(
        SileroVad(settings.vad_threshold, settings.vad_min_silence_ms, settings.vad_speech_pad_ms, settings.sample_rate),
        transcriber,
        self_voice,
        matcher,
        settings.sample_rate,
        settings.max_utterance_s,
    )
    providers = ProviderSwitcher(
        face,
        audio,
        lambda name: _build_recognizer(settings.model_copy(update={"face_backend": name})),
        lambda name: _build_transcriber(settings.model_copy(update={"stt_backend": name})),
        hub.publish,
        settings.face_backend,
        recognizer,
        settings.stt_backend,
        transcriber,
    )
    return Services(settings, db, hub, face, audio, matcher, persona, self_voice, providers, lips, voices)
