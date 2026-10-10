from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECALL_", env_file=".env", extra="ignore")

    db_path: str = "recall.db"

    # Face provider selected by the recognizer factory. InsightFace runs locally; Rekognition uses AWS.
    face_backend: Literal["rekognition", "insightface", "mock"] = "rekognition"
    mock_face_count: int = 3
    mock_face_hold_windows: int = 6

    # AWS credentials come from the standard boto3 chain: env vars, ~/.aws, SSO profile
    aws_region: str = "us-east-1"
    rekognition_collection_id: str = "recall-ai-faces"
    face_match_threshold: float = 90.0
    local_face_db_path: str = "recall_faces.sqlite3"
    local_face_match_threshold: float = 0.55
    frames_per_window: int = 1
    face_confirm_windows: int = 2
    face_absent_confirm_windows: int = 3

    # Audio
    sample_rate: Literal[16000] = 16000
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 600
    vad_speech_pad_ms: int = 100
    max_utterance_s: float = 15.0
    whisper_model: str = "base.en"
    transcriber_backend: Literal["faster-whisper", "sarvam"] = "faster-whisper"
    sarvam_model: str = "saaras:v4"
    sarvam_mode: Literal["transcribe", "translate", "verbatim", "translit", "codemix"] = "codemix"
    sarvam_language_code: str = "unknown"
    sarvam_keyterms: list[str] = Field(default_factory=list)
    speech_enhancer: Literal["none", "deepfilternet"] = "none"
    self_voice_threshold: float = 0.75
    min_self_voice_enroll_s: float = 3.0

    # Conversation lifecycle
    min_speech_duration_s: float = 1.2
    min_silence_duration_s: float = 6.0

    # Persona (Anthropic credentials come from ANTHROPIC_API_KEY or an `ant auth login` profile)
    anthropic_model: str = "claude-opus-5-5"
    persona_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    persona_update_interval_s: float = 300.0
    persona_min_new_segments: int = 3
