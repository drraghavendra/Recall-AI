from typing import Literal

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
    # Best score below this = a genuinely new person; between this and the match threshold = ambiguous (ignored).
    local_face_new_face_threshold: float = 0.35
    local_face_min_det_score: float = 0.6
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
