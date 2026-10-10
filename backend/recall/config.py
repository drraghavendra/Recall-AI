from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

FaceBackend = Literal["rekognition", "insightface", "mock"]
SttBackend = Literal["whisper", "parakeet", "sarvam"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECALL_", env_file=".env", extra="ignore")

    db_path: str = "recall.db"

    # Face provider selected by the recognizer factory. InsightFace runs locally; Rekognition uses AWS.
    face_backend: FaceBackend = "rekognition"
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
    # Which face is "in focus" (enrolled, photographed, and credited with other people's speech):
    # "center" = nearest the middle of the frame (on glasses, who you're looking at), "largest" = biggest.
    focus_rule: Literal["center", "largest"] = "center"
    face_confirm_windows: int = 2
    face_absent_confirm_windows: int = 3

    # Audio
    sample_rate: Literal[16000] = 16000
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 600
    vad_speech_pad_ms: int = 100
    # Long speech is cut into pieces this long, so each piece is transcribed, lip-checked and shown sooner.
    max_utterance_s: float = 6.0

    # Speech-to-text provider selected by the transcriber factory. Whisper and Parakeet run locally;
    # Sarvam is a paid API (key from SARVAM_API_KEY in the environment, like the other SDK credentials).
    stt_backend: SttBackend = "whisper"
    whisper_model: str = "base.en"
    parakeet_model: str = "nemo-parakeet-tdt-0.6b-v2"
    parakeet_quantization: str | None = "int8"
    sarvam_model: str = "saaras:v3"
    sarvam_language_code: str = "unknown"
    sarvam_mode: str | None = "transcribe"
    sarvam_timeout_s: float = 10.0

    self_voice_threshold: float = 0.75
    min_self_voice_enroll_s: float = 3.0

    # Conversation lifecycle
    min_speech_duration_s: float = 1.2
    min_silence_duration_s: float = 6.0

    # Attributing someone else's speech to the face in focus. Quieter speech is treated as background
    # (unset to disable); the log prints each utterance's level so this can be tuned per mic and room.
    # Off by default: the browser's auto gain (needed so quiet mics give usable speech) makes loudness a
    # poor near/far signal. Set it (e.g. -50) only for clients that send un-normalised audio.
    min_other_speech_dbfs: float | None = None
    # Lip check (InsightFace only), off unless RECALL_LIP_CHECK=true: still too unreliable on real
    # recordings, so by default speech simply goes to the focused face. When on, the client sends ~10 fps
    # lip frames while it hears sound. A line is the
    # focused face's only if its mouth openness (inner-lip gap / mouth width) varied by at least
    # `lip_min_openness_std` in `lip_min_talking_fraction` of the 1 s slices with sound, and (when there are
    # enough frames, and if `lip_min_sync` is set) rose and fell in step with the loudness. Otherwise it's
    # background. Every line logs both numbers ("lip check ...") for tuning.
    lip_check: bool = False
    # False: speech goes to the focused face, and lips only filter out lines no watched mouth moved for.
    # True: speech goes to whichever face's mouth moved most.
    lip_picks_speaker: bool = False
    lip_min_openness_std: float = 0.02
    lip_min_talking_fraction: float = 0.3
    # Off by default (None): on real recordings the sync score was too noisy to reject on; it's still logged.
    lip_min_sync: float | None = None

    # Persona (Anthropic credentials come from ANTHROPIC_API_KEY or an `ant auth login` profile)
    # Sonnet is plenty for a short profile and faster than Opus. The persona job runs `persona_batch_s`
    # after the first unsent line (so mid-conversation too), right away when a conversation ends, and
    # every `persona_cycle_s` as a safety net that retries failed calls.
    anthropic_model: str = "claude-sonnet-5-5"
    persona_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    persona_cycle_s: float = 90.0
    persona_batch_s: float = 30.0
