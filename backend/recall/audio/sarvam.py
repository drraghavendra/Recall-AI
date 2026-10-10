import io
import logging
import wave

import httpx
import numpy as np

log = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


class SarvamTranscriber:
    """Sarvam AI speech-to-text REST API (paid; strong on Indian languages and code-mixed speech).

    One synchronous request per utterance. A failed request is logged and yields "" so the utterance
    is dropped instead of resetting the whole audio stream.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        language_code: str,
        mode: str | None,
        timeout_s: float,
        sample_rate: int = 16_000,
        client: httpx.Client | None = None,
    ):
        if not api_key:
            raise RuntimeError("SARVAM_API_KEY is not set (put it in backend/.env and start uvicorn with --env-file).")
        self._form = {"model": model, "language_code": language_code}
        if mode:
            self._form["mode"] = mode
        self._sample_rate = sample_rate
        # One pooled client: reusing the TLS connection saves a handshake on every utterance.
        self._client = client or httpx.Client(timeout=timeout_s)
        self._headers = {"api-subscription-key": api_key}

    def transcribe(self, audio: np.ndarray) -> str:
        files = {"file": ("utterance.wav", _to_wav(audio, self._sample_rate), "audio/wav")}
        try:
            response = self._client.post(SARVAM_STT_URL, headers=self._headers, data=self._form, files=files)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            log.error("Sarvam STT returned HTTP %d: %s", exc.response.status_code, exc.response.text[:300])
            return ""
        except httpx.HTTPError as exc:
            log.warning("Sarvam STT request failed: %s", exc)
            return ""
        return (response.json().get("transcript") or "").strip()
