"""Saaras REST adapter. Kept dependency-free so it can be enabled only with an API key."""

import io
import json
import os
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np


class SarvamTranscriber:
    def __init__(
        self,
        sample_rate: int,
        api_key: str | None = None,
        model: str = "saaras:v4",
        mode: str = "codemix",
        language_code: str = "unknown",
        keyterms: list[str] | None = None,
        timeout_s: float = 45.0,
    ):
        self._sample_rate = sample_rate
        self._api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self._api_key:
            raise ValueError("SARVAM_API_KEY is required when RECALL_TRANSCRIBER_BACKEND=sarvam")
        if model != "saaras:v4" and keyterms:
            raise ValueError("Sarvam keyterms are supported only by saaras:v4")
        if len(keyterms or []) > 50 or any(not term or len(term) > 64 for term in (keyterms or [])):
            raise ValueError("Sarvam keyterms must contain at most 50 non-empty values of 64 characters or less")
        self._model = model
        self._mode = mode
        self._language_code = language_code
        self._keyterms = keyterms or []
        self._timeout_s = timeout_s

    def transcribe(self, audio: np.ndarray) -> str:
        wav_bytes = self._to_wav(audio)
        boundary = f"RecallBoundary{uuid.uuid4().hex}"
        body = bytearray()
        self._add_field(body, boundary, "model", self._model)
        self._add_field(body, boundary, "mode", self._mode)
        self._add_field(body, boundary, "language_code", self._language_code)
        if self._keyterms:
            self._add_field(body, boundary, "keyterms", json.dumps(self._keyterms, ensure_ascii=False))
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"utterance.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode())
        body.extend(wav_bytes)
        body.extend(f"\r\n--{boundary}--\r\n".encode())

        request = urllib.request.Request(
            "https://api.sarvam.ai/speech-to-text",
            data=bytes(body),
            headers={
                "api-subscription-key": self._api_key,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Keep response diagnostics useful while never logging the API key.
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(f"Sarvam transcription failed (HTTP {exc.code}): {detail}") from exc
        transcript = payload.get("transcript")
        if not isinstance(transcript, str):
            raise RuntimeError("Sarvam response did not contain a transcript string")
        return transcript.strip()

    def _to_wav(self, audio: np.ndarray) -> bytes:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self._sample_rate)
            wav.writeframes(pcm)
        return output.getvalue()

    @staticmethod
    def _add_field(body: bytearray, boundary: str, name: str, value: str) -> None:
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode("utf-8"))
