import io
import wave

import httpx
import numpy as np
import pytest

from recall.audio.parakeet import ParakeetTranscriber
from recall.audio.sarvam import SARVAM_STT_URL, SarvamTranscriber

ONE_SECOND = np.zeros(16_000, dtype=np.float32)


class FakeParakeet:
    def __init__(self):
        self.calls = []

    def recognize(self, audio, sample_rate):
        self.calls.append((audio.dtype, len(audio), sample_rate))
        return "  hello there  "


def test_parakeet_passes_16k_float32_and_strips_text():
    model = FakeParakeet()
    text = ParakeetTranscriber("nemo-parakeet-tdt-0.6b-v2", model=model).transcribe(ONE_SECOND.astype(np.float64))
    assert text == "hello there"
    assert model.calls == [(np.float32, 16_000, 16_000)]


def sarvam(handler, mode="transcribe"):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return SarvamTranscriber("key-123", "saaras:v3", "unknown", mode, 5.0, client=client)


def test_sarvam_sends_wav_with_key_and_settings():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers["api-subscription-key"]
        seen["body"] = request.read()
        return httpx.Response(200, json={"request_id": "r1", "transcript": " namaste ", "language_code": "hi-IN"})

    assert sarvam(handler).transcribe(ONE_SECOND) == "namaste"
    assert seen["url"] == SARVAM_STT_URL
    assert seen["key"] == "key-123"
    body = seen["body"]
    for field, value in [("model", b"saaras:v3"), ("language_code", b"unknown"), ("mode", b"transcribe")]:
        assert f'name="{field}"'.encode() in body and value in body
    wav_bytes = body[body.index(b"RIFF") :]
    with wave.open(io.BytesIO(wav_bytes)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16_000)


def test_sarvam_omits_mode_when_unset():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, json={"transcript": "hi"})

    sarvam(handler, mode=None).transcribe(ONE_SECOND)
    assert b'name="mode"' not in seen["body"]


@pytest.mark.parametrize(
    "respond",
    [lambda _r: httpx.Response(401, json={"error": "bad key"}), lambda _r: (_ for _ in ()).throw(httpx.ConnectTimeout("slow"))],
)
def test_sarvam_failure_drops_the_utterance_instead_of_raising(respond):
    assert sarvam(respond).transcribe(ONE_SECOND) == ""


def test_sarvam_requires_an_api_key():
    with pytest.raises(RuntimeError, match="SARVAM_API_KEY"):
        SarvamTranscriber("", "saaras:v3", "unknown", None, 5.0)
