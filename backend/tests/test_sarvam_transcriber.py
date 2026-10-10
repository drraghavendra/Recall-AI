import io
import wave
from unittest.mock import patch

import numpy as np
import pytest

from recall.audio.sarvam import SarvamTranscriber


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return '{"transcript":"ನನ್ನ phone ಇದೆ"}'.encode("utf-8")


def test_sarvam_sends_wav_codemix_and_keyterms():
    transcriber = SarvamTranscriber(
        sample_rate=16000,
        api_key="test-key",
        mode="codemix",
        keyterms=["Koramangala", "swalpa adjust maadi"],
    )

    with patch("urllib.request.urlopen", return_value=FakeResponse()) as open_url:
        result = transcriber.transcribe(np.array([0.0, 0.25, -0.25], dtype=np.float32))

    assert result == "ನನ್ನ phone ಇದೆ"
    request = open_url.call_args.args[0]
    assert request.full_url == "https://api.sarvam.ai/speech-to-text"
    assert request.get_header("Api-subscription-key") == "test-key"
    body = request.data
    assert b'name="mode"\r\n\r\ncodemix' in body
    assert b'name="keyterms"\r\n\r\n["Koramangala", "swalpa adjust maadi"]' in body
    wav_start = body.index(b"RIFF")
    with wave.open(io.BytesIO(body[wav_start : body.rfind(b"\r\n--RecallBoundary")]), "rb") as audio:
        assert audio.getframerate() == 16000
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getnframes() == 3


def test_sarvam_requires_api_key():
    with patch.dict("os.environ", {"SARVAM_API_KEY": ""}):
        try:
            SarvamTranscriber(sample_rate=16000)
        except ValueError as exc:
            assert "SARVAM_API_KEY" in str(exc)
        else:
            raise AssertionError("expected missing API key to be rejected")


def test_sarvam_keyterm_limits_are_validated():
    with pytest.raises(ValueError, match="at most 50"):
        SarvamTranscriber(sample_rate=16000, api_key="test", keyterms=[f"term-{n}" for n in range(51)])


