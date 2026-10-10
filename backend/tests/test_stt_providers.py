import numpy as np

from recall.audio.parakeet import ParakeetTranscriber

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

