import asyncio

import pytest
from pydantic import ValidationError

from recall.providers import FACE_BACKENDS, STT_BACKENDS, ProviderChoice, ProviderError, ProviderSwitcher


class Target:
    def __init__(self):
        self.recognizer = self.transcriber = None

    def set_recognizer(self, recognizer):
        self.recognizer = recognizer

    def set_transcriber(self, transcriber):
        self.transcriber = transcriber


@pytest.fixture
def env():
    target, events, built = Target(), [], []

    def build(kind):
        def factory(name):
            if name in ("rekognition", "sarvam"):
                raise RuntimeError(f"{name} needs credentials")
            built.append((kind, name))
            return f"{kind}:{name}"

        return factory

    switcher = ProviderSwitcher(
        target, target, build("face"), build("stt"), events.append, "insightface", "face:insightface", "whisper", "stt:whisper"
    )
    return switcher, target, events, built


def run(coro):
    return asyncio.run(coro)


def test_switch_builds_once_applies_and_announces(env):
    switcher, target, events, built = env
    assert run(switcher.switch(ProviderChoice(face_backend="mock", stt_backend="parakeet"))) == {
        "face_backend": "mock",
        "stt_backend": "parakeet",
    }
    assert (target.recognizer, target.transcriber) == ("face:mock", "stt:parakeet")
    assert events == [{"type": "providers_changed", "face_backend": "mock", "stt_backend": "parakeet"}]

    run(switcher.switch(ProviderChoice(stt_backend="whisper")))
    run(switcher.switch(ProviderChoice(stt_backend="parakeet")))
    assert built == [("face", "mock"), ("stt", "parakeet")]  # startup providers and repeats come from the cache
    assert target.transcriber == "stt:parakeet"


def test_choosing_the_current_provider_is_a_no_op(env):
    switcher, target, events, built = env
    assert run(switcher.switch(ProviderChoice(face_backend="insightface"))) == switcher.current
    assert (target.recognizer, events, built) == (None, [], [])


def test_failed_start_keeps_the_current_provider(env):
    switcher, target, events, _ = env
    with pytest.raises(ProviderError, match="sarvam needs credentials"):
        run(switcher.switch(ProviderChoice(stt_backend="sarvam")))
    assert switcher.current["stt_backend"] == "whisper"
    assert (target.transcriber, events) == (None, [])


def test_partial_failure_still_announces_what_did_change(env):
    switcher, _, events, _ = env
    with pytest.raises(ProviderError):
        run(switcher.switch(ProviderChoice(face_backend="mock", stt_backend="sarvam")))
    assert switcher.current == {"face_backend": "mock", "stt_backend": "whisper"}
    assert events == [{"type": "providers_changed", "face_backend": "mock", "stt_backend": "whisper"}]


def test_options_match_the_settings_and_unknown_names_are_rejected(env):
    switcher, *_ = env
    assert switcher.options() == {"face_backend": list(FACE_BACKENDS), "stt_backend": list(STT_BACKENDS)}
    assert set(FACE_BACKENDS) == {"rekognition", "insightface", "mock"}
    assert set(STT_BACKENDS) == {"whisper", "parakeet", "sarvam"}
    with pytest.raises(ValidationError):
        ProviderChoice(face_backend="opencv")
