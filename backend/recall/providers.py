"""Runtime switching of the face recognizer and speech-to-text provider (no restart needed)."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Protocol, get_args

from pydantic import BaseModel

from recall.config import FaceBackend, SttBackend
from recall.events import Publish

log = logging.getLogger(__name__)

FACE_BACKENDS: tuple[str, ...] = get_args(FaceBackend)
STT_BACKENDS: tuple[str, ...] = get_args(SttBackend)


class ProviderChoice(BaseModel):
    """Body of PUT /providers; omitted fields keep their current provider."""

    face_backend: FaceBackend | None = None
    stt_backend: SttBackend | None = None


class ProviderError(RuntimeError):
    """The requested provider could not be started; the previous one is still active."""


class _FaceTarget(Protocol):
    def set_recognizer(self, recognizer: Any) -> None: ...


class _SttTarget(Protocol):
    def set_transcriber(self, transcriber: Any) -> None: ...


class ProviderSwitcher:
    """Swaps the live providers inside FaceService / AudioPipeline.

    A provider is built off the event loop the first time it is selected (model loads can take
    seconds, or minutes on a first download) and then cached, so switching back is instant at the
    cost of keeping its model in memory. If building fails, the current provider stays in place.
    """

    def __init__(
        self,
        face: _FaceTarget,
        audio: _SttTarget,
        build_recognizer: Callable[[str], Any],
        build_transcriber: Callable[[str], Any],
        publish: Publish,
        face_backend: str,
        recognizer: Any,
        stt_backend: str,
        transcriber: Any,
    ):
        self._builders = {"face_backend": build_recognizer, "stt_backend": build_transcriber}
        self._apply = {"face_backend": face.set_recognizer, "stt_backend": audio.set_transcriber}
        self._cache = {("face_backend", face_backend): recognizer, ("stt_backend", stt_backend): transcriber}
        self._current = {"face_backend": face_backend, "stt_backend": stt_backend}
        self._publish = publish
        self._lock = asyncio.Lock()

    @property
    def current(self) -> dict[str, str]:
        return dict(self._current)

    def options(self) -> dict[str, list[str]]:
        return {"face_backend": list(FACE_BACKENDS), "stt_backend": list(STT_BACKENDS)}

    async def switch(self, choice: ProviderChoice) -> dict[str, str]:
        changed = False
        try:
            async with self._lock:
                for kind in ("face_backend", "stt_backend"):
                    name = getattr(choice, kind)
                    if name is None or name == self._current[kind]:
                        continue
                    provider = self._cache.get((kind, name))
                    if provider is None:
                        log.info("starting %s provider %s", kind, name)
                        try:
                            provider = await asyncio.to_thread(self._builders[kind], name)
                        except Exception as exc:
                            log.exception("could not start %s provider %s", kind, name)
                            raise ProviderError(f"Could not start {name}: {exc}") from exc
                        self._cache[(kind, name)] = provider
                    self._apply[kind](provider)
                    self._current[kind] = name
                    changed = True
                    log.info("switched %s to %s", kind, name)
                return self.current
        finally:
            # Also on a partial failure (e.g. face switched, speech failed) so clients stay in sync.
            if changed:
                self._publish({"type": "providers_changed", **self.current})
