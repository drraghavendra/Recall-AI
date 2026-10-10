"""Read-path view state (DESIGN.md §5a). Pure: no camera, network or GUI.

Fed by server events from the connection thread, read by the UI on the main thread — every
public method takes the lock. Methods that need a persona fetch return the face_id to fetch;
the caller performs the HTTP request and reports back with `persona_fetched` / `fetch_failed`.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ViewSnapshot:
    face_id: str | None
    persona: dict | None
    first_meeting: bool
    in_conversation: bool
    conversation_face_id: str | None
    conversation_persona: dict | None


@dataclass
class _CacheEntry:
    persona: dict | None
    fetched_at: float


class ViewState:
    def __init__(self, persona_ttl_s: float, clock: Callable[[], float] = time.monotonic):
        self._ttl_s = persona_ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._face_id: str | None = None
        self._cache: dict[str, _CacheEntry] = {}
        self._pending: set[str] = set()
        self._in_conversation = False
        self._conversation_face_id: str | None = None

    def face_update(self, face_id: str | None) -> str | None:
        """Returns a face_id to fetch, or None when the cache is good (or no face is in view)."""
        with self._lock:
            self._face_id = face_id
            if face_id is None or face_id in self._pending:
                return None
            entry = self._cache.get(face_id)
            if entry is not None and self._clock() - entry.fetched_at < self._ttl_s:
                return None
            self._pending.add(face_id)
            return face_id

    def persona_fetched(self, face_id: str, persona: dict | None) -> None:
        """`persona=None` means the server has none yet (404): a first meeting."""
        with self._lock:
            self._pending.discard(face_id)
            self._cache[face_id] = _CacheEntry(persona, self._clock())

    def fetch_failed(self, face_id: str) -> None:
        with self._lock:
            self._pending.discard(face_id)

    def persona_updated(self, face_id: str, persona: dict) -> None:
        with self._lock:
            self._cache[face_id] = _CacheEntry(persona, self._clock())

    def conversation_started(self, face_id: str | None) -> None:
        with self._lock:
            self._in_conversation = True
            self._conversation_face_id = face_id

    def conversation_ended(self) -> None:
        with self._lock:
            self._in_conversation = False
            self._conversation_face_id = None

    def snapshot(self) -> ViewSnapshot:
        with self._lock:
            entry = self._cache.get(self._face_id) if self._face_id else None
            conv_entry = self._cache.get(self._conversation_face_id) if self._conversation_face_id else None
            return ViewSnapshot(
                face_id=self._face_id,
                persona=entry.persona if entry else None,
                first_meeting=entry is not None and entry.persona is None,
                in_conversation=self._in_conversation,
                conversation_face_id=self._conversation_face_id,
                conversation_persona=conv_entry.persona if conv_entry else None,
            )
