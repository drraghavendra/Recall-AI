import asyncio
import logging

from recall.db import Database
from recall.events import Publish
from recall.persona.models import FACT_GROUPS, Persona, PersonaUpdateError, PersonaWriter

log = logging.getLogger(__name__)


class PersonaService:
    """Read: get(face_id). Write: one background job that keeps personas up to date.

    Every `cycle_s` after the previous run finished (so slow Claude calls never overlap), the job takes
    the face <> transcript map of lines not yet sent to Claude, and for each face asks Claude whether
    those lines add anything. Each face's bookmark only advances after its update succeeded, so a failed
    call is retried on the next cycle; an unchanged persona is neither saved nor pushed to the UI.
    `request_run()` runs the job early (e.g. when a conversation ends) without waiting for the cycle.
    """

    def __init__(self, db: Database, publish: Publish, writer: PersonaWriter, cycle_s: float):
        self._db = db
        self._publish = publish
        self._writer = writer
        self._cycle_s = cycle_s
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()

    def get(self, face_id: str) -> dict | None:
        persona = self._db.get_persona(face_id)
        if persona is not None:
            # Older clients (the native Mac overlay) read a flat fact list.
            persona["key_facts"] = [fact for group in FACT_GROUPS for fact in persona.get(group, [])]
        return persona

    def request_run(self) -> None:
        self._wake.set()

    async def run(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._cycle_s)
            except TimeoutError:
                pass
            self._wake.clear()
            try:
                await self.run_once()
            except Exception:
                log.exception("persona cycle failed")

    async def run_once(self) -> None:
        async with self._lock:
            pending = self._db.pending_lines()
            for face_id, segments in pending.items():
                await self._update(face_id, segments)

    async def _update(self, face_id: str, segments) -> None:
        stored = self._db.get_persona(face_id)
        current = None if stored is None else Persona.model_validate({k: stored[k] for k in Persona.model_fields})
        try:
            update = await self._writer.update(current, [s.text for s in segments])
        except PersonaUpdateError:
            log.exception("persona update for %s failed; retrying next cycle", face_id)
            return
        changed = update.changed and update.persona.is_good() and update.persona != current
        log.info("persona %s: %d new lines, %s", face_id, len(segments), "updated" if changed else "nothing new")
        self._db.save_persona(face_id, update.persona.model_dump() if changed else None, segments[-1].id)
        if changed:
            self._publish({"type": "persona_updated", "face_id": face_id, "persona": self.get(face_id)})
