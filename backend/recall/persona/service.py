import asyncio
import logging

from recall.db import Database
from recall.events import Publish
from recall.persona.models import FACT_GROUPS, Persona, PersonaUpdateError, PersonaWriter

log = logging.getLogger(__name__)


class PersonaService:
    """Read: get(face_id). Write: one background job that keeps personas up to date.

    The job takes the face <> transcript map of lines not yet sent to Claude, and for each face (in
    parallel) asks Claude whether those lines add anything. It runs:
    - `batch_s` after the first unsent line arrives (`line_added()`), so a long conversation is
      processed every `batch_s` while it's still going;
    - straight away when a conversation ends (`request_run()`);
    - every `cycle_s` otherwise, as a safety net that retries failed calls.
    Runs never overlap, so slow Claude calls don't pile up. Each face's bookmark only advances after its
    update succeeded; an unchanged persona is neither saved nor pushed to the UI.
    """

    def __init__(self, db: Database, publish: Publish, writer: PersonaWriter, cycle_s: float, batch_s: float = 30.0):
        self._db = db
        self._publish = publish
        self._writer = writer
        self._cycle_s = cycle_s
        self._batch_s = batch_s
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        # Event-loop time the next run is due, or None when nothing is waiting to be sent.
        self._due: float | None = None

    def get(self, face_id: str) -> dict | None:
        persona = self._db.get_persona(face_id)
        if persona is not None:
            # Older clients (the native Mac overlay) read a flat fact list.
            persona["key_facts"] = [fact for group in FACT_GROUPS for fact in persona.get(group, [])]
        return persona

    def request_run(self) -> None:
        """Run now (e.g. a conversation just ended)."""
        self._schedule(0.0)

    def line_added(self) -> None:
        """A new line was stored: make sure a run is due within `batch_s`, without pushing it back."""
        self._schedule(self._batch_s)

    def _schedule(self, delay_s: float) -> None:
        due = asyncio.get_running_loop().time() + delay_s
        if self._due is None or due < self._due:
            self._due = due
            self._wake.set()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            timeout = self._cycle_s if self._due is None else max(0.0, self._due - loop.time())
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=timeout)
            except TimeoutError:
                pass
            self._wake.clear()
            if self._due is not None and loop.time() < self._due:
                continue  # woken to reschedule, not yet due
            self._due = None
            try:
                await self.run_once()
            except Exception:
                log.exception("persona run failed")

    async def run_once(self) -> None:
        async with self._lock:
            pending = self._db.pending_lines()
            await asyncio.gather(*(self._update(face_id, segments) for face_id, segments in pending.items()))

    async def _update(self, face_id: str, segments) -> None:
        stored = self._db.get_persona(face_id)
        current = None if stored is None else Persona.model_validate({k: stored[k] for k in Persona.model_fields})
        try:
            update = await self._writer.update(current, [s.text for s in segments])
        except PersonaUpdateError:
            log.exception("persona update for %s failed; retrying next run", face_id)
            return
        changed = update.changed and update.persona.is_good() and update.persona != current
        log.info("persona %s: %d new lines, %s", face_id, len(segments), "updated" if changed else "nothing new")
        self._db.save_persona(face_id, update.persona.model_dump() if changed else None, segments[-1].id)
        if changed:
            self._publish({"type": "persona_updated", "face_id": face_id, "persona": self.get(face_id)})
