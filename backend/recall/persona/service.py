import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable

from recall.db import Database
from recall.events import Publish
from recall.matching.matcher import Conversation
from recall.persona.models import PersonaUpdateError, PersonaWriter

log = logging.getLogger(__name__)


class PersonaService:
    """Read: get(face_id). Write: watermarked incremental updates (DESIGN.md §7, persona freshness).

    Each conversation's `persona_watermark` is the last segment already sent to the LLM; it only
    advances after the persona is written, so a failed update is retried by the next run.
    """

    def __init__(
        self,
        db: Database,
        publish: Publish,
        writer: PersonaWriter,
        update_interval_s: float,
        min_new_segments: int,
        poll_interval_s: float = 5.0,
    ):
        self._db = db
        self._publish = publish
        self._writer = writer
        self._update_interval_s = update_interval_s
        self._min_new_segments = min_new_segments
        self._poll_interval_s = poll_interval_s
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._last_checkpoint: dict[int, float] = {}
        self._background: set[asyncio.Task] = set()

    def get(self, face_id: str) -> dict | None:
        return self._db.get_persona(face_id)

    def on_conversation_finalized(self, conv: Conversation) -> None:
        task = asyncio.create_task(self.finalize(conv))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def checkpoint_loop(self, get_open: Callable[[], Conversation | None]) -> None:
        while True:
            await asyncio.sleep(self._poll_interval_s)
            conv = get_open()
            if conv is None or conv.face_id is None:
                continue
            now = time.monotonic()
            last = self._last_checkpoint.setdefault(conv.conv_id, now)
            if now - last >= self._update_interval_s:
                self._last_checkpoint[conv.conv_id] = now
                await self.checkpoint(conv)

    async def checkpoint(self, conv: Conversation) -> None:
        await self._update(conv, min_new=self._min_new_segments, final=False)

    async def finalize(self, conv: Conversation) -> None:
        self._last_checkpoint.pop(conv.conv_id, None)
        await self._update(conv, min_new=1, final=True)

    async def _update(self, conv: Conversation, *, min_new: int, final: bool) -> None:
        if conv.face_id is None:
            return
        face_id = conv.face_id
        async with self._locks[face_id]:
            watermark = self._db.get_watermark(conv.conv_id)
            segments = self._db.other_segments_after(conv.conv_id, watermark)
            if len(segments) < min_new:
                if final and watermark:
                    self._db.increment_conversation_count(face_id)
                return
            try:
                update = await self._writer.update(self._db.get_persona(face_id), [s.text for s in segments])
            except PersonaUpdateError:
                log.exception("persona update for %s (conversation %d) failed", face_id, conv.conv_id)
                return
            self._db.upsert_persona(face_id, update.model_dump(), count_conversation=final)
            self._db.set_watermark(conv.conv_id, segments[-1].id)
            persona = self._db.get_persona(face_id)
        self._publish({"type": "persona_updated", "face_id": face_id, "persona": persona})
