import asyncio
import logging
from collections.abc import Callable

log = logging.getLogger(__name__)

Publish = Callable[[dict], None]


class EventHub:
    """Fan-out of server events (face_update, persona_updated, ...) to connected clients."""

    def __init__(self, queue_size: int = 256):
        self._queue_size = queue_size
        self._subscribers: set[asyncio.Queue[dict]] = set()

    def subscribe(self) -> asyncio.Queue[dict]:
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict]) -> None:
        self._subscribers.discard(queue)

    def publish(self, event: dict) -> None:
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("dropping %s event for a slow subscriber", event.get("type"))
