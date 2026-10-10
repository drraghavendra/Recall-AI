import asyncio
import json
import logging
import threading
from collections.abc import Callable
from enum import Enum

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

log = logging.getLogger(__name__)

TRY_AGAIN_LATER = 1013


class Status(Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    BUSY = "another capture client is connected"
    DISCONNECTED = "disconnected"


class ServerConnection:
    """WebSocket to the backend on its own thread + asyncio loop, reconnecting with backoff.

    `send` is thread-safe (called from the camera loop and the PortAudio callback). Messages sent
    while disconnected are dropped — the server starts a fresh audio stream on every connection.
    `on_event` runs on this connection's loop; it may schedule coroutines there.
    """

    def __init__(self, url: str, on_event: Callable[[dict], None], outbox_size: int = 500):
        self._url = url
        self._on_event = on_event
        self._outbox_size = outbox_size
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="server-connection", daemon=True)
        self._outbox: asyncio.Queue[str] | None = None
        self._ws: ClientConnection | None = None
        self._main: asyncio.Task | None = None
        self.status = Status.CONNECTING

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        if self._thread.is_alive():
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        self._thread.join(timeout=5)

    async def _shutdown(self) -> None:
        if self._ws is not None:
            await self._ws.close()
        if self._main is not None:
            self._main.cancel()

    def send(self, text: str) -> None:
        if self._loop.is_closed():
            return
        try:
            self._loop.call_soon_threadsafe(self._enqueue, text)
        except RuntimeError:
            pass  # loop closed between the check and the call (shutdown)

    def _enqueue(self, text: str) -> None:
        if self.status is not Status.CONNECTED or self._outbox is None:
            return
        try:
            self._outbox.put_nowait(text)
        except asyncio.QueueFull:
            log.warning("server is not keeping up; dropping an outbound message")

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._main = self._loop.create_task(self._reconnect_forever())
        try:
            self._loop.run_until_complete(self._main)
        except asyncio.CancelledError:
            pass
        finally:
            self._loop.close()

    async def _reconnect_forever(self) -> None:
        backoff = 1.0
        while True:
            self.status = Status.CONNECTING
            try:
                close_code = await self._session()
            except (OSError, TimeoutError, InvalidHandshake) as e:
                log.warning("cannot reach %s: %s", self._url, e)
                self.status = Status.DISCONNECTED
            else:
                if close_code == TRY_AGAIN_LATER:
                    self.status = Status.BUSY
                    log.error("server rejected this client: another capture client is already connected")
                else:
                    self.status = Status.DISCONNECTED
                    backoff = 1.0
            log.info("reconnecting in %.0fs", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    async def _session(self) -> int | None:
        async with connect(self._url, max_size=None) as ws:
            self._ws = ws
            self._outbox = asyncio.Queue(maxsize=self._outbox_size)
            self.status = Status.CONNECTED
            log.info("connected to %s", self._url)
            sender = asyncio.create_task(self._send_loop(ws, self._outbox))
            try:
                async for raw in ws:
                    self._dispatch(raw)
            except ConnectionClosed:
                pass
            finally:
                self.status = Status.DISCONNECTED
                self._outbox = None
                self._ws = None
                sender.cancel()
            return ws.close_code

    async def _send_loop(self, ws: ClientConnection, outbox: asyncio.Queue[str]) -> None:
        try:
            while True:
                await ws.send(await outbox.get())
        except ConnectionClosed:
            pass

    def _dispatch(self, raw: str | bytes) -> None:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("ignoring non-JSON message from server")
            return
        try:
            self._on_event(event)
        except Exception:
            log.exception("failed to handle %s event", event.get("type"))
