import asyncio
import base64
import binascii
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from recall.audio.pcm import pcm16_to_float
from recall.config import Settings
from recall.protocol import FramesMessage, LipFramesMessage, inbound_adapter
from recall.providers import ProviderChoice, ProviderError
from recall.wiring import Services, build_services

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("recall")


@asynccontextmanager
async def lifespan(app: FastAPI):
    services = build_services(Settings())
    services.start()
    app.state.services = services
    try:
        yield
    finally:
        await services.stop()


app = FastAPI(title="Recall-AI backend", lifespan=lifespan)
WEB_DIR = Path(__file__).with_name("web")
app.mount("/web-assets", StaticFiles(directory=WEB_DIR), name="web-assets")


@app.get("/", include_in_schema=False)
async def web_client() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


def _services(holder: Request | WebSocket) -> Services:
    return holder.app.state.services


@app.get("/health")
async def health(request: Request) -> dict:
    services = _services(request)
    return {
        "status": "ok",
        **services.providers.current,
        "lip_check": services.lips is not None,
        "voice_id": services.voices is not None,
    }


@app.get("/providers")
async def get_providers(request: Request) -> dict:
    providers = _services(request).providers
    return {**providers.current, "options": providers.options()}


@app.put("/providers")
async def switch_providers(choice: ProviderChoice, request: Request) -> dict:
    """Switch the live face and/or speech-to-text provider. The first switch to a provider loads its model."""
    providers = _services(request).providers
    try:
        await providers.switch(choice)
    except ProviderError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
    return {**providers.current, "options": providers.options()}


@app.get("/persona/{face_id}")
async def get_persona(face_id: str, request: Request) -> dict:
    persona = _services(request).persona.get(face_id)
    if persona is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no persona yet - first meeting")
    return persona


@app.get("/faces/{face_id}/portrait", include_in_schema=False)
async def get_face_portrait(face_id: str, request: Request) -> Response:
    image = _services(request).db.get_face_portrait(face_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no saved portrait for this face")
    return Response(content=image, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=60"})


@app.delete("/faces/{face_id}/portrait", status_code=status.HTTP_204_NO_CONTENT)
async def delete_face_portrait(face_id: str, request: Request) -> None:
    if not _services(request).db.delete_face_portrait(face_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no saved portrait for this face")


class WearerChoice(BaseModel):
    face_id: str | None


@app.get("/wearer")
async def get_wearer(request: Request) -> dict:
    """Which face is the user themselves (laptop demo: the user faces the camera), if marked."""
    return {"face_id": _services(request).db.get_wearer_face()}


@app.put("/wearer")
async def set_wearer(choice: WearerChoice, request: Request) -> dict:
    services = _services(request)
    services.db.set_wearer_face(choice.face_id)
    services.hub.publish({"type": "wearer_changed", "face_id": choice.face_id})
    return {"face_id": choice.face_id}


@app.get("/self-voice")
async def self_voice_status(request: Request) -> dict:
    return {"enrolled": _services(request).db.get_self_voice() is not None}


@app.post("/self-voice", status_code=status.HTTP_204_NO_CONTENT)
async def enroll_self_voice(request: Request) -> None:
    """Body: raw 16 kHz mono s16le PCM of the user speaking (a few seconds)."""
    services = _services(request)
    body = await request.body()
    min_samples = int(services.settings.min_self_voice_enroll_s * services.settings.sample_rate)
    if len(body) % 2 or len(body) // 2 < min_samples:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"need at least {services.settings.min_self_voice_enroll_s}s of 16 kHz mono s16le PCM",
        )
    embedding = await asyncio.to_thread(services.self_voice.encoder.embed, pcm16_to_float(body))
    services.db.set_self_voice(embedding.astype(np.float32).tobytes())
    services.self_voice.set_profile(embedding)


@app.websocket("/ws")
async def capture_socket(ws: WebSocket) -> None:
    """Capture client: sends frames + audio, receives face/conversation/persona events."""
    services = _services(ws)
    await ws.accept()
    if services.capture_client_connected:
        await ws.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="another capture client is connected")
        return

    services.capture_client_connected = True
    events = services.hub.subscribe()
    sender = asyncio.create_task(_forward_events(ws, events))
    try:
        while True:
            _route(services, await ws.receive_text())
    except WebSocketDisconnect:
        log.info("capture client disconnected")
    finally:
        sender.cancel()
        services.hub.unsubscribe(events)
        services.audio.reset()
        services.capture_client_connected = False


async def _forward_events(ws: WebSocket, events: asyncio.Queue[dict]) -> None:
    try:
        while True:
            await ws.send_json(await events.get())
    except (WebSocketDisconnect, RuntimeError):
        pass


def _route(services: Services, raw: str) -> None:
    try:
        message = inbound_adapter.validate_json(raw)
        if isinstance(message, FramesMessage):
            services.face.submit(
                message.ts,
                [base64.b64decode(f, validate=True) for f in message.frames],
                store_portrait=message.store_portrait,
            )
            return
        if isinstance(message, LipFramesMessage):
            if services.lips is not None:
                services.lips.submit(message.ts, base64.b64decode(message.frame, validate=True))
            return
        pcm = base64.b64decode(message.pcm, validate=True)
    except (ValidationError, binascii.Error) as e:
        log.warning("dropping malformed message: %s", e)
        return
    if len(pcm) % 2:
        log.warning("dropping audio chunk with odd byte length")
        return
    services.audio.submit(message.ts, pcm)
