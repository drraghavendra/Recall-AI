from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class FramesMessage(BaseModel):
    """One sampling window: a burst of JPEG frames captured around `ts`."""

    type: Literal["frames"]
    ts: float
    frames: list[str] = Field(min_length=1, description="base64-encoded JPEG frames")
    store_portrait: bool = False


class AudioMessage(BaseModel):
    """A chunk of mic audio whose first sample was captured at `ts`."""

    type: Literal["audio"]
    ts: float
    pcm: str = Field(description="base64-encoded 16 kHz mono signed 16-bit little-endian PCM")


class LipFramesMessage(BaseModel):
    """One frame from a short burst (~5 fps) sent while the client hears sound, for the lip check."""

    type: Literal["lip_frames"]
    ts: float
    frame: str = Field(description="base64-encoded JPEG")


InboundMessage = Annotated[FramesMessage | AudioMessage | LipFramesMessage, Field(discriminator="type")]
inbound_adapter: TypeAdapter[InboundMessage] = TypeAdapter(InboundMessage)
