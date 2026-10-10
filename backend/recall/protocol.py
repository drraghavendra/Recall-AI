from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class FramesMessage(BaseModel):
    """One sampling window: a burst of JPEG frames captured around `ts`."""

    type: Literal["frames"]
    ts: float
    frames: list[str] = Field(min_length=1, description="base64-encoded JPEG frames")


class AudioMessage(BaseModel):
    """A chunk of mic audio whose first sample was captured at `ts`."""

    type: Literal["audio"]
    ts: float
    pcm: str = Field(description="base64-encoded 16 kHz mono signed 16-bit little-endian PCM")


InboundMessage = Annotated[FramesMessage | AudioMessage, Field(discriminator="type")]
inbound_adapter: TypeAdapter[InboundMessage] = TypeAdapter(InboundMessage)
