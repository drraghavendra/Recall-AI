from dataclasses import dataclass


def ws_url_for(server_url: str) -> str:
    base = server_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :] + "/ws"
    if base.startswith("http://"):
        return "ws://" + base[len("http://") :] + "/ws"
    raise ValueError(f"server URL must start with http:// or https://, got {server_url!r}")


@dataclass(frozen=True)
class ClientConfig:
    server_url: str = "http://localhost:8000"
    camera_index: int = 0
    window_s: float = 5.0
    frames_per_window: int = 1
    jpeg_width: int = 640
    jpeg_quality: int = 80
    sample_rate: int = 16000
    audio_block_s: float = 0.1
    persona_ttl_s: float = 210.0

    @property
    def ws_url(self) -> str:
        return ws_url_for(self.server_url)
