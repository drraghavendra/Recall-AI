import json
import urllib.error
import urllib.parse
import urllib.request


class ApiError(Exception):
    pass


class RecallApi:
    """Blocking HTTP calls to the backend. Run off the event loop / UI thread."""

    def __init__(self, server_url: str, timeout_s: float = 5.0):
        self._base = server_url.rstrip("/")
        self._timeout_s = timeout_s

    def get_persona(self, face_id: str) -> dict | None:
        """None means no persona yet (first meeting)."""
        url = f"{self._base}/persona/{urllib.parse.quote(face_id, safe='')}"
        try:
            with urllib.request.urlopen(url, timeout=self._timeout_s) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise ApiError(f"GET {url} -> HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise ApiError(f"GET {url} failed: {e}") from e

    def enroll_self_voice(self, pcm: bytes) -> None:
        request = urllib.request.Request(
            f"{self._base}/self-voice",
            data=pcm,
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                response.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise ApiError(f"POST /self-voice -> HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise ApiError(f"POST /self-voice failed: {e}") from e
