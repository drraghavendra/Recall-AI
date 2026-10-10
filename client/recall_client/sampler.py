from collections.abc import Callable

from recall_client.messages import frames_message


class FrameWindow:
    """Spreads `frames_per_window` captures evenly across each `window_s` window.

    Call `step` once per camera frame; `grab` is only invoked when a sample is due, so JPEG
    encoding happens N times per window rather than for every frame. Returns the finished
    window's message (ts = window start) when a window closes.
    """

    def __init__(self, window_s: float, frames_per_window: int):
        if frames_per_window < 1:
            raise ValueError("frames_per_window must be >= 1")
        self._window_s = window_s
        self._frames_per_window = frames_per_window
        self._interval = window_s / frames_per_window
        self._start: float | None = None
        self._frames: list[bytes] = []

    def step(self, now: float, grab: Callable[[], bytes]) -> str | None:
        message = None
        if self._start is None:
            self._start = now
        elif now >= self._start + self._window_s:
            if self._frames:
                message = frames_message(self._start, self._frames)
            self._start, self._frames = now, []

        due_at = self._start + len(self._frames) * self._interval
        if len(self._frames) < self._frames_per_window and now >= due_at:
            self._frames.append(grab())
        return message
