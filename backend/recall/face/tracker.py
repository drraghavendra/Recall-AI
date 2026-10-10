import bisect


class FocusedFaceTracker:
    """Debounces per-window face observations and keeps a timeline of confirmed changes.

    A new face must be seen in `confirm_windows` consecutive windows (absence: `absent_confirm_windows`)
    before it becomes current. The change is timestamped at the first window it was seen, so audio
    can later ask "who was in focus at time t?" via `face_at`.
    """

    def __init__(self, confirm_windows: int, absent_confirm_windows: int):
        self._confirm_windows = confirm_windows
        self._absent_confirm_windows = absent_confirm_windows
        self.current: str | None = None
        self._candidate: str | None = None
        self._candidate_since = 0.0
        self._candidate_count = 0
        self._change_times: list[float] = [float("-inf")]
        self._change_faces: list[str | None] = [None]

    def observe(self, ts: float, face_id: str | None) -> bool:
        """Record one window's focused face. Returns True if the confirmed face changed."""
        if face_id == self.current:
            self._candidate_count = 0
            return False
        if self._candidate_count and face_id == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_since, self._candidate_count = face_id, ts, 1
        needed = self._absent_confirm_windows if face_id is None else self._confirm_windows
        if self._candidate_count < needed:
            return False
        self.current = face_id
        self._change_times.append(self._candidate_since)
        self._change_faces.append(face_id)
        self._candidate_count = 0
        return True

    def change_at(self, ts: float) -> tuple[float, str | None]:
        """The latest confirmed change at or before `ts`, as (change_time, face_id)."""
        i = bisect.bisect_right(self._change_times, ts) - 1
        return self._change_times[i], self._change_faces[i]

    def face_at(self, ts: float) -> str | None:
        return self.change_at(ts)[1]
