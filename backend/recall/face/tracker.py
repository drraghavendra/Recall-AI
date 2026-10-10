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


class PresenceTracker:
    """Which known faces are confirmed in view, for the UI's green/red dots.

    Like FocusedFaceTracker but for every face in the frame: a face counts as present after
    `confirm_windows` consecutive windows and as gone after `absent_windows` consecutive windows
    without it, so one blurry or covered frame never changes what's shown.
    """

    def __init__(self, confirm_windows: int, absent_windows: int):
        self._confirm_windows = confirm_windows
        self._absent_windows = absent_windows
        self.present: set[str] = set()
        self._seen_streak: dict[str, int] = {}
        self._missing_streak: dict[str, int] = {}

    def observe(self, seen: set[str]) -> bool:
        """Record one window's faces. Returns True if the confirmed set changed."""
        before = set(self.present)
        for face_id in seen:
            self._missing_streak.pop(face_id, None)
            self._seen_streak[face_id] = self._seen_streak.get(face_id, 0) + 1
            if self._seen_streak[face_id] >= self._confirm_windows:
                self.present.add(face_id)
        for face_id in set(self._seen_streak) - seen:
            self._seen_streak.pop(face_id)
        for face_id in self.present - seen:
            self._missing_streak[face_id] = self._missing_streak.get(face_id, 0) + 1
            if self._missing_streak[face_id] >= self._absent_windows:
                self.present.discard(face_id)
                self._missing_streak.pop(face_id)
        return self.present != before
