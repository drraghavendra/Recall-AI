import textwrap

import cv2
import numpy as np

from recall_client.connection import Status
from recall_client.state import ViewSnapshot

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREY = (190, 190, 190)
RED = (40, 40, 230)
AMBER = (0, 170, 255)
PANEL_WIDTH = 360


def display_name(face_id: str | None, persona: dict | None) -> str:
    if face_id is None:
        return "no one in view"
    if persona:
        if persona.get("nickname"):
            return persona["nickname"]
        if persona.get("names"):
            return persona["names"][0]
    return f"face {face_id[:6]}"


def persona_lines(snapshot: ViewSnapshot) -> list[tuple[str, tuple[int, int, int]]]:
    if snapshot.face_id is None:
        return []
    if snapshot.first_meeting:
        return [("You're meeting them", WHITE), ("for the first time", WHITE)]
    persona = snapshot.persona
    if persona is None:
        return [("Looking up...", GREY)]

    lines = [(display_name(snapshot.face_id, persona), WHITE)]
    names = persona.get("names") or []
    if len(names) > 1:
        lines.append(("aka " + ", ".join(names[1:]), GREY))
    if persona.get("relation") and persona["relation"] != "unknown":
        lines.append((persona["relation"], GREY))
    lines.append(("", WHITE))
    for fact in persona.get("key_facts") or []:
        lines.extend((line, WHITE) for line in textwrap.wrap("- " + fact, 34))
    if persona.get("summary"):
        lines.append(("", WHITE))
        lines.extend((line, GREY) for line in textwrap.wrap(persona["summary"], 36))
    return lines


def _text(frame: np.ndarray, text: str, origin: tuple[int, int], color, scale: float = 0.5) -> None:
    cv2.putText(frame, text, origin, FONT, scale, color, 1, cv2.LINE_AA)


def _shade(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    region = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = (region * 0.35).astype(frame.dtype)


def _label(frame: np.ndarray, text: str, origin: tuple[int, int], color, scale: float) -> None:
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, scale, 1)
    x, y = origin
    _shade(frame, x - 6, y - text_height - 6, x + text_width + 6, y + baseline + 4)
    _text(frame, text, origin, color, scale)


def draw(frame: np.ndarray, snapshot: ViewSnapshot, status: Status) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]

    _shade(out, 8, 8, 82, 36)
    cv2.circle(out, (22, 22), 8, RED, -1, cv2.LINE_AA)
    _text(out, "REC", (36, 28), RED, 0.6)

    if status is not Status.CONNECTED:
        _label(out, f"server: {status.value}", (100, 28), AMBER, 0.55)

    if snapshot.in_conversation:
        who = (
            "yourself / someone out of view"
            if snapshot.conversation_face_id is None
            else display_name(snapshot.conversation_face_id, snapshot.conversation_persona)
        )
        conversation = f"Talking with {who}"
    else:
        conversation = "Idle"
    _label(out, conversation, (14, height - 16), WHITE, 0.6)

    lines = persona_lines(snapshot)
    if lines:
        x0 = max(width - PANEL_WIDTH, 0)
        panel_height = min(24 + 20 * len(lines), height)
        _shade(out, x0, 0, width, panel_height)
        for i, (line, color) in enumerate(lines):
            _text(out, line, (x0 + 12, 24 + 20 * i), color, 0.5 if i else 0.65)
    return out
