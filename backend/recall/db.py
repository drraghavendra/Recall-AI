import json
import re
import sqlite3
import time
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    face_id TEXT PRIMARY KEY,
    provider_face_id TEXT NOT NULL,
    enrolled_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS face_portraits (
    face_id TEXT PRIMARY KEY REFERENCES faces(face_id) ON DELETE CASCADE,
    image BLOB NOT NULL,
    saved_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS face_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    focused_face_id TEXT,
    confidence REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    conv_id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    persona_watermark INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS transcript_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id INTEGER NOT NULL REFERENCES conversations(conv_id),
    ts_start REAL NOT NULL,
    ts_end REAL NOT NULL,
    text TEXT NOT NULL,
    speaker TEXT NOT NULL CHECK (speaker IN ('self', 'other')),
    face_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_segments_conv ON transcript_segments(conv_id, id);
CREATE TABLE IF NOT EXISTS personas (
    face_id TEXT PRIMARY KEY,
    names TEXT NOT NULL,
    nickname TEXT NOT NULL,
    relation TEXT NOT NULL,
    key_facts TEXT NOT NULL,
    summary TEXT NOT NULL,
    conversation_count INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS self_voice (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    embedding BLOB NOT NULL,
    enrolled_at REAL NOT NULL
);
"""


@dataclass(frozen=True)
class Segment:
    id: int
    ts_start: float
    ts_end: float
    text: str


class Database:
    """Thin SQLite wrapper. Must only be used from the event-loop thread."""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._conn:
            return self._conn.execute(sql, params)

    def insert_face(self, face_id: str, provider_face_id: str) -> None:
        self._execute(
            "INSERT INTO faces (face_id, provider_face_id, enrolled_at) VALUES (?, ?, ?)",
            (face_id, provider_face_id, time.time()),
        )

    def next_face_id(self) -> str:
        """Allocate the next stable, human-readable ID for a newly seen person."""
        rows = self._conn.execute("SELECT face_id FROM faces").fetchall()
        numbers = [int(match.group(1)) for row in rows if (match := re.fullmatch(r"person_(\d+)", row["face_id"]))]
        return f"person_{max(numbers, default=0) + 1:03d}"

    def save_face_portrait(self, face_id: str, image: bytes) -> None:
        self._execute(
            "INSERT OR IGNORE INTO face_portraits(face_id, image, saved_at) VALUES (?, ?, ?)",
            (face_id, image, time.time()),
        )

    def get_face_portrait(self, face_id: str) -> bytes | None:
        row = self._conn.execute("SELECT image FROM face_portraits WHERE face_id = ?", (face_id,)).fetchone()
        return None if row is None else row["image"]

    def delete_face_portrait(self, face_id: str) -> bool:
        cur = self._execute("DELETE FROM face_portraits WHERE face_id = ?", (face_id,))
        return cur.rowcount > 0

    def insert_face_sample(self, ts: float, focused_face_id: str | None, confidence: float) -> None:
        self._execute(
            "INSERT INTO face_samples (ts, focused_face_id, confidence) VALUES (?, ?, ?)",
            (ts, focused_face_id, confidence),
        )

    def open_conversation(self, face_id: str | None, started_at: float) -> int:
        cur = self._execute(
            "INSERT INTO conversations (face_id, started_at) VALUES (?, ?)", (face_id, started_at)
        )
        return cur.lastrowid

    def close_conversation(self, conv_id: int, ended_at: float) -> None:
        self._execute("UPDATE conversations SET ended_at = ? WHERE conv_id = ?", (ended_at, conv_id))

    def insert_segment(
        self, conv_id: int, ts_start: float, ts_end: float, text: str, speaker: str, face_id: str | None
    ) -> int:
        cur = self._execute(
            "INSERT INTO transcript_segments (conv_id, ts_start, ts_end, text, speaker, face_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, ts_start, ts_end, text, speaker, face_id),
        )
        return cur.lastrowid

    def get_watermark(self, conv_id: int) -> int:
        row = self._conn.execute(
            "SELECT persona_watermark FROM conversations WHERE conv_id = ?", (conv_id,)
        ).fetchone()
        return row["persona_watermark"]

    def set_watermark(self, conv_id: int, segment_id: int) -> None:
        self._execute(
            "UPDATE conversations SET persona_watermark = ? WHERE conv_id = ?", (segment_id, conv_id)
        )

    def other_segments_after(self, conv_id: int, after_id: int) -> list[Segment]:
        rows = self._conn.execute(
            "SELECT id, ts_start, ts_end, text FROM transcript_segments "
            "WHERE conv_id = ? AND id > ? AND speaker = 'other' ORDER BY id",
            (conv_id, after_id),
        ).fetchall()
        return [Segment(r["id"], r["ts_start"], r["ts_end"], r["text"]) for r in rows]

    def get_persona(self, face_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM personas WHERE face_id = ?", (face_id,)).fetchone()
        if row is None:
            return None
        persona = dict(row)
        persona["names"] = json.loads(persona["names"])
        persona["key_facts"] = json.loads(persona["key_facts"])
        return persona

    def upsert_persona(self, face_id: str, fields: dict, count_conversation: bool) -> None:
        increment = 1 if count_conversation else 0
        self._execute(
            "INSERT INTO personas "
            "(face_id, names, nickname, relation, key_facts, summary, conversation_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(face_id) DO UPDATE SET "
            "names = excluded.names, nickname = excluded.nickname, relation = excluded.relation, "
            "key_facts = excluded.key_facts, summary = excluded.summary, "
            "conversation_count = personas.conversation_count + ?, updated_at = excluded.updated_at",
            (
                face_id,
                json.dumps(fields["names"]),
                fields["nickname"],
                fields["relation"],
                json.dumps(fields["key_facts"]),
                fields["summary"],
                increment,
                time.time(),
                increment,
            ),
        )

    def increment_conversation_count(self, face_id: str) -> None:
        self._execute(
            "UPDATE personas SET conversation_count = conversation_count + 1 WHERE face_id = ?",
            (face_id,),
        )

    def get_self_voice(self) -> bytes | None:
        row = self._conn.execute("SELECT embedding FROM self_voice WHERE id = 1").fetchone()
        return None if row is None else row["embedding"]

    def set_self_voice(self, embedding: bytes) -> None:
        self._execute(
            "INSERT INTO self_voice (id, embedding, enrolled_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET embedding = excluded.embedding, enrolled_at = excluded.enrolled_at",
            (embedding, time.time()),
        )
