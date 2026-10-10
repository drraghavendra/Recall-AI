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
    saved_at REAL NOT NULL,
    score REAL NOT NULL DEFAULT 0
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
    ended_at REAL
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
CREATE INDEX IF NOT EXISTS idx_face_samples_ts ON face_samples(ts);
-- One row per person the persona job has looked at. `persona` is the structured Persona as JSON (NULL
-- until something worth remembering was said); `last_segment_id` is the last transcript line already
-- sent to Claude, so each line is considered exactly once.
CREATE TABLE IF NOT EXISTS profiles (
    face_id TEXT PRIMARY KEY,
    persona TEXT,
    last_segment_id INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);
-- Voice embeddings of lines credited to each face, to build that person's voiceprint.
CREATE TABLE IF NOT EXISTS voice_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    seconds REAL NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_voice_samples_face ON voice_samples(face_id, id);
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
        self._migrate_flat_personas()
        if "score" not in {row["name"] for row in self._conn.execute("PRAGMA table_info(face_portraits)")}:
            self._execute("ALTER TABLE face_portraits ADD COLUMN score REAL NOT NULL DEFAULT 0")

    def _migrate_flat_personas(self) -> None:
        """Earlier builds kept flat personas in a `personas` table; carry them into `profiles` once."""
        if not self._conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'personas'").fetchone():
            return
        conv_columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(conversations)")}
        with self._conn:
            for row in self._conn.execute("SELECT * FROM personas").fetchall():
                persona = {
                    "names": json.loads(row["names"]), "nickname": row["nickname"], "relation": row["relation"],
                    "work": [], "family": [], "interests": [], "plans": [], "other": json.loads(row["key_facts"]),
                    "last_talked_about": "", "summary": row["summary"],
                }
                # Lines the old per-conversation updates already used shouldn't be sent to Claude again.
                sent = 0
                if "persona_watermark" in conv_columns:
                    sent = self._conn.execute(
                        "SELECT COALESCE(MAX(persona_watermark), 0) FROM conversations WHERE face_id = ?",
                        (row["face_id"],),
                    ).fetchone()[0]
                self._conn.execute(
                    "INSERT OR IGNORE INTO profiles (face_id, persona, last_segment_id, updated_at) VALUES (?, ?, ?, ?)",
                    (row["face_id"], json.dumps(persona), sent, row["updated_at"]),
                )
            self._conn.execute("DROP TABLE personas")

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

    def save_face_portrait(self, face_id: str, image: bytes, score: float = 0.0) -> None:
        """Save the person's photo, replacing the current one."""
        self._execute(
            "INSERT INTO face_portraits(face_id, image, saved_at, score) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(face_id) DO UPDATE SET image = excluded.image, saved_at = excluded.saved_at, "
            "score = excluded.score",
            (face_id, image, time.time(), score),
        )

    def portrait_info(self, face_id: str) -> tuple[float, float] | None:
        """(score, saved_at) of the person's photo, or None if there isn't one."""
        row = self._conn.execute("SELECT score, saved_at FROM face_portraits WHERE face_id = ?", (face_id,)).fetchone()
        return None if row is None else (row["score"], row["saved_at"])

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

    def majority_face(self, start: float, end: float, pad_s: float = 0.5) -> str | None:
        """The face most often in focus between `start` and `end`, from the 1-per-second face samples."""
        row = self._conn.execute(
            "SELECT focused_face_id, COUNT(*) AS n FROM face_samples "
            "WHERE ts BETWEEN ? AND ? AND focused_face_id IS NOT NULL "
            "GROUP BY focused_face_id ORDER BY n DESC, MAX(ts) DESC LIMIT 1",
            (start - pad_s, end + pad_s),
        ).fetchone()
        return None if row is None else row["focused_face_id"]

    def pending_lines(self) -> dict[str, list[Segment]]:
        """Other people's transcript lines not yet sent to Claude, grouped by face (the face <> transcript map)."""
        rows = self._conn.execute(
            "SELECT s.id, s.face_id, s.ts_start, s.ts_end, s.text FROM transcript_segments s "
            "LEFT JOIN profiles p ON p.face_id = s.face_id "
            "WHERE s.speaker = 'other' AND s.face_id IS NOT NULL AND s.id > COALESCE(p.last_segment_id, 0) "
            "ORDER BY s.id"
        ).fetchall()
        lines: dict[str, list[Segment]] = {}
        for r in rows:
            lines.setdefault(r["face_id"], []).append(Segment(r["id"], r["ts_start"], r["ts_end"], r["text"]))
        return lines

    def get_persona(self, face_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT persona, updated_at FROM profiles WHERE face_id = ? AND persona IS NOT NULL", (face_id,)
        ).fetchone()
        if row is None:
            return None
        stats = self._conn.execute(
            "SELECT (SELECT COUNT(*) FROM conversations WHERE face_id = ?) AS times_met, "
            "(SELECT MAX(ts) FROM face_samples WHERE focused_face_id = ?) AS last_seen",
            (face_id, face_id),
        ).fetchone()
        return {**json.loads(row["persona"]), "updated_at": row["updated_at"], **dict(stats)}

    def save_persona(self, face_id: str, persona: dict | None, last_segment_id: int) -> None:
        """Advance the face's bookmark; also replace its persona unless `persona` is None (nothing new)."""
        self._execute(
            "INSERT INTO profiles (face_id, persona, last_segment_id, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(face_id) DO UPDATE SET last_segment_id = excluded.last_segment_id, "
            "persona = COALESCE(excluded.persona, profiles.persona), "
            "updated_at = CASE WHEN excluded.persona IS NULL THEN profiles.updated_at ELSE excluded.updated_at END",
            (face_id, None if persona is None else json.dumps(persona), last_segment_id, time.time()),
        )

    def add_voice_sample(self, face_id: str, embedding: bytes, seconds: float, keep: int) -> None:
        """Store a voice sample for `face_id`, keeping only its `keep` most recent."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO voice_samples (face_id, embedding, seconds, ts) VALUES (?, ?, ?, ?)",
                (face_id, embedding, seconds, time.time()),
            )
            self._conn.execute(
                "DELETE FROM voice_samples WHERE face_id = ? AND id NOT IN "
                "(SELECT id FROM voice_samples WHERE face_id = ? ORDER BY id DESC LIMIT ?)",
                (face_id, face_id, keep),
            )

    def voice_samples(self) -> dict[str, list[bytes]]:
        samples: dict[str, list[bytes]] = {}
        for row in self._conn.execute("SELECT face_id, embedding FROM voice_samples ORDER BY id"):
            samples.setdefault(row["face_id"], []).append(row["embedding"])
        return samples

    def get_self_voice(self) -> bytes | None:
        row = self._conn.execute("SELECT embedding FROM self_voice WHERE id = 1").fetchone()
        return None if row is None else row["embedding"]

    def set_self_voice(self, embedding: bytes) -> None:
        self._execute(
            "INSERT INTO self_voice (id, embedding, enrolled_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET embedding = excluded.embedding, enrolled_at = excluded.enrolled_at",
            (embedding, time.time()),
        )
