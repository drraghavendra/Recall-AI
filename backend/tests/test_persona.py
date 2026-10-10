import asyncio

from recall.db import Database
from recall.persona.models import Persona, PersonaUpdate, PersonaUpdateError
from recall.persona.service import PersonaService


def persona(**fields) -> Persona:
    base = dict(
        names=[], nickname="", relation="unknown", work=[], family=[], interests=[], plans=[], other=[],
        last_talked_about="", summary="",
    )
    return Persona(**{**base, **fields})


class FakeWriter:
    """Adds every line containing "fact" to `other`; anything else is small talk (changed=False)."""

    def __init__(self, fail_first: bool = False):
        self.calls: list[tuple[str, list[str]]] = []
        self._fail_next = fail_first

    async def update(self, current, lines):
        if self._fail_next:
            self._fail_next = False
            raise PersonaUpdateError("simulated outage")
        name = (current.names if current else ["?"])[0]
        self.calls.append((name, lines))
        facts = [line for line in lines if "fact" in line]
        if not facts:
            return PersonaUpdate(changed=False, persona=current or persona())
        base = current or persona(names=[lines[0].split()[0]])
        return PersonaUpdate(changed=True, persona=base.model_copy(update={"other": base.other + facts}))


def setup(writer):
    db = Database(":memory:")
    conv_id = db.open_conversation("alice", 0.0)
    published: list[dict] = []
    service = PersonaService(db, published.append, writer, cycle_s=90)
    return db, conv_id, service, published


def test_each_line_reaches_claude_once_grouped_by_face():
    writer = FakeWriter()
    db, conv, service, published = setup(writer)

    async def scenario():
        db.insert_segment(conv, 0, 1, "Alice fact one", "other", "alice")
        db.insert_segment(conv, 1, 2, "Bob fact one", "other", "bob")
        db.insert_segment(conv, 2, 3, "my own words", "self", None)
        await service.run_once()
        db.insert_segment(conv, 3, 4, "fact two", "other", "alice")
        await service.run_once()
        await service.run_once()  # nothing new: no call

    asyncio.run(scenario())
    assert writer.calls == [("?", ["Alice fact one"]), ("?", ["Bob fact one"]), ("Alice", ["fact two"])]
    assert service.get("alice")["other"] == ["Alice fact one", "fact two"]
    assert service.get("alice")["key_facts"] == ["Alice fact one", "fact two"]
    assert [e["face_id"] for e in published] == ["alice", "bob", "alice"]


def test_small_talk_is_neither_saved_nor_pushed_but_is_not_resent():
    writer = FakeWriter()
    db, conv, service, published = setup(writer)

    async def scenario():
        db.insert_segment(conv, 0, 1, "hello there", "other", "alice")
        await service.run_once()
        await service.run_once()

    asyncio.run(scenario())
    assert writer.calls == [("?", ["hello there"])]
    assert service.get("alice") is None
    assert published == []


def test_failed_update_is_retried_with_the_same_lines():
    writer = FakeWriter(fail_first=True)
    db, conv, service, _ = setup(writer)

    async def scenario():
        db.insert_segment(conv, 0, 1, "Alice fact a", "other", "alice")
        await service.run_once()
        db.insert_segment(conv, 1, 2, "fact b", "other", "alice")
        await service.run_once()

    asyncio.run(scenario())
    assert writer.calls == [("?", ["Alice fact a", "fact b"])]


def test_line_goes_to_the_face_in_focus_for_most_of_it():
    db = Database(":memory:")
    for ts, face in [(10, "alice"), (11, "bob"), (12, "bob"), (13, "bob"), (14, None)]:
        db.insert_face_sample(ts, face, 1.0)
    assert db.majority_face(10.2, 13.8) == "bob"
    assert db.majority_face(30, 32) is None


def test_flat_personas_from_earlier_builds_are_migrated(tmp_path):
    import sqlite3

    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE conversations (conv_id INTEGER PRIMARY KEY, face_id TEXT, started_at REAL, ended_at REAL, "
        "persona_watermark INTEGER NOT NULL DEFAULT 0);"
        "CREATE TABLE personas (face_id TEXT PRIMARY KEY, names TEXT, nickname TEXT, relation TEXT, key_facts TEXT, "
        "summary TEXT, conversation_count INTEGER, updated_at REAL);"
        "INSERT INTO conversations VALUES (1, 'alice', 0, 1, 7);"
        """INSERT INTO personas VALUES ('alice', '["Alice"]', '', 'friend', '["Likes tea"]', 'A friend.', 1, 5.0);"""
    )
    old.commit()
    old.close()

    db = Database(path)
    persona = db.get_persona("alice")
    assert (persona["names"], persona["other"], persona["summary"], persona["relation"]) == (
        ["Alice"], ["Likes tea"], "A friend.", "friend"
    )
    assert db._conn.execute("SELECT last_segment_id FROM profiles").fetchone()[0] == 7
    assert Database(path).get_persona("alice") is not None  # idempotent
