import asyncio

from recall.db import Database
from recall.matching.matcher import Conversation
from recall.persona.models import PersonaUpdate, PersonaUpdateError
from recall.persona.service import PersonaService


class FakeWriter:
    def __init__(self, fail_first: bool = False):
        self.calls: list[list[str]] = []
        self._fail_next = fail_first

    async def update(self, current, lines):
        if self._fail_next:
            self._fail_next = False
            raise PersonaUpdateError("simulated outage")
        self.calls.append(lines)
        facts = (current or {}).get("key_facts", []) + lines
        return PersonaUpdate(names=["Alice"], nickname="", relation="unknown", key_facts=facts, summary="...")


def setup(writer):
    db = Database(":memory:")
    conv_id = db.open_conversation("alice", 0.0)
    conv = Conversation(conv_id, "alice", 0.0, 0.0)
    published: list[dict] = []
    service = PersonaService(db, published.append, writer, update_interval_s=300, min_new_segments=2)
    return db, conv, service, published


def test_each_segment_reaches_the_llm_exactly_once():
    writer = FakeWriter()
    db, conv, service, published = setup(writer)

    async def scenario():
        db.insert_segment(conv.conv_id, 0, 1, "a", "other", "alice")
        await service.checkpoint(conv)
        db.insert_segment(conv.conv_id, 1, 2, "b", "other", "alice")
        db.insert_segment(conv.conv_id, 2, 3, "my own words", "self", None)
        await service.checkpoint(conv)
        db.insert_segment(conv.conv_id, 3, 4, "c", "other", "alice")
        await service.finalize(conv)

    asyncio.run(scenario())
    assert writer.calls == [["a", "b"], ["c"]]
    persona = db.get_persona("alice")
    assert persona["key_facts"] == ["a", "b", "c"]
    assert persona["conversation_count"] == 1
    assert [e["type"] for e in published] == ["persona_updated", "persona_updated"]


def test_failed_update_keeps_segments_for_the_next_run():
    writer = FakeWriter(fail_first=True)
    db, conv, service, _ = setup(writer)

    async def scenario():
        db.insert_segment(conv.conv_id, 0, 1, "a", "other", "alice")
        db.insert_segment(conv.conv_id, 1, 2, "b", "other", "alice")
        await service.checkpoint(conv)
        await service.finalize(conv)

    asyncio.run(scenario())
    assert writer.calls == [["a", "b"]]
    assert db.get_persona("alice")["conversation_count"] == 1
