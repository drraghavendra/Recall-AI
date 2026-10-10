from typing import Protocol

from pydantic import BaseModel, ConfigDict


class PersonaUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: list[str]
    nickname: str
    relation: str
    key_facts: list[str]
    summary: str


class PersonaUpdateError(Exception):
    pass


class PersonaWriter(Protocol):
    async def update(self, current: dict | None, lines: list[str]) -> PersonaUpdate:
        """Merge new lines spoken by the person into their current persona. Raises PersonaUpdateError."""
        ...
