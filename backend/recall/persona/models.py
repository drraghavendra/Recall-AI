from typing import Protocol

from pydantic import BaseModel, ConfigDict

FACT_GROUPS = ("work", "family", "interests", "plans", "other")


class Persona(BaseModel):
    """A structured profile of one person, built only from what they said."""

    model_config = ConfigDict(extra="forbid")

    names: list[str]
    nickname: str
    relation: str
    work: list[str]
    family: list[str]
    interests: list[str]
    plans: list[str]
    other: list[str]
    last_talked_about: str
    summary: str

    def facts(self) -> list[str]:
        return [fact for group in FACT_GROUPS for fact in getattr(self, group)]

    def is_good(self) -> bool:
        """Worth showing on a card: we know their name or at least one fact about them."""
        return bool(self.names or self.facts())


class PersonaUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    changed: bool
    persona: Persona


class PersonaUpdateError(Exception):
    pass


class PersonaWriter(Protocol):
    async def update(self, current: Persona | None, lines: list[str]) -> PersonaUpdate:
        """Decide whether new lines spoken by the person add anything to their persona. Raises PersonaUpdateError."""
        ...
