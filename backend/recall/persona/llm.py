import json
import logging

import anthropic
from pydantic import ValidationError

from recall.persona.models import Persona, PersonaUpdate, PersonaUpdateError

log = logging.getLogger(__name__)

_LIST = {"type": "array", "items": {"type": "string"}}

# Mirrors PersonaUpdate; hand-written because structured outputs need additionalProperties=false
# and every field required.
PERSONA_SCHEMA = {
    "type": "object",
    "properties": {
        "changed": {"type": "boolean"},
        "persona": {
            "type": "object",
            "properties": {
                "names": _LIST,
                "nickname": {"type": "string"},
                "relation": {"type": "string"},
                "work": _LIST,
                "family": _LIST,
                "interests": _LIST,
                "plans": _LIST,
                "other": _LIST,
                "last_talked_about": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": [
                "names", "nickname", "relation", "work", "family", "interests", "plans", "other",
                "last_talked_about", "summary",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["changed", "persona"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You maintain a short, structured profile of a person the user talks to, built only from what that \
person said to the user. Every minute or two you receive the current profile (or none) and the new \
lines that person has spoken since the last update, in order. Decide whether the new lines add \
anything worth remembering, and return the full profile.

- changed: false if the new lines add nothing durable (small talk, filler, noise, or things already \
in the profile); then return the current profile unchanged. true if you added or corrected anything.
- names: names the person goes by, if stated. Keep earlier ones.
- nickname: a nickname if one is used, otherwise an empty string.
- relation: their relationship to the user (colleague, friend, daughter, neighbour, ...) if stated or \
clearly implied, otherwise "unknown".
- work, family, interests, plans, other: short facts in plain words, grouped by topic. Keep existing \
facts unless the new lines contradict them, drop duplicates, at most 5 per group.
- last_talked_about: one short line on what the most recent lines were about, e.g. "Her trip to Goa \
next month". Keep the current value if the new lines don't say anything substantive.
- summary: 1-2 warm, simple sentences the user could glance at right before talking to them again.

Refer to the person by name or as "they" unless they said which pronouns they use; never guess \
gender from a name. Lines come from speech recognition and may contain errors; ignore lines that \
are clearly noise. Never invent details that the lines do not support."""


class ClaudePersonaWriter:
    def __init__(self, model: str, effort: str):
        self._client = anthropic.AsyncAnthropic()
        self._model = model
        self._effort = effort

    async def update(self, current: Persona | None, lines: list[str]) -> PersonaUpdate:
        profile = "none" if current is None else json.dumps(current.model_dump(), indent=2)
        transcript = "\n".join(f"- {line}" for line in lines)
        content = (
            f"Current profile:\n{profile}\n\n"
            f"New lines from this person:\n<transcript>\n{transcript}\n</transcript>"
        )
        try:
            response = await self._client.beta.messages.create(
                model=self._model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
                output_config={
                    "effort": self._effort,
                    "format": {"type": "json_schema", "schema": PERSONA_SCHEMA},
                },
            )
        except anthropic.APIStatusError as e:
            raise PersonaUpdateError(f"Claude API error {e.status_code} (request {e.request_id})") from e
        except anthropic.APIConnectionError as e:
            raise PersonaUpdateError("could not reach the Claude API") from e

        if response.stop_reason != "end_turn":
            raise PersonaUpdateError(f"persona generation stopped with {response.stop_reason}")
        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise PersonaUpdateError("persona response had no text block")
        try:
            return PersonaUpdate.model_validate_json(text)
        except ValidationError as e:
            raise PersonaUpdateError("persona response did not match the schema") from e
