import json
import logging

import anthropic
from pydantic import ValidationError

from recall.persona.models import PersonaUpdate, PersonaUpdateError

log = logging.getLogger(__name__)

# Mirrors PersonaUpdate; hand-written because structured outputs need additionalProperties=false
# and every field required.
PERSONA_SCHEMA = {
    "type": "object",
    "properties": {
        "names": {"type": "array", "items": {"type": "string"}},
        "nickname": {"type": "string"},
        "relation": {"type": "string"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["names", "nickname", "relation", "key_facts", "summary"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You maintain a short profile of a person the user talks to. The profile is built only from what \
that person said to the user. You receive the current profile (or none) and new lines spoken by \
that person, in order, and return the full updated profile.

- names: names the person goes by, if stated. Keep earlier ones.
- nickname: a nickname if one is used, otherwise an empty string.
- relation: their relationship to the user (colleague, friend, family, ...) if stated or clearly \
implied, otherwise "unknown".
- key_facts: short, durable facts about them (work, interests, plans, preferences). Keep existing \
facts unless the new lines contradict them, drop duplicates, at most 12.
- summary: 2-3 sentences the user could glance at right before talking to them again.

Lines come from speech recognition and may contain errors; ignore lines that are clearly noise. \
Never invent details that the lines do not support."""


class ClaudePersonaWriter:
    def __init__(self, model: str, effort: str):
        self._client = anthropic.AsyncAnthropic()
        self._model = model
        self._effort = effort

    async def update(self, current: dict | None, lines: list[str]) -> PersonaUpdate:
        profile = "none" if current is None else json.dumps(
            {key: current[key] for key in PERSONA_SCHEMA["properties"]}, indent=2
        )
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
