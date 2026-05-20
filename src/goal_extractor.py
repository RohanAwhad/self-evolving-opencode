"""Extract distinct goals/threads from an OpenCode conversation session."""

from dataclasses import dataclass
from pathlib import Path

from src.llm import DEFAULT_MODEL, complete_tool
from src.opencode_db import DB_PATH, get_conversation_transcript


@dataclass
class Goal:
    title: str
    description: str
    message_range: str  # e.g. "msgs 1-5"


EXTRACT_GOALS_TOOL = {
    "name": "extract_goals",
    "description": "Return the list of distinct goals identified in the conversation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "goals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short goal title"},
                        "description": {"type": "string", "description": "One-line description"},
                        "message_range": {"type": "string", "description": "e.g. 'msgs 1-8'"},
                    },
                    "required": ["title", "description", "message_range"],
                },
            }
        },
        "required": ["goals"],
    },
}


async def extract_goals(
    session_id: str,
    db_path: Path = DB_PATH,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> list[Goal]:
    """Call Claude via Vertex to extract goals from a conversation."""
    transcript = await get_conversation_transcript(session_id, db_path=db_path)

    result = await complete_tool(
        messages=[
            {
                "role": "user",
                "content": f"""Analyze this OpenCode conversation transcript and identify the distinct goals/tasks the user was trying to accomplish.

The conversation may contain multiple threads or phases, each pointing to a separate goal.

For each goal, provide:
1. A short title (title)
2. A one-line description (description)
3. Which messages (by number) relate to this goal (message_range, e.g. "msgs 1-8")

--- TRANSCRIPT ---
{transcript}""",
            }
        ],
        tool=EXTRACT_GOALS_TOOL,
        model=model,
        max_tokens=max_tokens,
    )

    return [
        Goal(
            title=g["title"],
            description=g["description"],
            message_range=g.get("message_range", ""),
        )
        for g in result["goals"]
    ]
