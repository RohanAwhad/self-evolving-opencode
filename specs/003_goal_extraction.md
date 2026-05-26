# 003 — Goal Extraction (`goal_extractor.py`)

## Purpose

Extracts user goals from an OpenCode conversation transcript via LLM forced tool-use. The LLM segments the conversation into logical threads and returns each as a structured goal.

## API

### `extract_goals(session_id, model=DEFAULT_MODEL, max_tokens=4096, db_path=None) → list[Goal]`

1. Fetches conversation transcript via `get_conversation_transcript(session_id)`
2. Sends transcript to LLM with forced tool-use schema `EXTRACT_GOALS_TOOL`
3. Returns parsed goal list

## Tool Schema

```python
EXTRACT_GOALS_TOOL = {
    "name": "extract_goals",
    "input_schema": {
        "goals": [{
            "title": "string",
            "description": "string",
            "message_range": "string (e.g. 'msgs 1-8')"
        }]
    }
}
```

## Data Type

```python
@dataclass
class Goal:
    title: str
    description: str
    message_range: str   # e.g. "msgs 1-8"
```

## System Prompt

Instructs the LLM to segment the conversation into logical user goals, providing a title, description, and the message range where the goal was discussed.

## Dependencies
- `src.llm.complete_tool()` for LLM call
- `src.opencode_db.get_conversation_transcript()` for transcript

## Testing (`tests/test_goal_extractor.py` — 4 tests)

**Integration tests only** (`@pytest.mark.redis` — requires DB fixture + Redis).

**Approach**: Pre-seed Redis cache with deterministic tool-call responses. Tests exercise real code paths (DB fetch → transcript → LLM call → parse) without hitting the real API.

**Coverage**:
- Single goal extraction: parses title, description, message_range correctly
- Multiple goals: returns all goals from transcript
- Empty transcript: returns empty list
- Missing message_range: defaults to empty string (graceful handling of incomplete LLM output)
