# 004 — Goal Checking (`goal_checker.py`)

## Purpose

Evaluates whether a user's goal was achieved based on the conversation messages. LLM reviews the thread and returns a verdict with reasoning.

## API

### `check_goal_achieved(messages, goal, model=DEFAULT_MODEL, max_tokens=1024) → GoalResult`

1. Formats messages into a readable transcript via `_format_messages_for_prompt()`
2. Sends to LLM with forced tool-use schema `GOAL_RESULT_TOOL`
3. Returns structured result

## Tool Schema

```python
GOAL_RESULT_TOOL = {
    "name": "report_goal_result",
    "input_schema": {
        "achieved": "boolean",
        "reasoning": "string (1-2 sentences)"
    }
}
```

## Data Type

```python
@dataclass
class GoalResult:
    achieved: bool
    reasoning: str
```

## Message Formatting

### `_format_messages_for_prompt(messages: list[dict]) → str`

Handles both string content and structured content blocks (`[{"type": "text", "text": "..."}]`). Output format: `--- Message N (role) ---\n<content>`

## System Prompt

Evaluates based on evidence in the messages only. No speculation beyond what is shown.

## Testing (`tests/test_goal_checker.py` — 13 tests)

**Unit tests** (pure functions, no external deps):
- `_format_messages_for_prompt`: string content, list content with `{"type": "text", "text": "..."}`, plain strings in list, empty messages, mixed roles, empty content, message numbering

**Integration tests** (`@pytest.mark.redis`):
- `check_goal_achieved` with pre-seeded cache: achieved=true case, achieved=false case, mixture of achieved/not achieved
- Returns `GoalResult` with correct types (`bool` + `str`)
- Uses `tests/fixtures/goal_checker_responses.json` for pre-recorded LLM responses

## Dependencies
- `src.llm.complete_tool()` for LLM call
