# 005 — Conversation Summarization (`conversation_summarizer.py`)

## Purpose

Summarizes rich tool-use messages into structured markdown. Extracts Goal, Intent, Outcome, What Happened, User Messages, Assistant Actions, Tool Usage, and Evaluation Criteria sections.

## API

### `summarize_conversation(messages, model=DEFAULT_MODEL, max_tokens=8192, system=None) → str`

1. Formats rich messages via `_format_rich_messages()` into a tool-aware transcript
2. Sends to LLM via `complete()` with the summary system prompt
3. Extracts markdown between delimiter tags via `_extract_summary()`

## Delimiter Tags

```
<|CONVERSATION_SUMMARY_SEA_START|>
... markdown ...
<|CONVERSATION_SUMMARY_SEA_END|>
```

## Output Sections

```markdown
## Goal          — What the user was trying to achieve (1-2 sentences)
## Intent        — Why they wanted this
## What Happened — Narrative of conversation flow, key decisions
## User Messages — Numbered summary of each user message
## Assistant Actions — Numbered summary of each assistant reply
## Tool Usage    — Key tool calls with brief inputs/results
## Outcome       — Final result, completion status
## Evaluation Criteria — Bullet list for judging success
```

## Rich Message Formatting

### `_format_rich_messages(messages: list[dict]) → str`

Handles tool calls with input (truncated to 200 chars) and output (truncated to 500 chars). Labels reasoning blocks. Format:
```
--- Message N (role) ---
[text content]

[tool: tool_name] — title (status)
  Input:
    key: value
  Output: truncated output...
```

## Tag Extraction

### `_extract_summary(response: str) → str`

Regex extraction between delimiter tags. Falls back to raw response if tags not found (with warning).

## Dependencies
- `src.llm.complete()` for LLM call

## Testing (`tests/test_conversation_summarizer.py` — 19 tests)

**Unit tests** (pure functions):
- `_format_rich_messages`: text-only messages, tool calls with input/output, tool with title, tool with status, reasoning blocks, mixed parts, message numbering, empty parts
- `_extract_summary`: correctly extracts between delimiter tags, multiline content, missing end tag fallback, missing start tag fallback

**Integration tests** (`@pytest.mark.redis`):
- `summarize_conversation` with pre-seeded cache: returns markdown with expected sections (Goal, Intent, Outcome), handles empty messages, handles system override
