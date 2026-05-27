# 001 — OpenCode DB Access (`opencode_db.py`)

## Purpose

All SQLite access to OpenCode's database. Wraps sync sqlite3 calls in `asyncio.to_thread()` for non-blocking I/O.

## Database Path

Default: `~/.local/share/opencode/opencode.db` (overridable via `db_path` kwarg)

## Tables Queried

### `session`
- `id`, `title`, `directory`, `agent`, `model` (JSON string), `cost`, `tokens_input`, `tokens_output`, `time_created`, `time_updated`

### `message`
- `id`, `session_id` (FK), `time_created`, `data` (JSON with `role` key)

### `part`
- `id`, `message_id` (FK), `time_created`, `data` (JSON with `type`: `text`, `tool`, `reasoning`)

## API

### `get_sessions(limit=30, db_path=None) → list[Session]`
Returns sessions ordered by `time_updated DESC`. LEFT JOINs with `message` table for `message_count`.

### `get_messages_for_session(session_id, db_path=None) → list[dict]`
Returns simple messages: `[{"role": "user"|"assistant", "content": "..."}]`. Text parts concatenated, tool parts formatted as `[tool: name]`.

### `get_rich_messages_for_session(session_id, db_path=None) → list[dict]`
Returns structured messages: `[{"role": "user"|"assistant", "parts": [{"type": "text"|"tool"|"reasoning", ...}]}]`. Preserves tool input/output, reasoning blocks.

### `get_conversation_transcript(session_id, db_path=None) → str`
Returns formatted transcript: `--- Message 1 (user) ---\n<content>\n\n--- Message 2 (assistant) ---\n<content>`. Combines text parts, labels tool calls.

### `parse_message_range(range_str: str) → tuple[int, int]`
Parses `"msgs 1-8"` → `(0, 8)` (0-indexed). Falls back to `(0, 999999)` for invalid input.

### `slice_messages(messages: list[dict], message_range: str) → list[dict]`
Slices a message list by range string. Handles out-of-bounds ranges.

### `get_skills_for_session(session_id, db_path=None) → list[str]`

Returns distinct skill names invoked during a session. Queries `tool:skill` parts in the part table.

```sql
SELECT DISTINCT json_extract(data, '$.state.input.name')
FROM part
WHERE json_extract(data, '$.type') = 'tool'
  AND json_extract(data, '$.tool') = 'skill'
  AND message_id IN (SELECT id FROM message WHERE session_id = ?)
```

## Data Type

```python
@dataclass
class Session:
    id: str
    title: str
    directory: str
    agent: str
    model_id: str          # extracted from JSON model field
    cost: float
    tokens_input: int
    tokens_output: int
    time_created: str
    time_updated: str
    message_count: int     # from LEFT JOIN with message table
```

## Testing (`tests/test_opencode_db.py` — 30 tests)

**No external dependencies** — pure SQLite, fastest test file.

**Fixture**: `db_path` — tmp SQLite with 5 sessions (s1-s5), 11 messages, 12 parts. Seeds edge cases: session with NULL fields (s3), session with 0 messages (s4), message with 0 parts (m11), message with tool+text parts (m4), multi-part text (m2).

**Coverage**:
- `get_sessions`: all sessions, `limit` param, ordering by `time_updated DESC`, model JSON parsing (valid & malformed), NULL cost/tokens handling
- `get_messages_for_session`: text parts, tool parts formatted as `[tool: name]`, empty session, ordering
- `get_rich_messages_for_session`: tool parts with input/output/status, reasoning blocks, text parts
- `get_conversation_transcript`: format `--- Message N (role) ---`, multi-part concatenation, empty session
- `parse_message_range`: `"msgs 1-8"` → `(0,8)`, `"msgs 3-3"`, single number, garbage input fallback
- `slice_messages`: correct slice, empty list, out-of-bounds range
- `get_skills_for_session`: session with skill invocations, session with none, deduplication
