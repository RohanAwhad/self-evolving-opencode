"""Shared test fixtures: SQLite DB factory, Redis connection, markers."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest
import redis.asyncio as aioredis


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "redis: requires Redis on localhost:6380")
    config.addinivalue_line("markers", "live: hits real LLM API (skipped by default)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_live = pytest.mark.skip(reason="live LLM tests not requested (pass -m live)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


# ---------------------------------------------------------------------------
# SQLite DB factory
# ---------------------------------------------------------------------------

# Schema matching OpenCode's real tables
_SCHEMA = """
CREATE TABLE session (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    directory   TEXT,
    agent       TEXT,
    model       TEXT,
    cost        REAL,
    tokens_input  INTEGER,
    tokens_output INTEGER,
    time_created  TEXT,
    time_updated  TEXT
);

CREATE TABLE message (
    id           TEXT PRIMARY KEY,
    session_id   TEXT,
    data         TEXT,
    time_created TEXT,
    FOREIGN KEY (session_id) REFERENCES session(id)
);

CREATE TABLE part (
    id           TEXT PRIMARY KEY,
    message_id   TEXT,
    data         TEXT,
    time_created TEXT,
    FOREIGN KEY (message_id) REFERENCES message(id)
);
"""

# Seed data: 5 sessions, ~20 messages, ~40 parts
_SESSIONS = [
    ("s1", "Fix login bug", "/home/user/app", "coder", '{"id":"claude-opus-4-20250514"}', 0.05, 1000, 500, "2025-01-01T10:00:00Z", "2025-01-01T10:30:00Z"),
    ("s2", "Add dark mode", "/home/user/app", "coder", '{"id":"claude-sonnet-4-20250514"}', 0.12, 3000, 1500, "2025-01-02T09:00:00Z", "2025-01-02T09:45:00Z"),
    ("s3", None, "/home/user/lib", "reviewer", "not-valid-json", 0.0, 0, 0, "2025-01-03T08:00:00Z", "2025-01-03T08:01:00Z"),
    ("s4", "Refactor DB layer", "/home/user/app", "coder", '{"id":"claude-opus-4-20250514"}', 1.23, 50000, 25000, "2025-01-04T14:00:00Z", "2025-01-04T15:00:00Z"),
    ("s5", "Empty session", "/home/user/app", "coder", '{"id":"claude-opus-4-20250514"}', None, None, None, "2025-01-05T12:00:00Z", "2025-01-05T12:00:00Z"),
]

_MESSAGES = [
    # s1: 4 messages
    ("m1", "s1", '{"role":"user"}',      "2025-01-01T10:00:01Z"),
    ("m2", "s1", '{"role":"assistant"}',  "2025-01-01T10:00:02Z"),
    ("m3", "s1", '{"role":"user"}',      "2025-01-01T10:00:03Z"),
    ("m4", "s1", '{"role":"assistant"}',  "2025-01-01T10:00:04Z"),
    # s2: 6 messages
    ("m5", "s2", '{"role":"user"}',      "2025-01-02T09:00:01Z"),
    ("m6", "s2", '{"role":"assistant"}',  "2025-01-02T09:00:02Z"),
    ("m7", "s2", '{"role":"user"}',      "2025-01-02T09:00:03Z"),
    ("m8", "s2", '{"role":"assistant"}',  "2025-01-02T09:00:04Z"),
    ("m9", "s2", '{"role":"user"}',      "2025-01-02T09:00:05Z"),
    ("m10", "s2", '{"role":"assistant"}', "2025-01-02T09:00:06Z"),
    # s3: 2 messages
    ("m11", "s3", '{"role":"user"}',      "2025-01-03T08:00:01Z"),
    ("m12", "s3", '{"role":"assistant"}',  "2025-01-03T08:00:02Z"),
    # s4: 8 messages
    ("m13", "s4", '{"role":"user"}',      "2025-01-04T14:00:01Z"),
    ("m14", "s4", '{"role":"assistant"}',  "2025-01-04T14:00:02Z"),
    ("m15", "s4", '{"role":"user"}',      "2025-01-04T14:00:03Z"),
    ("m16", "s4", '{"role":"assistant"}',  "2025-01-04T14:00:04Z"),
    ("m17", "s4", '{"role":"user"}',      "2025-01-04T14:00:05Z"),
    ("m18", "s4", '{"role":"assistant"}',  "2025-01-04T14:00:06Z"),
    ("m19", "s4", '{"role":"user"}',      "2025-01-04T14:00:07Z"),
    ("m20", "s4", '{"role":"assistant"}',  "2025-01-04T14:00:08Z"),
    # s5: 0 messages (empty session)
]

def _part(pid: str, mid: str, pdata: dict, ts: str) -> tuple[str, str, str, str]:
    return (pid, mid, json.dumps(pdata), ts)

_PARTS = [
    # m1: user text
    _part("p1",  "m1", {"type": "text", "text": "Fix the login bug please"}, "2025-01-01T10:00:01Z"),
    # m2: assistant text + tool
    _part("p2",  "m2", {"type": "text", "text": "I'll look into the login issue"}, "2025-01-01T10:00:02Z"),
    _part("p3",  "m2", {"type": "tool", "tool": "read_file", "state": {"status": "done", "input": {"path": "auth.py"}, "output": "file contents", "title": "Read auth.py"}}, "2025-01-01T10:00:02Z"),
    # m3: user text
    _part("p4",  "m3", {"type": "text", "text": "Looks good, apply the fix"}, "2025-01-01T10:00:03Z"),
    # m4: assistant text
    _part("p5",  "m4", {"type": "text", "text": "Done, the login bug is fixed"}, "2025-01-01T10:00:04Z"),
    # m5: user text
    _part("p6",  "m5", {"type": "text", "text": "Add dark mode support"}, "2025-01-02T09:00:01Z"),
    # m6: assistant multi-part (text + tool + text)
    _part("p7",  "m6", {"type": "text", "text": "I'll add dark mode"}, "2025-01-02T09:00:02Z"),
    _part("p8",  "m6", {"type": "tool", "tool": "edit_file", "state": {"status": "done", "input": {}, "output": "", "title": "Edit styles"}}, "2025-01-02T09:00:02Z"),
    _part("p9",  "m6", {"type": "text", "text": "Updated the stylesheet"}, "2025-01-02T09:00:02Z"),
    # m7-m10: simple text parts
    _part("p10", "m7",  {"type": "text", "text": "Can you also add a toggle?"}, "2025-01-02T09:00:03Z"),
    _part("p11", "m8",  {"type": "text", "text": "Sure, adding toggle component"}, "2025-01-02T09:00:04Z"),
    _part("p12", "m9",  {"type": "text", "text": "Perfect, ship it"}, "2025-01-02T09:00:05Z"),
    _part("p13", "m10", {"type": "text", "text": "All done!"}, "2025-01-02T09:00:06Z"),
    # m11-m12: s3 (malformed model session)
    _part("p14", "m11", {"type": "text", "text": "Review this PR"}, "2025-01-03T08:00:01Z"),
    _part("p15", "m12", {"type": "text", "text": "LGTM"}, "2025-01-03T08:00:02Z"),
    # m13-m20: s4 (refactor session, mix of text/tool/reasoning)
    _part("p16", "m13", {"type": "text", "text": "Refactor the DB layer"}, "2025-01-04T14:00:01Z"),
    _part("p17", "m14", {"type": "reasoning", "text": "Let me think about the best approach..."}, "2025-01-04T14:00:02Z"),
    _part("p18", "m14", {"type": "text", "text": "I'll restructure the queries"}, "2025-01-04T14:00:02Z"),
    _part("p19", "m15", {"type": "text", "text": "Use async throughout"}, "2025-01-04T14:00:03Z"),
    _part("p20", "m16", {"type": "tool", "tool": "edit_file", "state": {"status": "done", "input": {}, "output": "", "title": "Refactor"}}, "2025-01-04T14:00:04Z"),
    _part("p21", "m16", {"type": "text", "text": "Refactored to async"}, "2025-01-04T14:00:04Z"),
    _part("p22", "m17", {"type": "text", "text": "Add connection pooling"}, "2025-01-04T14:00:05Z"),
    _part("p23", "m18", {"type": "text", "text": "Added pooling support"}, "2025-01-04T14:00:06Z"),
    _part("p24", "m19", {"type": "text", "text": "Run the tests"}, "2025-01-04T14:00:07Z"),
    _part("p25", "m20", {"type": "text", "text": "All tests pass"}, "2025-01-04T14:00:08Z"),
    # m4: extra part with empty text (edge case) and a step-start (should be skipped by rich parser)
    _part("p26", "m4", {"type": "text", "text": ""}, "2025-01-01T10:00:04Z"),
    _part("p27", "m4", {"type": "step-start"}, "2025-01-01T10:00:04Z"),
    # message with no parts covered by m5 having only one part (already above)
]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create a seeded SQLite DB and return its path."""
    db_file = tmp_path / "test_opencode.db"
    conn = sqlite3.connect(db_file)
    conn.executescript(_SCHEMA)
    conn.executemany(
        "INSERT INTO session VALUES (?,?,?,?,?,?,?,?,?,?)",
        _SESSIONS,
    )
    conn.executemany(
        "INSERT INTO message VALUES (?,?,?,?)",
        _MESSAGES,
    )
    conn.executemany(
        "INSERT INTO part VALUES (?,?,?,?)",
        _PARTS,
    )
    conn.commit()
    conn.close()
    return db_file


# ---------------------------------------------------------------------------
# Redis fixture
# ---------------------------------------------------------------------------

@pytest.fixture
async def redis_client():
    """Async Redis client on localhost:6380. Skips if unavailable.

    Yields (redis_connection, seeded_keys_list). Append cache keys to the list
    for automatic cleanup after the test.
    """
    r = aioredis.Redis(host="localhost", port=6380, decode_responses=True)
    try:
        await r.ping()
    except (ConnectionError, OSError, aioredis.RedisError):
        pytest.skip("Redis not available on localhost:6380")

    seeded_keys: list[str] = []
    yield r, seeded_keys

    # cleanup tracked keys
    if seeded_keys:
        await r.delete(*seeded_keys)
    await r.aclose()


@pytest.fixture(autouse=True)
def _reset_llm_redis():
    """Reset module-level Redis client to avoid event-loop mismatch across tests."""
    import src.llm as _llm

    _llm._redis = aioredis.Redis(host="localhost", port=6380, decode_responses=True)


# ---------------------------------------------------------------------------
# Cache pre-seeder for complete_tool
# ---------------------------------------------------------------------------


async def preseed_complete_tool(
    r: aioredis.Redis,
    *,
    messages: list[dict],
    tool: dict,
    model: str,
    max_tokens: int,
    system: str | None,
    response: dict,
) -> str:
    """Pre-seed Redis cache for a complete_tool call. Returns the cache key."""
    from src.llm.cache import cache_key, cache_set

    key = cache_key(
        "complete_tool",
        messages=messages,
        tool=tool,
        model=model,
        max_tokens=max_tokens,
        system=system,
    )
    await cache_set(r, key, json.dumps(response, sort_keys=True))
    return key
