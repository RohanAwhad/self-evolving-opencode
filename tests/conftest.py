"""Shared test fixtures -- DB factory, Redis client, cache pre-seeder."""

import json
import sqlite3
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

_SCHEMA = """\
CREATE TABLE session (
    id TEXT PRIMARY KEY,
    title TEXT,
    directory TEXT,
    agent TEXT,
    model TEXT,
    cost REAL,
    tokens_input INTEGER,
    tokens_output INTEGER,
    time_created TEXT,
    time_updated TEXT
);

CREATE TABLE message (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES session(id),
    time_created TEXT,
    data TEXT
);

CREATE TABLE part (
    id TEXT PRIMARY KEY,
    message_id TEXT REFERENCES message(id),
    time_created TEXT,
    data TEXT
);
"""

# Seed: 5 sessions, 11 messages, 12 parts -- covers all edge cases
_SESSIONS = [
    # fmt: off
    # (id, title, directory, agent, model, cost, tokens_in, tokens_out, time_created, time_updated)
    ("s1", "Session Alpha", "/proj/alpha", "coder",    '{"id":"claude-3"}',    0.05, 1000, 500, "2024-01-01T00:00:00Z", "2024-01-06T00:00:00Z"),
    ("s2", "Session Beta",  "/proj/beta",  "reviewer", '{"id":"gpt-4"}',       0.10, 2000, 800, "2024-01-02T00:00:00Z", "2024-01-05T00:00:00Z"),
    ("s3", None,             None,          None,       None,                   None, None, None, "2024-01-03T00:00:00Z", "2024-01-04T00:00:00Z"),
    ("s4", "Empty Session",  "/proj/delta", "coder",    '{"id":"claude-opus"}', 0.01, 100,  50,  "2024-01-04T00:00:00Z", "2024-01-03T00:00:00Z"),
    ("s5", "Sparse Session", "/proj/echo",  "coder",    '{"id":"claude-3"}',    0.02, 200,  100, "2024-01-05T00:00:00Z", "2024-01-02T00:00:00Z"),
    # fmt: on
]

_MESSAGES = [
    # (id, session_id, time_created, data)
    # s1: 4 messages (user/assistant alternating)
    ("m1", "s1", "2024-01-01T00:01:00Z", '{"role":"user"}'),
    ("m2", "s1", "2024-01-01T00:02:00Z", '{"role":"assistant"}'),
    ("m3", "s1", "2024-01-01T00:03:00Z", '{"role":"user"}'),
    ("m4", "s1", "2024-01-01T00:04:00Z", '{"role":"assistant"}'),
    # s2: 3 messages
    ("m5", "s2", "2024-01-02T00:01:00Z", '{"role":"user"}'),
    ("m6", "s2", "2024-01-02T00:02:00Z", '{"role":"assistant"}'),
    ("m7", "s2", "2024-01-02T00:03:00Z", '{"role":"user"}'),
    # s3: 2 messages
    ("m8", "s3", "2024-01-03T00:01:00Z", '{"role":"user"}'),
    ("m9", "s3", "2024-01-03T00:02:00Z", '{"role":"assistant"}'),
    # s4: 0 messages
    # s5: 2 messages (m11 has no parts -> gets skipped)
    ("m10", "s5", "2024-01-05T00:01:00Z", '{"role":"assistant"}'),
    ("m11", "s5", "2024-01-05T00:02:00Z", '{"role":"user"}'),
]


def _part(pid: str, msg_id: str, ts: str, data: dict) -> tuple:
    return (pid, msg_id, ts, json.dumps(data))


_PARTS = [
    # m1: single text
    _part("p1", "m1", "2024-01-01T00:01:01Z", {"type": "text", "text": "Hello"}),
    # m2: two text parts (multi-part concat test)
    _part("p2", "m2", "2024-01-01T00:02:01Z", {"type": "text", "text": "Hi there"}),
    _part("p3", "m2", "2024-01-01T00:02:02Z", {"type": "text", "text": "How can I help?"}),
    # m3: single text
    _part("p4", "m3", "2024-01-01T00:03:01Z", {"type": "text", "text": "Fix the bug"}),
    # m4: tool + text (tool formatting test)
    _part("p5", "m4", "2024-01-01T00:04:01Z", {"type": "tool", "tool": "bash"}),
    _part("p6", "m4", "2024-01-01T00:04:02Z", {"type": "text", "text": "Done fixing"}),
    # m5-m7: single text parts
    _part("p7", "m5", "2024-01-02T00:01:01Z", {"type": "text", "text": "Review this code"}),
    _part("p8", "m6", "2024-01-02T00:02:01Z", {"type": "text", "text": "LGTM"}),
    _part("p9", "m7", "2024-01-02T00:03:01Z", {"type": "text", "text": "Thanks"}),
    # m8-m9: single text parts
    _part("p10", "m8", "2024-01-03T00:01:01Z", {"type": "text", "text": "Quick test"}),
    _part("p11", "m9", "2024-01-03T00:02:01Z", {"type": "text", "text": "Response here"}),
    # m10: single text part
    _part("p12", "m10", "2024-01-05T00:01:01Z", {"type": "text", "text": "I'll help"}),
    # m11: NO parts (tests message-with-no-parts-skipped)
]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Seeded SQLite DB for opencode_db tests."""
    db_file = tmp_path / "opencode.db"
    conn = sqlite3.connect(db_file)
    conn.executescript(_SCHEMA)
    conn.executemany("INSERT INTO session VALUES (?,?,?,?,?,?,?,?,?,?)", _SESSIONS)
    conn.executemany("INSERT INTO message VALUES (?,?,?,?)", _MESSAGES)
    conn.executemany("INSERT INTO part VALUES (?,?,?,?)", _PARTS)
    conn.commit()
    conn.close()
    return db_file


# ---------------------------------------------------------------------------
# Redis fixtures
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
