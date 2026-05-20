"""Shared DB access layer for OpenCode's SQLite database."""

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


DB_PATH = Path.home() / ".local/share/opencode/opencode.db"


@dataclass
class Session:
    id: str
    title: str
    directory: str
    agent: str
    model_id: str
    cost: float
    tokens_input: int
    tokens_output: int
    time_created: str
    time_updated: str
    message_count: int


def get_sessions(limit: int = 50, db_path: Path = DB_PATH) -> list[Session]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
            s.id,
            s.title,
            s.directory,
            s.agent,
            s.model,
            s.cost,
            s.tokens_input,
            s.tokens_output,
            s.time_created,
            s.time_updated,
            COUNT(m.id) AS message_count
        FROM session s
        LEFT JOIN message m ON m.session_id = s.id
        GROUP BY s.id
        ORDER BY s.time_updated DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    sessions = []
    for r in rows:
        model_raw = r["model"] or "{}"
        model_data = json.loads(model_raw) if isinstance(model_raw, str) else model_raw
        model_id = model_data.get("id", "unknown") if isinstance(model_data, dict) else str(model_data)

        sessions.append(
            Session(
                id=r["id"],
                title=r["title"] or "(untitled)",
                directory=r["directory"] or "",
                agent=r["agent"] or "",
                model_id=model_id,
                cost=r["cost"] or 0.0,
                tokens_input=r["tokens_input"] or 0,
                tokens_output=r["tokens_output"] or 0,
                time_created=r["time_created"] or "",
                time_updated=r["time_updated"] or "",
                message_count=r["message_count"],
            )
        )
    return sessions


def get_messages_for_session(session_id: str, db_path: Path = DB_PATH) -> list[dict]:
    """Load all messages for a session as list[dict] with role/content keys."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    messages = conn.execute(
        """
        SELECT m.id, json_extract(m.data, '$.role') AS role
        FROM message m
        WHERE m.session_id = ?
        ORDER BY m.time_created ASC
        """,
        (session_id,),
    ).fetchall()

    result: list[dict] = []
    for msg in messages:
        role = msg["role"] or "unknown"
        parts = conn.execute(
            "SELECT data FROM part WHERE message_id = ? ORDER BY time_created ASC",
            (msg["id"],),
        ).fetchall()

        text_chunks: list[str] = []
        for p in parts:
            pdata = json.loads(p["data"]) if isinstance(p["data"], str) else p["data"]
            ptype = pdata.get("type", "")
            if ptype == "text" and pdata.get("text"):
                text_chunks.append(pdata["text"])
            elif ptype == "tool" and pdata.get("tool"):
                text_chunks.append(f"[tool: {pdata['tool']}]")

        if text_chunks:
            result.append({"role": role, "content": "\n".join(text_chunks)})

    conn.close()
    return result


def get_conversation_transcript(session_id: str, db_path: Path = DB_PATH) -> str:
    """Load messages+parts for a session, return a human-readable transcript."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    messages = conn.execute(
        """
        SELECT m.id, json_extract(m.data, '$.role') AS role
        FROM message m
        WHERE m.session_id = ?
        ORDER BY m.time_created ASC
        """,
        (session_id,),
    ).fetchall()

    lines: list[str] = []
    for i, msg in enumerate(messages, 1):
        role = msg["role"] or "unknown"
        parts = conn.execute(
            "SELECT data FROM part WHERE message_id = ? ORDER BY time_created ASC",
            (msg["id"],),
        ).fetchall()

        text_chunks: list[str] = []
        for p in parts:
            pdata = json.loads(p["data"]) if isinstance(p["data"], str) else p["data"]
            ptype = pdata.get("type", "")
            if ptype == "text" and pdata.get("text"):
                text_chunks.append(pdata["text"])
            elif ptype == "tool" and pdata.get("tool"):
                text_chunks.append(f"[tool: {pdata['tool']}]")

        if text_chunks:
            body = "\n".join(text_chunks)
            lines.append(f"--- Message {i} ({role}) ---\n{body}\n")

    conn.close()
    return "\n".join(lines)


def parse_message_range(range_str: str) -> tuple[int, int]:
    """Parse 'msgs 1-8' or 'msgs 3-15' into (start, end) 0-indexed."""
    m = re.search(r"(\d+)\s*-\s*(\d+)", range_str)
    if not m:
        single = re.search(r"(\d+)", range_str)
        if single:
            n = int(single.group(1)) - 1
            return (n, n + 1)
        return (0, 999999)
    return (int(m.group(1)) - 1, int(m.group(2)))


def slice_messages(all_messages: list[dict], range_str: str) -> list[dict]:
    """Slice messages by a range string like 'msgs 1-8'."""
    start, end = parse_message_range(range_str)
    return all_messages[start:end]
