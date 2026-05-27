"""SQLite-backed rule database for skills.

Rule IDs are stored as plain strings (e.g. "gitlab-api-00001"). Counters live
only in SQLite — never in the SKILL.md markdown.
"""

import asyncio
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SKILLS_DB_PATH = Path("./skills.db")


@dataclass
class RuleRow:
    id: str
    skill_name: str
    content: str
    helpful_count: int
    harmful_count: int


@dataclass
class RuleTag:
    rule_id: str
    tag: Literal["irrelevant", "followed_helpful", "followed_harmful", "not_followed"]
    session_id: str


@dataclass
class RuleStats:
    total: int
    high_performing: int
    suspicious: int
    unused: int
    average_helpful: float
    average_harmful: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RULE_ID_RE = re.compile(r"^.+?-(\d+)$")


def _parse_rule_suffix(rule_id: str) -> int | None:
    m = _RULE_ID_RE.match(rule_id)
    if m:
        return int(m.group(1))
    return None


def _next_rule_id(skill_name: str, max_num: int) -> str:
    return f"{skill_name}-{max_num + 1:05d}"


# ---------------------------------------------------------------------------
# Sync implementations
# ---------------------------------------------------------------------------


def _insert_rules_sync(skill_name: str, rules: list[tuple[str, str]], db_path: Path) -> None:
    import datetime

    conn = sqlite3.connect(db_path)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.executemany(
        "INSERT INTO rules (id, skill_name, content, helpful_count, harmful_count, created_at, updated_at) VALUES (?, ?, ?, 0, 0, ?, ?)",
        [(rid, skill_name, content, now, now) for rid, content in rules],
    )
    conn.commit()
    conn.close()


async def insert_rules(
    skill_name: str, rules: list[tuple[str, str]], db_path: Path = SKILLS_DB_PATH
) -> None:
    return await asyncio.to_thread(_insert_rules_sync, skill_name, rules, db_path)


def _get_rules_for_skill_sync(skill_name: str, db_path: Path) -> list[RuleRow]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, skill_name, content, helpful_count, harmful_count FROM rules WHERE skill_name = ? ORDER BY id",
        (skill_name,),
    ).fetchall()
    conn.close()
    return [RuleRow(**dict(r)) for r in rows]


async def get_rules_for_skill(
    skill_name: str, db_path: Path = SKILLS_DB_PATH
) -> list[RuleRow]:
    return await asyncio.to_thread(_get_rules_for_skill_sync, skill_name, db_path)


def _update_counters_sync(tags: list[RuleTag], db_path: Path) -> None:
    import datetime

    conn = sqlite3.connect(db_path)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for tag in tags:
        if tag.tag == "irrelevant":
            continue
        field = "helpful_count" if tag.tag == "followed_helpful" else "harmful_count"
        conn.execute(
            f"UPDATE rules SET {field} = {field} + 1, updated_at = ? WHERE id = ?",
            (now, tag.rule_id),
        )
    conn.commit()
    conn.close()


async def update_counters(
    tags: list[RuleTag], db_path: Path = SKILLS_DB_PATH
) -> None:
    return await asyncio.to_thread(_update_counters_sync, tags, db_path)


def _get_max_rule_id_sync(skill_name: str, db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id FROM rules WHERE skill_name = ? ORDER BY id DESC LIMIT 1",
        (skill_name,),
    ).fetchone()
    conn.close()
    if row is None:
        return 0
    suffix = _parse_rule_suffix(row["id"])
    return suffix if suffix is not None else 0


async def get_max_rule_id(
    skill_name: str, db_path: Path = SKILLS_DB_PATH
) -> int:
    return await asyncio.to_thread(_get_max_rule_id_sync, skill_name, db_path)


def _get_rule_stats_sync(skill_name: str, db_path: Path) -> RuleStats:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT helpful_count, harmful_count FROM rules WHERE skill_name = ?",
        (skill_name,),
    ).fetchall()
    conn.close()

    if not rows:
        return RuleStats(total=0, high_performing=0, suspicious=0, unused=0, average_helpful=0.0, average_harmful=0.0)

    total = len(rows)
    helpfuls = [r["helpful_count"] for r in rows]
    harmfuls = [r["harmful_count"] for r in rows]

    high_performing = sum(1 for h, hm in zip(helpfuls, harmfuls) if h > hm * 2)
    suspicious = sum(1 for h, hm in zip(helpfuls, harmfuls) if hm > h * 3)
    unused = sum(1 for h, hm in zip(helpfuls, harmfuls) if h + hm == 0)

    return RuleStats(
        total=total,
        high_performing=high_performing,
        suspicious=suspicious,
        unused=unused,
        average_helpful=sum(helpfuls) / total,
        average_harmful=sum(harmfuls) / total,
    )


async def get_rule_stats(
    skill_name: str, db_path: Path = SKILLS_DB_PATH
) -> RuleStats:
    return await asyncio.to_thread(_get_rule_stats_sync, skill_name, db_path)


def _next_rule_ids_sync(skill_name: str, count: int, db_path: Path) -> list[str]:
    max_id = _get_max_rule_id_sync(skill_name, db_path)
    return [_next_rule_id(skill_name, max_id + i) for i in range(count)]


async def next_rule_ids(
    skill_name: str, count: int, db_path: Path = SKILLS_DB_PATH
) -> list[str]:
    return await asyncio.to_thread(_next_rule_ids_sync, skill_name, count, db_path)
