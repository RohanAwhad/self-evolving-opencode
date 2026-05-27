"""Skill registry — scan, search, decide, and session tracking.

Manages the skills directory (~/.claude/skills/), provides semantic search
over skill descriptions, LLM-based new-vs-update decisions, and dual-queue
session processing tracking.
"""

import asyncio
import re
import sqlite3
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm import DEFAULT_MODEL, complete_tool
from src.opencode_db import DB_PATH as OPENCODE_DB_PATH
from src.skill_rules import SKILLS_DB_PATH

SKILLS_DIR_DEFAULT = Path.home() / ".claude" / "skills"
EMBEDDING_MODEL = "all-mpnet-base-v2"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


@dataclass
class SkillInfo:
    name: str
    description: str
    path: Path
    content: str
    has_rules: bool


@dataclass
class SkillDecision:
    action: Literal["new", "update"]
    target_skill: str | None
    reasoning: str


# ---------------------------------------------------------------------------
# Skill directory scanning
# ---------------------------------------------------------------------------


def _parse_frontmatter(content: str) -> dict[str, str] | None:
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return None
    try:
        return yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return None


def _scan_skills_sync(skills_dir: Path) -> list[SkillInfo]:
    results: list[SkillInfo] = []
    if not skills_dir.exists():
        return results
    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            continue
        content = skill_md.read_text()
        fm = _parse_frontmatter(content)
        if fm is None:
            continue
        name = fm.get("name") or skill_dir.name
        description = fm.get("description") or ""
        has_rules = "## Rules" in content
        results.append(SkillInfo(name=name, description=description, path=skill_md, content=content, has_rules=has_rules))
    return results


async def scan_skills(skills_dir: Path = SKILLS_DIR_DEFAULT) -> list[SkillInfo]:
    return await asyncio.to_thread(_scan_skills_sync, skills_dir)


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _embed(texts: list[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    model = _get_model(model_name)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def _search_similar_sync(
    query: str, skills: list[SkillInfo], top_k: int = 3
) -> list[tuple[SkillInfo, float]]:
    if not skills:
        return []
    descriptions = [s.description for s in skills]
    query_emb = _embed([query])
    skill_embs = _embed(descriptions)
    sims = cosine_similarity(query_emb, skill_embs)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(skills[i], float(sims[i])) for i in top_indices if sims[i] > 0]


async def search_similar(
    query: str, skills: list[SkillInfo], top_k: int = 3
) -> list[tuple[SkillInfo, float]]:
    return await asyncio.to_thread(_search_similar_sync, query, skills, top_k)


async def find_closest_skill(
    query: str, skills_dir: Path = SKILLS_DIR_DEFAULT, top_k: int = 3
) -> list[tuple[SkillInfo, float]]:
    skills = await scan_skills(skills_dir)
    return await search_similar(query, skills, top_k)


# ---------------------------------------------------------------------------
# LLM decision: new skill vs update existing
# ---------------------------------------------------------------------------

DECIDE_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a skill classifier. You receive:
    1. A proposed new skill (name + description)
    2. The 3 closest existing skills (name + description + similarity score)

    Decide whether this is a completely new skill or an update to one of the
    existing skills.
    """)

DECIDE_TOOL = {
    "name": "decide_skill_action",
    "description": "Decide whether to create a new skill or update an existing one.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["new", "update"], "description": "Whether to create a new skill or update existing"},
            "target_skill": {"type": "string", "description": "Skill name to update (null if new)", "nullable": True},
            "reasoning": {"type": "string", "description": "Short explanation (1-2 sentences)"},
        },
        "required": ["action", "reasoning"],
    },
}


async def decide_new_or_update(
    new_name: str,
    new_description: str,
    top_skills: list[tuple[SkillInfo, float]],
    model: str = DEFAULT_MODEL,
) -> SkillDecision:
    skills_text = "\n".join(
        f"- {s.name} (score={score:.3f}): {s.description}"
        for s, score in top_skills
    )
    prompt = (
        f"Proposed skill:\n  name: {new_name}\n  description: {new_description}\n\n"
        f"Closest existing skills:\n{skills_text}\n\n"
        f"Decide: create new or update existing?"
    )
    data = await complete_tool(
        messages=[{"role": "user", "content": prompt}],
        tool=DECIDE_TOOL,
        model=model,
        max_tokens=512,
        system=DECIDE_SYSTEM_PROMPT,
    )
    return SkillDecision(
        action=data["action"],
        target_skill=data.get("target_skill"),
        reasoning=data.get("reasoning", ""),
    )


# ---------------------------------------------------------------------------
# Session tracking (dual queue)
# ---------------------------------------------------------------------------


def _get_unprocessed_sessions_sync(
    queue: str,
    limit: int,
    skills_db_path: Path,
    opencode_db_path: Path,
) -> list[str]:
    table = f"processed_{queue}"
    order = "ASC" if queue == "synthesize" else "DESC"

    skills_conn = sqlite3.connect(skills_db_path)
    processed_rows = skills_conn.execute(f"SELECT session_id FROM {table}").fetchall()
    skills_conn.close()
    processed_ids = {r[0] for r in processed_rows}

    opencode_conn = sqlite3.connect(opencode_db_path)
    if processed_ids:
        placeholders = ",".join("?" * len(processed_ids))
        query = (
            f"SELECT id FROM session WHERE id NOT IN ({placeholders}) "
            f"ORDER BY time_created {order} LIMIT ?"
        )
        rows = opencode_conn.execute(query, (*processed_ids, limit)).fetchall()
    else:
        query = f"SELECT id FROM session ORDER BY time_created {order} LIMIT ?"
        rows = opencode_conn.execute(query, (limit,)).fetchall()
    opencode_conn.close()
    return [r[0] for r in rows]


async def get_unprocessed_sessions(
    queue: str,
    limit: int = 50,
    skills_db_path: Path = SKILLS_DB_PATH,
    opencode_db_path: Path = OPENCODE_DB_PATH,
) -> list[str]:
    return await asyncio.to_thread(
        _get_unprocessed_sessions_sync, queue, limit, skills_db_path, opencode_db_path
    )


def _mark_sessions_processed_sync(
    queue: str,
    session_ids: list[str],
    db_path: Path,
    skill_name: str | None = None,
    action: str | None = None,
) -> None:
    import datetime

    table = f"processed_{queue}"
    conn = sqlite3.connect(db_path)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if queue == "synthesize":
        conn.executemany(
            f"INSERT OR IGNORE INTO {table} (session_id, processed_at, skill_name, action) VALUES (?, ?, ?, ?)",
            [(sid, now, skill_name, action) for sid in session_ids],
        )
    else:
        conn.executemany(
            f"INSERT OR IGNORE INTO {table} (session_id, processed_at) VALUES (?, ?)",
            [(sid, now) for sid in session_ids],
        )
    conn.commit()
    conn.close()


async def mark_sessions_processed(
    queue: str,
    session_ids: list[str],
    db_path: Path = SKILLS_DB_PATH,
    skill_name: str | None = None,
    action: str | None = None,
) -> None:
    return await asyncio.to_thread(
        _mark_sessions_processed_sync, queue, session_ids, db_path, skill_name, action
    )


def _is_session_processed_sync(
    queue: str, session_id: str, db_path: Path
) -> bool:
    table = f"processed_{queue}"
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        f"SELECT 1 FROM {table} WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    return row is not None


async def is_session_processed(
    queue: str, session_id: str, db_path: Path = SKILLS_DB_PATH
) -> bool:
    return await asyncio.to_thread(_is_session_processed_sync, queue, session_id, db_path)
