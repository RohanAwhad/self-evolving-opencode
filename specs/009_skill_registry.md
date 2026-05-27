# 009 — Skill Registry (`skill_registry.py`)

## Purpose

Manages the skills directory (`~/.claude/skills/`), provides semantic search over skills, and tracks which conversations contributed to which skills.

## API

### `scan_skills(skills_dir="~/.claude/skills") → list[SkillInfo]`

Scans all `SKILL.md` files, parses YAML frontmatter.

```python
@dataclass
class SkillInfo:
    name: str
    description: str
    path: Path
    content: str            # full markdown content
    has_rules: bool         # True if ## Rules section exists
```

### `search_similar(query: str, skills: list[SkillInfo], top_k=3) → list[tuple[SkillInfo, float]]`

Embeds query (skill description or goal cluster summary) using `all-mpnet-base-v2`, computes cosine similarity against all skill descriptions. Returns top-K with scores.

### `find_closest_skill(query: str, skills_dir="~/.claude/skills", top_k=3) → list[tuple[SkillInfo, float]]`

Convenience: `scan_skills()` + `search_similar()`.

## Decision: New or Update

### `decide_new_or_update(new_frontmatter: SkillFrontmatter, top_skills: list[tuple[SkillInfo, float]], model) → SkillDecision`

```python
@dataclass
class SkillDecision:
    action: Literal["new", "update"]
    target_skill: str | None    # skill name if updating
    reasoning: str
```

LLM prompt: "Given this new skill draft and the 3 closest existing skills, decide whether this is a completely new skill or an update to one of the existing ones."

The LLM can choose `update:X` if the overlap is strong, or `new` if it's a genuinely new domain.

## Session Tracking

### `mark_sessions_processed(session_ids: list[str], skill_name: str, db) → None`

Inserts into `processed_sessions` table.

### `get_unprocessed_sessions(limit=50, db) → list[str]`

Returns session IDs not yet in `processed_sessions`, ordered by recency.

## Skills DB (SQLite)

Location: `~/.claude/skills/skills.db`

```sql
CREATE TABLE processed_sessions (
    session_id TEXT PRIMARY KEY,
    skill_name TEXT NOT NULL,
    processed_at TEXT NOT NULL
);

CREATE TABLE skill_clusters (
    skill_name TEXT NOT NULL,
    cluster_id INTEGER NOT NULL,
    goal_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (skill_name, cluster_id, goal_text)
);
```

## Design Decisions

- **Embedding base**: Embed skill `description` only (from YAML frontmatter). Cheaper and more focused than full markdown.
- **Session filtering for periodic**: Both approaches — check `processed_sessions` table AND compare `time_created` against last run timestamp.
