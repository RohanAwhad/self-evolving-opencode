# 009 — Skill Registry (`skill_registry.py`)

## Purpose

Manages the skills directory (`~/.claude/skills/`), provides semantic search over skills, and tracks which sessions have been processed by each queue.

## Skills Directory

Skills live at `~/.claude/skills/<name>/SKILL.md`. Directory structure:

```
~/.claude/skills/<name>/
├── SKILL.md          # Required: frontmatter + body
├── scripts/          # Optional
├── references/       # Optional
└── assets/           # Optional
```

SKILL.md frontmatter:

```yaml
---
name: skill-name           # kebab-case, matches directory name
description: ...           # 1-1024 chars, imperative phrasing
---
```

## Skills DB

Location: `./skills.db` (project root). Separate from opencode DB. Contains:

- `processed_synthesize` — tracks synthesizer queue
- `processed_evolve` — tracks evolve queue
- `rules` table — rule ID/counter tracking (spec 010)
- `skill_clusters` — cluster-to-skill mapping

## API

### `scan_skills(skills_dir="~/.claude/skills") → list[SkillInfo]`

Scans all `SKILL.md` files, parses YAML frontmatter.

```python
@dataclass
class SkillInfo:
    name: str
    description: str          # from frontmatter
    path: Path                # full path to SKILL.md
    content: str              # full markdown content
    has_rules: bool           # True if ## Rules section exists
```

### `search_similar(query: str, skills: list[SkillInfo], top_k=3) → list[tuple[SkillInfo, float]]`

Embeds query using `all-mpnet-base-v2`, computes cosine similarity against all skill `description` fields. Returns top-K with scores.

### `find_closest_skill(query: str, skills_dir="~/.claude/skills", top_k=3) → list[tuple[SkillInfo, float]]`

Convenience: `scan_skills()` + `search_similar()`.

### `decide_new_or_update(new_name: str, new_description: str, top_skills: list[tuple[SkillInfo, float]], model) → SkillDecision`

```python
@dataclass
class SkillDecision:
    action: Literal["new", "update"]
    target_skill: str | None    # skill name if updating
    reasoning: str
```

LLM prompt: "Given this new skill draft and the 3 closest existing skills, decide whether this is a completely new skill or an update to one of the existing ones."

## Session Tracking (two queues)

### `get_unprocessed_sessions(queue: str, limit=50, db_path="./skills.db") → list[str]`

Returns session IDs not yet in the queue's processed table, from the opencode DB. `queue` is `"synthesize"` or `"evolve"`.

For synthesize: oldest first (fill gaps). For evolve: newest first (stay current).

### `mark_sessions_processed(queue: str, session_ids: list[str], skill_name: str, db_path="./skills.db") → None`

Inserts into the queue's processed table.

## Processed Tables

Simple tracking — just session ID + timestamp. No skill_name column (a session may touch multiple skills; skill→session relationships are in `skill_clusters`).

```sql
CREATE TABLE processed_synthesize (
    session_id TEXT PRIMARY KEY,
    processed_at TEXT NOT NULL
);

CREATE TABLE processed_evolve (
    session_id TEXT PRIMARY KEY,
    processed_at TEXT NOT NULL,
    rules_tagged INTEGER DEFAULT 0,
    rules_added INTEGER DEFAULT 0
);

CREATE TABLE skill_clusters (
    skill_name TEXT NOT NULL,
    cluster_id INTEGER NOT NULL,
    goal_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (skill_name, cluster_id, goal_text)
);
```

## Database Initialization

`init_skills_db()` is a separate script (`scripts/init_skills_db.py`), run manually once. Creates all tables. Not toggled by `DRY_RUN` — it's an explicit setup step, not part of the evolution pipeline.

```bash
uv run python scripts/init_skills_db.py
```

## Design Decisions

- **Embedding base**: Skill `description` only (from YAML frontmatter). Cheaper and more focused than full markdown.
- **Session filtering**: Both timestamp AND processed tables. New sessions = `time_created > last_run` AND `NOT IN processed_<queue>`.
- **Two processed tables**: Synthesizer and evolve are independent queues. Same session can appear in both.
- **DB path**: `db_path` kwarg (same pattern as `opencode_db.py`). Default: `./skills.db`.