# 010 — Rule Database (`skill_rules.py`)

## Purpose

SQLite-backed tracking of skill rules — IDs, content, helpful/harmful counters. Counters are NOT in the skill markdown (they would skew the model). They live only here.

## Schema

```sql
CREATE TABLE rules (
    id TEXT PRIMARY KEY,              -- "skillname-00001"
    skill_name TEXT NOT NULL,         -- which skill owns this rule
    content TEXT NOT NULL,            -- the rule text (without ID prefix)
    helpful_count INTEGER DEFAULT 0,
    harmful_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_rules_skill ON rules(skill_name);
```

## API

### `insert_rules(skill_name: str, rules: list[tuple[str, str]], db) → None`

Insert new rules. Each rule: `(rule_id, content)`. Counters start at 0.

```python
await insert_rules("gitlab-api", [
    ("gitlab-api-00001", "Always verify repo context before editing files"),
    ("gitlab-api-00002", "Prefer minimal diffs over large refactors"),
], db)
```

### `get_rules_for_skill(skill_name: str, db_path="./skills.db") → list[RuleRow]`

```python
@dataclass
class RuleRow:
    id: str
    skill_name: str
    content: str
    helpful_count: int
    harmful_count: int
```

### `update_counters(tags: list[RuleTag], db_path="./skills.db") → None`

Batch-update counters after reflector runs.

```python
@dataclass
class RuleTag:
    rule_id: str
    tag: Literal["irrelevant", "followed_helpful", "followed_harmful", "not_followed"]
    session_id: str  # for auditing
```

Counter logic:
- `irrelevant` → no change
- `followed_helpful` → `helpful_count += 1`
- `followed_harmful` → `harmful_count += 1`
- `not_followed` → `harmful_count += 1`

### `get_max_rule_id(skill_name: str, db_path="./skills.db") → int`

Returns highest numeric ID for a skill (e.g., `gitlab-api-00042` → 42). Used for generating next ID.

### `get_rule_stats(skill_name: str, db_path="./skills.db") → RuleStats`

```python
@dataclass
class RuleStats:
    total: int
    high_performing: int     # helpful > harmful * 2
    suspicious: int           # harmful > helpful * 3
    unused: int               # helpful + harmful == 0
    average_helpful: float
    average_harmful: float
```

Used by curator to understand which rules are working.

## Skills DB Location

Same DB as registry: `./skills.db` (project root, shared with processed tables and skill_clusters from spec 009).

## API

All functions accept `db_path="./skills.db"` kwarg (consistent with existing `db_path` pattern from spec 001).

### `insert_rules(skill_name: str, rules: list[tuple[str, str]], db_path="./skills.db") → None`
