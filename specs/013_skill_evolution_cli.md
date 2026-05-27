# 013 — Skill Evolution CLI (`play.py` extensions)

## Purpose

Single `--evolve` mode that runs two independent queues: synthesizer (creates/updates workflows from raw threads) and evolve (reflector → curator for rule management). Sequential execution to avoid race conditions on SKILL.md writes.

## CLI

```bash
uv run python play.py --evolve [-n N]
```

## Two Queues

Each queue has its own `processed_` table. Same session can appear in both queues.

| Queue | Table | Job |
|---|---|---|
| Synthesizer | `processed_synthesize` | Extract workflow from ≤10 threads per cluster. Create new skills or update existing workflow. |
| Evolve | `processed_evolve` | Reflector tags rules per thread, curator adds new rules per skill. |

## Flow

```
--evolve
  │
  │  [Sequential — synthesizer first, then evolve]
  │
  ├─► Synthesizer queue
  │     Find oldest N sessions NOT in processed_synthesize
  │     Extract goals → cluster
  │     Per cluster:
  │       Summarize ≤10 threads
  │       Semantic search against ~/.claude/skills/
  │       → no match: Synthesizer → create SKILL.md
  │       → match: Synthesizer → update workflow
  │     Mark sessions in processed_synthesize
  │
  └─► Evolve queue
        Find newest M sessions NOT in processed_evolve
        Detect skills (tool:skill parts)
        Group threads by goal cluster
        Reflector per thread → insights_by_skill + rule_tags
        Aggregate per skill per cluster
        Curator per skill → ADD rules → update SKILL.md
        Mark sessions in processed_evolve
```

## Sequential Execution

Synthesizer runs first, evolve second. Both queues can write to the same SKILL.md (synthesizer writes `## Workflow`, curator appends to `## Rules`). Running sequentially avoids race conditions without needing file-level locks.

**Note on async**: Running both queues in parallel (`asyncio.gather`) is possible in theory since they write to different markdown sections. But without file-level locking, concurrent writes could corrupt SKILL.md. Sequential execution is safer. Revisit async when we're fully tested.

## Processed Sessions

Two tables in `~/.claude/skills/skills.db`:

```sql
CREATE TABLE processed_synthesize (
    session_id TEXT PRIMARY KEY,
    processed_at TEXT NOT NULL,
    skill_name TEXT,
    action TEXT  -- "created" or "updated"
);

CREATE TABLE processed_evolve (
    session_id TEXT PRIMARY KEY,
    processed_at TEXT NOT NULL,
    rules_tagged INTEGER DEFAULT 0,
    rules_added INTEGER DEFAULT 0
);
```

## DRY_RUN

```
DRY_RUN=1 uv run python play.py --evolve   # no writes to disk or DB
DRY_RUN=0 (default)                          # normal operation
```

In dry run mode: everything runs (LLM calls, reflector, curator, synthesizer), but SKILL.md writes and SQLite inserts are skipped. Output is printed to stdout.

## Testing

Same pattern as existing modules:
- Pre-seed Redis cache with LLM responses for reflector, curator, synthesizer, and semantic search
- Run with `DRY_RUN=1`
- Assert on printed output (which skills would be created/updated, which rules would be added)
- Assert no files written to `~/.claude/skills/`
- Assert no rows in `processed_` tables
- Use small fixture clusters (2-3 goals, 2-3 threads)

## Output

```
== Skill Evolution ==

--- Synthesizer queue ---
3 unprocessed sessions (oldest first)
Cluster 1 (2 goals): no match → new skill "mcp-debugging" (12 rules)
Cluster 2 (1 goal): match "branch-context-gathering" → updated workflow

--- Evolve queue ---
5 unprocessed sessions (newest first)
3 skills with new insights
  mcp-debugging: +4 rules
  branch-context-gathering: +2 rules
  gitlab-api: +1 rule

Summary: 1 new skill, 1 workflow updated, 7 rules added, 8 sessions processed
```

## Design Decisions

- **Single `--evolve` flag** — no `--first-time` mode. System auto-detects state from `processed_` tables.
- **Separate `processed_` tables** — synthesizer and evolve are independent queues. Same session goes through both.
- **Sequential execution** — avoids race conditions on SKILL.md writes. Async parallelization is noted but not yet implemented.
- **Synthesizer ≤10 threads** — keeps context window manageable while capturing enough patterns for workflow extraction.
- **Default `-n` picks sessions based on recency** — oldest first for synthesizer (fill gaps), newest first for evolve (stay current).