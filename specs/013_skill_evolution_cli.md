# 013 — Skill Evolution CLI (`play.py` extensions)

## Purpose

New CLI modes for initial run (skill bootstrapping from all past conversations) and periodic runs (ongoing refinement from new conversations).

## New Flags

| Flag | Description |
|---|---|
| `--evolve` | Run skill evolution pipeline (default: periodic mode) |
| `--first-time` | With `--evolve`: run initial bootstrapping over all past conversations |
| `--min-cluster-size N` | Minimum goals per cluster for skill synthesis (default: 3) |
| `--max-threads-per-cluster N` | Max conversation summaries fed to synthesizer (default: 5) |
| `--skills-dir PATH` | Skills directory (default: ~/.claude/skills/) |
| `--cluster-id N` | Process only one specific cluster (debug mode) |

## DRY_RUN Mode

Env var `DRY_RUN` controls write behavior:

```
DRY_RUN=0 (default)  → normal operation; writes skills to disk, updates SQLite
DRY_RUN=1            → everything runs except: no SKILL.md writes, no DB updates
```

Used during testing/development. No CLI flag needed — env var is cleaner (sets it once for a session).

## Periodic Run (default)

```bash
# Daily run over new sessions
uv run python play.py --evolve

# Dry run (no writes)
DRY_RUN=1 uv run python play.py --evolve

# Debug single cluster
uv run python play.py --evolve --cluster-id 3
```

Flow:
```
1. Find new sessions (time_created > last_run AND not in processed_sessions)
2. For each session:
   a. Extract goals
   b. Summarize threads
   c. Detect invoked skills (tool:skill parts in message data)
   d. Reflector (tag mode) per thread → rule_tags + insights_by_skill
   e. Update rule counters in SQLite (unless DRY_RUN)
   f. Mark session processed (unless DRY_RUN)
3. Group threads by goal cluster
4. For each cluster:
   a. Aggregate insights_by_skill across all threads
   b. For each skill with insights:
      Curator → ADD operations
   c. Write new rules to SKILL.md (unless DRY_RUN)
   d. Insert new rules into SQLite (unless DRY_RUN)
5. Print summary: N sessions processed, M rules added, K skills updated
```

## Initial Run (--first-time)

```bash
# Full initial run over all sessions
uv run python play.py --evolve --first-time

# Dry run (no writes)
DRY_RUN=1 uv run python play.py --evolve --first-time

# Target a specific cluster for testing
uv run python play.py --evolve --first-time --cluster-id 3

# Limit to specific sessions
uv run python play.py --evolve --first-time --goals-file sessions.txt
```

Flow:
```
1. Extract goals from sessions (--goals-file or all available)
2. Cluster all goals
3. Sort clusters by size DESC
4. For each cluster (above min size):
   a. Summarize threads (conversation_summarizer)
   b. Reflector (no-tag mode) per thread → insights_by_skill
   c. Synthesize frontmatter (skill_synthesizer phase A)
   d. Semantic search (skill_registry)
   e. Decide new/update (skill_registry)
   f. Synthesize full skill (skill_synthesizer phase D)
   g. Write SKILL.md (unless DRY_RUN)
   h. Insert rules into SQLite (unless DRY_RUN)
   i. Mark sessions processed (unless DRY_RUN)
5. Print summary: N skills created, M skills updated
```

## State Tracking

CLI must manage:
- `processed_sessions` table (which sessions have been processed, when, by which skill)
- `skills.db` for rule counters (in `~/.claude/skills/`)
- Skills directory for SKILL.md files (`~/.claude/skills/`)

## Output

### Periodic run output:
```
== Skill Evolution: Periodic ==
23 new sessions since last run
45 threads processed, 87 rule tags applied

mcp-debugging: +3 rules (2 sessions)
branch-context-gathering: +2 rules (4 sessions)

Summary: 2 skills updated, 5 rules added, 23 sessions marked processed
```

### Initial run output:
```
== Skill Evolution: Initial Run ==
Processing 234 sessions...
Extracted 567 goals across 45 clusters

Cluster 1 (23 goals) → new skill "mcp-debugging" (15 rules)
Cluster 2 (19 goals) → update skill "branch-context-gathering" (+8 rules)
Cluster 3 (12 goals) → new skill "gitlab-issue-workflow" (11 rules)
...

Summary: 12 new skills, 8 updated skills, 25 clusters skipped (too small)
```

## Design Decisions

- **`--evolve` defaults to periodic** — daily mode is the ongoing use case. Initial is a one-off bootstrap, explicitly signaled with `--first-time`.
- **`DRY_RUN` env var, not CLI flag** — sets once for a session during dev/testing. Cleaner than passing `--dry-run` on every command.
- **Session filtering**: Both timestamp AND `processed_sessions` table. New sessions = `time_created > last_run` AND `NOT IN processed_sessions`.
- **Curator grouping**: Threads grouped by goal cluster (inherited from `goal_clusterer`). No separate re-clustering step.
- **Curator trigger**: Runs immediately after all threads in the cluster are reflected. No N=5 trigger — the cluster IS the batch boundary.