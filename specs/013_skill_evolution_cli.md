# 013 — Skill Evolution CLI (`play.py` extensions)

## Purpose

New CLI modes for initial run (skill bootstrapping from all past conversations) and periodic runs (ongoing refinement from new conversations).

## New Flags

| Flag | Description |
|---|---|
| `--evolve` | Run skill evolution pipeline |
| `--evolve-mode initial \| periodic` | Which mode (default: initial) |
| `--min-cluster-size N` | Minimum goals per cluster for skill synthesis (default: 3) |
| `--max-threads-per-cluster N` | Max conversation summaries fed to synthesizer (default: 5) |
| `--skills-dir PATH` | Skills directory (default: ~/.claude/skills/) |
| `--dry-run` | Run pipeline but don't write skills to disk |
| `--cluster-id N` | Process only one specific cluster (debug mode) |

## Initial Run

```bash
# Full initial run over all sessions
uv run python play.py --evolve --evolve-mode initial

# Dry run (no writes)
uv run python play.py --evolve --evolve-mode initial --dry-run

# Target a specific cluster for testing
uv run python play.py --evolve --evolve-mode initial --cluster-id 3

# With custom settings
uv run python play.py --evolve --evolve-mode initial \
    --min-cluster-size 5 \
    --max-threads-per-cluster 8 \
    --goals-file sessions.txt   # optional: limit to specific sessions
```

Flow:
```
1. Extract goals from sessions (--goals-file or all unprocessed)
2. Cluster all goals
3. Sort clusters by size DESC
4. For each cluster:
   a. Summarize threads (conversation_summarizer)
   b. Reflector (no-tag mode) per thread → insights
   c. Synthesize frontmatter (skill_synthesizer phase A)
   d. Semantic search (skill_registry)
   e. Decide new/update (skill_registry)
   f. Synthesize full skill (skill_synthesizer phase D)
   g. Write to disk (if not --dry-run)
   h. Insert rules into SQLite
   i. Mark sessions processed
5. Print summary: N skills created, M skills updated
```

## Periodic Run

```bash
# Daily run over new sessions
uv run python play.py --evolve --evolve-mode periodic

# Dry run
uv run python play.py --evolve --evolve-mode periodic --dry-run
```

Flow:
```
1. Find new sessions (not in processed_sessions)
2. For each session:
   a. Extract goals
   b. Summarize threads
   c. Detect invoked skills (from conversation metadata)
   d. Reflector (tag mode) per thread → tags + insights
   e. Update rule counters in SQLite
   f. Mark session processed
3. Re-cluster recent threads by goals
4. For each cluster above threshold:
   a. Curator → ADD operations
   b. Write new rules to SKILL.md
   c. Insert new rules into SQLite
5. Print summary: N sessions processed, M rules added, K skills updated
```

## State Tracking

CLI must manage:
- `processed_sessions` table (which sessions have been processed, when, by which skill)
- `skills.db` for rule counters
- Skills directory for SKILL.md files

## Output

### Initial run output:
```
== Skill Evolution: Initial Run ==
Processing 234 sessions...
Extracted 567 goals across 45 clusters

Cluster 1 (23 goals): [mcp-debugging] → new skill "mcp-debugging" (15 rules)
Cluster 2 (19 goals): [git-workflow] → update skill "branch-context-gathering" (+8 rules)
Cluster 3 (12 goals): [issue-management] → new skill "gitlab-issue-workflow" (11 rules)
...
Summary: 12 new skills, 8 updated skills, 25 clusters skipped (too small)
```

### Periodic run output:
```
== Skill Evolution: Periodic Run ==
23 new sessions since last run
45 threads processed, 87 rule tags applied
3 clusters with new insights

Cluster "mcp-concurrency" (5 threads): +3 rules → "mcp-debugging"
Cluster "git-safety" (4 threads): +2 rules → "branch-context-gathering"

Summary: 2 skills updated, 5 rules added, 23 sessions marked processed
```

## Design Decisions

- **`--evolve-mode` default**: `initial` is the default. Safer — one-time batch, explicit about creating/updating skills. Periodic requires `--evolve-mode periodic`.
- **Session filtering**: Both timestamp AND `processed_sessions` table. New sessions are those with `time_created > last_run_time` AND not in `processed_sessions`. This handles edge cases where sessions were created but skipped.
- **Curator grouping**: Threads are grouped by goal cluster (inherited from `goal_clusterer`). No separate re-clustering step needed. Same clusters used for both initial and periodic runs.
- **Curator trigger**: Runs when ≥5 new threads accumulate in a cluster (or end of daily batch, whichever comes first).
