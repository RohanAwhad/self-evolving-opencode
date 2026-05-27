# 012 — Curator (`curator.py`)

## Purpose

Runs per goal cluster after N threads have accumulated. Synthesizes new rules from reflector insights, assigns them to appropriate skills. Can also propose new skills if a cluster has no matching skill.

## When It Runs

- **First-time run**: Curator in "full builder" mode — produces complete skill from scratch (frontmatter + workflow + rules).
- **Periodic run**: Curator in "ADD mode" — only adds new rules to existing skills. Threads are re-clustered by goals, curator runs per cluster.

## API

### `curate_cluster(cluster_id, threads: list[Reflection], skills: list[SkillInfo], model) → list[CuratorOperation]`

```python
@dataclass
class CuratorOperation:
    type: Literal["ADD_RULE", "PROPOSE_SKILL"]
    target_skill: str
    rule_id: str | None
    content: str
    reasoning: str

@dataclass
class CuratorInput:
    cluster_id: int
    goal_texts: list[str]              # goals in this cluster
    reflections: list[Reflection]       # all reflector outputs for threads in cluster
    skills: list[SkillInfo]             # all skills referenced by these threads
    rule_stats: dict[str, RuleStats]    # per-skill rule statistics from SQLite
```

### First-time variant: `curate_new_skill(cluster_id, goals, summaries, model) → str`

Full builder mode. Produces complete SKILL.md content (frontmatter + workflow + rules). Used during initial run.

## LLM Prompt (periodic mode)

```
You are a skill curator. You will receive:
1. A cluster of related user goals
2. Reflections from multiple conversation threads (rule tags + new insights)
3. Statistics on existing rules (helpful/harmful counts)
4. The current content of all skills referenced by these threads

Your job:
- Synthesize new insights into concrete rules
- Decide which skill each rule belongs to
- Do NOT modify or delete existing rules (append-only)
- If a cluster has no matching skill, propose a new skill

Output: JSON array of ADD operations or SKILL_PROPOSAL operations.
```

## Output Format

```json
[
  {
    "type": "ADD_RULE",
    "target_skill": "gitlab-api",
    "content": "Before running git commands, always show current branch and repo state first",
    "reasoning": "Multiple sessions showed trust erosion when agent acted without verifying context"
  },
  {
    "type": "ADD_RULE",
    "target_skill": "branch-context-gathering",
    "content": "If branch has zero commits ahead of main, skip git log on branch, only check main history",
    "reasoning": "Recurring across 3 sessions, agent wasted time running git log on empty branch"
  }
]
```

## Rule Deduplication

Before adding new rules, curator should check if a similar rule already exists:
- Embed new rule content
- Cosine similarity against all existing rules in the target skill
- If similarity > 0.90: skip (duplicate)
- If similarity > 0.80: ask LLM "is this a duplicate or a refinement?" (if refinement, skip — curator can't update existing rules)

## Suspicious Rule Flagging

Rules with `harmful > helpful * 3` are flagged as `suspicious` in SQLite. The curator can see these stats but does NOT auto-delete or modify them. Human reviews suspicious rules.

## Staleness

Rules tagged `irrelevant` in the majority of recent sessions may be stale. Curator can flag them (mark as `stale` in SQLite) but does NOT remove them.

## Design Decisions

- **Curator frequency**: Per cluster, after N=5 new threads accumulate in that cluster. A cluster size of 5 gives enough signal for synthesis without being too noisy.
- **Context window**: A cluster maps to 1-2 skills (not 3-4). Threads in a cluster share the same domain, so skills are naturally few. Send only rules + reflections from the skills actually tagged in recent threads.
- **Re-clustering**: No separate step needed. Threads already have goals (from `goal_extractor`), goals already map to clusters (from `goal_clusterer`). Curator inherits this grouping directly.
- **Suspicious rule flagging**: Rules with `harmful > helpful * 3` flagged as `suspicious` in SQLite. Curator can see stats but does NOT auto-delete or modify them. Human review only.
- **Staleness**: Rules tagged `irrelevant` in majority of recent sessions flagged as `stale`. No auto-removal.
- **Append-only**: Curator adds new rules only. Does not modify or delete existing rules. This prevents context collapse.
