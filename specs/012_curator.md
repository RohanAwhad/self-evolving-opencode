# 012 — Curator (`curator.py`)

## Purpose

Runs per skill within a goal cluster, AFTER all threads in that cluster have been reflected. Synthesizes new rules from reflector insights. Always ADD-only — never modifies or deletes existing rules.

## When It Runs

- **First-time run**: Curator in "full builder" mode — produces complete skill from scratch (frontmatter + workflow + rules). One curator call per cluster.
- **Periodic run**: Curator in "ADD mode" — after all threads in a cluster are reflected, aggregate insights by skill, then one curator call per skill that has insights.

## Flow

```
Cluster threads (N threads)
  → Reflector per thread → insights_by_skill
  → Aggregate across threads:
      {
        "mcp-debugging": [insight1, insight2, insight3],
        "gitlab-api": [insight1]
      }
  → For each skill with insights:
      Curator(skill_name, insights, current_skill_content, rule_stats)
        → [ADD_RULE, ADD_RULE, ...]
```

If a cluster has N clusters with M skills each: N×M curator calls.

## API

### `curate_skill(skill_name, insights: list[str], current_skill: SkillInfo, rule_stats: RuleStats, model) → list[CuratorOperation]`

```python
@dataclass
class CuratorOperation:
    type: Literal["ADD_RULE"]
    target_skill: str
    content: str
    reasoning: str

@dataclass
class CuratorInput:
    skill_name: str
    insights: list[str]               # all new insights for this skill from this cluster
    current_skill: SkillInfo          # the skill's current content (for context, dedup)
    rule_stats: RuleStats             # stats from SQLite (helpful/harmful counts)
```

Curator does NOT see raw threads. It only sees insights that the reflector already extracted and organized by skill.

### First-time variant: `curate_new_skill(cluster_id, goals, summaries, model) → str`

Full builder mode. Produces complete SKILL.md content (frontmatter + workflow + rules). Used during initial run when a cluster maps to "new" skill.

## LLM Prompt (periodic ADD mode)

```
You are a skill curator. You will receive:
1. A skill (its current rules and workflow)
2. A list of new insights from recent conversation threads
3. Statistics on existing rules (helpful/harmful counts)

Your job:
- Synthesize new insights into concrete, actionable rules
- Do NOT modify or delete existing rules (append-only)
- If an insight is already covered by an existing rule, skip it
- If multiple insights say the same thing, combine them into one rule

Output: JSON array of ADD_RULE operations.
```

## Output Format

```json
[
  {
    "type": "ADD_RULE",
    "content": "Before running git commands, always show current branch and repo state first",
    "reasoning": "Multiple sessions showed trust erosion when agent acted without verifying context"
  },
  {
    "type": "ADD_RULE",
    "content": "If branch has zero commits ahead of main, skip git log, only check main history",
    "reasoning": "Recurring across 3 sessions, agent wasted time checking empty branch"
  }
]
```

Note: `target_skill` is implicit (curator is called per-skill).

## Rule Deduplication

Before adding new rules:
- Embed new rule content
- Cosine similarity against all existing rules in the target skill
- If similarity > 0.90: skip (duplicate)
- If similarity > 0.80: ask LLM "is this a duplicate or a refinement?" (if refinement, skip — curator can't update existing rules)

## Suspicious Rule Flagging

Rules with `harmful > helpful * 3` are flagged as `suspicious` in SQLite. The curator can see these stats but does NOT auto-delete or modify them. Human reviews suspicious rules.

## Staleness

Rules tagged `irrelevant` in majority of recent sessions may be stale. Curator can flag them (mark as `stale` in SQLite) but does NOT remove them.

## Design Decisions

- **One curator call per skill** — not per cluster. Insights arrive already grouped by skill from the reflector.
- **No N=5 trigger** — curator runs immediately after all threads in the cluster are reflected. The cluster IS the batch boundary.
- **Always ADD-only** — new rules appended, existing rules never modified. Prevents context collapse.
- **Curator never sees raw threads** — only reflector-extracted insights. Clean separation of concerns.