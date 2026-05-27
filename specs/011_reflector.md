# 011 — Reflector (`reflector.py`)

## Purpose

Runs per conversation thread. Tags which skill rules were relevant, followed, helpful, or harmful. Extracts new insights not covered by existing rules.

## When It Runs

- **First-time run**: Reflector in "no-tag mode" — only extracts insights (friction points, what worked, what didn't). No rules exist yet.
- **Periodic run**: Reflector in "tag mode" — tags existing rules + extracts insights.

## API

### `reflect_on_thread(session_id, thread_summary, skills: list[tuple[str, list[RuleRow]]], model) → Reflection`

```python
@dataclass
class Reflection:
    session_id: str
    rule_tags: list[RuleTag]              # per-rule tag (flat, all skills)
    insights_by_skill: dict[str, list[str]]  # new insights, grouped by skill
```

Each skill invoked in the session provides its `## Rules` section. The reflector evaluates the conversation against all these rules at once. New insights are organized by which skill they belong to.

### `reflect_insight_only(session_id, thread_summary, model) → Reflection`

First-time mode. No rules to tag (rule_tags = []). Outputs insights_by_skill using skill names from the `tool:skill` parts in the session.

## LLM Prompt (tag mode)

```
You are a skill rule evaluator. You will receive:
1. A conversation summary
2. A set of rules from skills that were active during this conversation

For each rule, tag it as one of:
- irrelevant: The rule doesn't apply to this conversation
- followed_helpful: The agent followed this rule and it helped
- followed_harmful: The agent followed this rule but it caused harm or a bad outcome
- not_followed: The rule was relevant but the agent didn't follow it

Also identify new insights — patterns or lessons from this conversation that aren't covered by the existing rules.
```

## Output Format

```json
{
  "rule_tags": [
    {"rule_id": "gitlab-api-00001", "tag": "followed_helpful"},
    {"rule_id": "gitlab-api-00003", "tag": "not_followed"},
    {"rule_id": "branch-context-00005", "tag": "irrelevant"}
  ],
  "insights_by_skill": {
    "mcp-debugging": [
      "Session creation races under high concurrency, add per-server lock"
    ],
    "gitlab-api": [
      "Always verify repo context before running git commands"
    ]
  }
}
```

Rule tags are flat (all skills merged) because the rule_id itself encodes the skill. Insights are grouped by skill so curator gets pre-organized input.

## Skill Detection

Skills invoked during a session are recorded as **tool parts** in OpenCode's DB:

```json
{
  "type": "tool",
  "tool": "skill",
  "callID": "...",
  "state": {
    "status": "completed",
    "input": {"name": "BuzzLLM Gateway"},
    "output": "<full SKILL.md content>",
    "title": "Loaded skill: BuzzLLM Gateway"
  }
}
```

Extraction query:
```sql
SELECT DISTINCT json_extract(data, '$.state.input.name')
FROM part
WHERE json_extract(data, '$.type') = 'tool'
  AND json_extract(data, '$.tool') = 'skill'
  AND message_id IN (SELECT id FROM message WHERE session_id = ?)
```

~436 invocations across 40+ distinct skills in the DB. Reliable signal.

## Design Decisions

- **Single-shot rule feeding**: Feed all rules from all skills invoked in the thread to the reflector at once. The LLM gets the conversation summary + all relevant rules → tags each.
- **Alternative (unexplored)**: Per-skill sequential evaluation — evaluate rules for each skill one at a time. May be more precise but adds LLM calls. Not explored yet.
- **Fallback**: If a session has no skill tool calls (older sessions), semantic search against all skill descriptions to find relevant skills.
