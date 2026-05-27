# 008 — Skill Synthesizer (`skill_synthesizer.py`)

## Purpose

Generates full skills from clusters of conversation threads. Extracts the common workflow across multiple threads at once. Runs independently of the curator (separate queue, separate `processed_` table). Handles both new skill creation and existing skill workflow updates.

## When It Runs

Part of the synthesizer queue in `--evolve`. Processes sessions from oldest to newest. For each goal cluster, examines ≤10 threads to extract the workflow pattern.

## API

### `synthesize_skill(cluster_id, goals: list[Goal], thread_summaries: list[str], existing_skill: SkillInfo | None, model) → str`

Input:
- Cluster goals + thread summaries (≤10 threads)
- `existing_skill` = None → create new skill
- `existing_skill` = SkillInfo → update existing skill's workflow

Output: complete SKILL.md content (frontmatter + workflow + rules section if any)

```python
@dataclass
class SkillFrontmatter:
    name: str
    description: str
```

## LLM Prompt

```
You are a skill synthesizer. You will receive:
1. A cluster of related user goals
2. Summaries of conversation threads that achieved these goals
3. (optional) An existing skill to update

Your job:
- Extract the common workflow pattern across all threads
- Identify what the user consistently does, in what order
- Identify decision points, checkpoints, and repeatable patterns
- If updating an existing skill, refine the workflow section with new insights
- Preserve the ## Rules section if it exists (curator owns that)
- Rule IDs go in markdown. Counters do NOT.

Sections to generate:
- ## Workflow: step-by-step instructions, decision points, checkpoints
- ## Rules: only if creating a new skill (use rule IDs without counters)
```

## Output

Full SKILL.md content:

```markdown
---
name: mcp-debugging
description: Diagnose and fix MCP connection, session, and concurrency issues in Langflow. Use when debugging MCP server failures, session timeouts, or high-concurrency connection drops.
---

## Workflow

### Phase 1: Verify the basics
1. Check that both servers are running (Langflow + MCP toolserver)
2. Confirm the correct port and URL for the MCP server
3. Test a single request to isolate server-side vs client-side issues

### Phase 2: Trace the failure path
...

## Rules
- [mcp-debugging-00001] For timeout errors, verify the real runtime port first, then check retry settings
- [mcp-debugging-00002] Validate the full target load before declaring a fix complete
```

## Context Window Management

Max 10 thread summaries per cluster. If cluster has more, sample:
- Threads where goals were achieved (success patterns)
- Threads with rich tool usage and decision points
- Avoid threads with only trivial/failed interactions

## Relationship to Curator

- **Synthesizer**: operates on raw threads, extracts the workflow. Creates or updates the `## Workflow` section.
- **Curator**: operates on reflector insights (post-processing), adds rules to the `## Rules` section.
- No shared state — they write to different sections of the same SKILL.md.

## Skill Routing

Before synthesizer runs, semantic search against `~/.claude/skills/` determines:
- No close match → `synthesize_skill(existing_skill=None)` → new skill
- Close match → `synthesize_skill(existing_skill=match)` → update workflow
- Synthesizer always runs, always outputs full content. Never skips.

## Testing

Same pattern as `goal_extractor`/`goal_checker`: pre-seed Redis cache with LLM responses. Assert on output content. DRY_RUN=1 prevents disk writes. With small fixture clusters and pre-seeded summaries.