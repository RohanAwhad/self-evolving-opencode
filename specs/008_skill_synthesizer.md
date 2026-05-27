# 008 — Skill Synthesizer (`skill_synthesizer.py`)

## Purpose

LLM generates Claude skill files from goal cluster data. Two-phase: frontmatter only (cheap), then full skill (only after destination is known).

## API

### Phase A: `synthesize_frontmatter(cluster_id, goals, summaries, model) → SkillFrontmatter`

Input: cluster ID, list of Goal objects, list of conversation summaries (markdown)
Output: YAML frontmatter with name + description

```python
@dataclass
class SkillFrontmatter:
    name: str           # kebab-case skill name
    description: str    # 1-2 sentences, when to use, what it does
```

LLM prompt: "Given these related goals and their conversation summaries, generate a concise skill name and description."

### Phase D: `synthesize_full_skill(cluster_id, goals, summaries, existing_skill_content, model) → str`

Input: cluster data + optionally existing skill content (if updating)
Output: complete SKILL.md content with:
- YAML frontmatter (name, description)
- `## Workflow` section (step-by-step instructions derived from conversations)
- `## Rules` section (bullet list with IDs, no counters)

```markdown
## Rules
- [skillname-00001] Always verify repo context before editing files
- [skillname-00002] When user says "checkout and wait", only switch branches and stop
```

LLM prompt: "You are synthesizing a Claude skill from real conversation data. The skill should encode workflows, patterns, and rules that the conversations reveal. Rules should be specific, actionable, and derived from what actually worked or failed in the conversations."

## Rule ID Generation

- Format: `{skillname}-{NNNNN}` (e.g., `gitlab-api-00001`, `gitlab-api-00002`)
- Sequential within each skill, starting from max existing ID + 1
- New rules start with `helpful=0 harmful=0` in SQLite (counters not in markdown)

## "Update" Mode

When updating an existing skill:
- Preserve existing `## Rules` section, append new rules with fresh IDs
- May refine/improve `## Workflow` section
- Frontmatter may be updated if skill scope has broadened

## Context Window Management

For large clusters (>5 conversations), sample the most representative threads:
- Prefer threads where goals were achieved (positive examples)
- Prefer threads with rich friction points (negative examples + corrections)
- Max: `max_threads_per_cluster` (default: 5)

---

## Open Questions

*None yet — will flag if they arise during implementation.*
