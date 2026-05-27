"""Tests for src/curator.py."""

import json
from pathlib import Path

import pytest

from src.llm.cache import cache_key, cache_set
from src.skill_registry import SkillInfo
from src.skill_rules import RuleStats


_CURATOR_RESPONSE = [
    {"type": "ADD_RULE", "content": "Before running git commands, always show current branch and repo state first", "reasoning": "Multiple sessions showed trust erosion when agent acted without verifying context"},
    {"type": "ADD_RULE", "content": "If branch has zero commits ahead of main, skip git log, only check main history", "reasoning": "Recurring across sessions, agent wasted time checking empty branch"},
]


@pytest.mark.redis
class TestCurateSkill:
    async def test_adds_rules_from_insights(self, redis_client):
        r, seeded_keys = redis_client
        from src.llm import DEFAULT_MODEL
        from src.curator import CURATOR_SYSTEM, curate_skill

        current = SkillInfo(
            name="git-api",
            description="Interact with git APIs",
            path=Path("git-api/SKILL.md"),
            content="---\nname: git-api\ndescription: Interact with git APIs\n---\n\n## Workflow\n1. Check branch\n\n## Rules\n- [git-api-00001] Always verify repo context before editing\n",
            has_rules=True,
        )
        stats = RuleStats(total=1, high_performing=1, suspicious=0, unused=0, average_helpful=3.0, average_harmful=0.0)
        insights = [
            "Agent should verify context before running git commands",
            "Don't check git log on empty branches",
        ]

        insights_text = "\n".join(f"- {i}" for i in insights)
        stats_text = (
            f"Rule stats: 1 total, 1 high-performing, 0 suspicious, 0 unused, "
            f"avg helpful=3.0, avg harmful=0.0"
        )
        prompt = (
            f"Skill: git-api\n\n"
            f"Current skill content:\n```markdown\n{current.content}\n```\n\n"
            f"New insights:\n{insights_text}\n\n"
            f"{stats_text}\n\n"
            f"Synthesize new rules from these insights."
        )
        key = cache_key("complete", messages=[{"role": "user", "content": prompt}], model=DEFAULT_MODEL, max_tokens=4096, system=CURATOR_SYSTEM)
        await cache_set(r, key, json.dumps(_CURATOR_RESPONSE))
        seeded_keys.append(key)

        ops = await curate_skill(skill_name="git-api", insights=insights, current_skill=current, rule_stats=stats)
        assert len(ops) == 2
        assert ops[0].type == "ADD_RULE"
        assert ops[0].target_skill == "git-api"
        assert "git" in ops[0].content.lower()

    async def test_dedup_skips_similar_rules(self, redis_client):
        """Rule identical to existing is skipped by _is_duplicate check."""
        r, seeded_keys = redis_client
        from src.llm import DEFAULT_MODEL
        from src.curator import CURATOR_SYSTEM, curate_skill

        current = SkillInfo(
            name="test-skill",
            description="Test",
            path=Path("test-skill/SKILL.md"),
            content="---\nname: test-skill\n---\n\n## Rules\n- [test-skill-00001] Always verify repo context before editing files\n",
            has_rules=True,
        )
        stats = RuleStats(total=1, high_performing=0, suspicious=0, unused=1, average_helpful=0.0, average_harmful=0.0)
        # LLM response where rule #1 is almost identical to existing
        response = [
            {"type": "ADD_RULE", "content": "Always verify repo context before editing files", "reasoning": "duplicate"},
            {"type": "ADD_RULE", "content": "Use minimal diffs for large refactors", "reasoning": "new insight"},
        ]

        insights_text = "\n".join(f"- {i}" for i in ["duplicate insight", "new insight"])
        stats_text = "Rule stats: 1 total, 0 high-performing, 0 suspicious, 1 unused, avg helpful=0.0, avg harmful=0.0"
        prompt = (
            f"Skill: test-skill\n\n"
            f"Current skill content:\n```markdown\n{current.content}\n```\n\n"
            f"New insights:\n{insights_text}\n\n"
            f"{stats_text}\n\n"
            f"Synthesize new rules from these insights."
        )
        key = cache_key("complete", messages=[{"role": "user", "content": prompt}], model=DEFAULT_MODEL, max_tokens=4096, system=CURATOR_SYSTEM)
        await cache_set(r, key, json.dumps(response))
        seeded_keys.append(key)

        ops = await curate_skill(skill_name="test-skill", insights=["duplicate insight", "new insight"], current_skill=current, rule_stats=stats)
        # First rule should be filtered out by dedup
        assert len(ops) == 1
        assert "minimal diffs" in ops[0].content.lower()

    async def test_no_insights_returns_empty(self, redis_client):
        r, seeded_keys = redis_client
        from src.llm import DEFAULT_MODEL
        from src.curator import CURATOR_SYSTEM, curate_skill

        current = SkillInfo(
            name="empty-skill", description="No rules",
            path=Path("empty-skill/SKILL.md"),
            content="---\nname: empty-skill\n---\n\n## Workflow\nDo stuff.\n",
            has_rules=False,
        )
        stats = RuleStats(total=0, high_performing=0, suspicious=0, unused=0, average_helpful=0.0, average_harmful=0.0)

        insights_text = ""  # no insights
        stats_text = "Rule stats: 0 total, 0 high-performing, 0 suspicious, 0 unused, avg helpful=0.0, avg harmful=0.0"
        prompt = (
            f"Skill: empty-skill\n\n"
            f"Current skill content:\n```markdown\n{current.content}\n```\n\n"
            f"New insights:\n{insights_text}\n\n"
            f"{stats_text}\n\n"
            f"Synthesize new rules from these insights."
        )
        response = "[]"
        key = cache_key("complete", messages=[{"role": "user", "content": prompt}], model=DEFAULT_MODEL, max_tokens=4096, system=CURATOR_SYSTEM)
        await cache_set(r, key, response)
        seeded_keys.append(key)

        ops = await curate_skill(skill_name="empty-skill", insights=[], current_skill=current, rule_stats=stats)
        assert ops == []
