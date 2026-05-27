"""Tests for src/skill_synthesizer.py."""

import pytest

from src.goal_extractor import Goal
from src.llm.cache import cache_key, cache_set

_SAMPLE_OUTPUT = """---
name: mcp-debugging
description: Diagnose and fix MCP connection, session, and concurrency issues.
---

## Workflow
### Phase 1: Verify basics
1. Check server is running
2. Confirm correct port

## Rules
- [mcp-debugging-00001] For timeout errors, verify the real runtime port first
"""


@pytest.mark.redis
class TestSynthesizeSkill:
    _GOALS = [
        Goal(title="Debug MCP failures", description="Fix MCP connection errors", message_range="msgs 1-10"),
        Goal(title="Handle MCP timeouts", description="Add retry logic for timeouts", message_range="msgs 1-8"),
    ]
    _SUMMARIES = [
        "Thread 1: Debugged MCP connection failures. Port was wrong in config.",
        "Thread 2: Added exponential backoff for MCP timeouts. Tested with high concurrency.",
    ]

    async def test_synthesize_new_skill(self, redis_client):
        """Cache hit: returns complete SKILL.md with frontmatter."""
        r, seeded_keys = redis_client
        from src.skill_synthesizer import SYNTHESIZE_NEW_SYSTEM, synthesize_skill
        from src.llm import DEFAULT_MODEL

        goals_text = "\n".join(f"- {g.title}: {g.description}" for g in self._GOALS)
        threads_text = "\n\n---\n\n".join(
            f"## Thread {i + 1}\n{s}" for i, s in enumerate(self._SUMMARIES)
        )
        prompt = (
            f"Goals (cluster 1):\n{goals_text}\n\n"
            f"Thread summaries:\n{threads_text}\n\n"
            f"Synthesize a complete SKILL.md."
        )
        key = cache_key("complete", messages=[{"role": "user", "content": prompt}], model=DEFAULT_MODEL, max_tokens=8192, system=SYNTHESIZE_NEW_SYSTEM)
        await cache_set(r, key, _SAMPLE_OUTPUT)
        seeded_keys.append(key)

        result = await synthesize_skill(cluster_id=1, goals=self._GOALS, thread_summaries=self._SUMMARIES)
        assert "---" in result
        assert "name: mcp-debugging" in result
        assert "description:" in result
        assert "## Workflow" in result
        assert "## Rules" in result

    async def test_synthesize_update_existing(self, redis_client):
        """Cache hit: preserves rules, updates workflow."""
        r, seeded_keys = redis_client
        from src.skill_registry import SkillInfo
        from src.skill_synthesizer import SYNTHESIZE_UPDATE_SYSTEM, synthesize_skill
        from src.llm import DEFAULT_MODEL
        from pathlib import Path

        existing = SkillInfo(
            name="mcp-debugging",
            description="Old description",
            path=Path("mcp-debugging/SKILL.md"),
            content="---\nname: mcp-debugging\ndescription: Old\n---\n\n## Workflow\nOld workflow\n\n## Rules\n- [mcp-debugging-00001] Old rule\n",
            has_rules=True,
        )

        threads_text = "\n\n---\n\n".join(
            f"## Thread {i + 1}\n{s}" for i, s in enumerate(self._SUMMARIES)
        )
        prompt = (
            f"Existing skill:\n```markdown\n{existing.content}\n```\n\n"
            f"New threads:\n{threads_text}\n\n"
            f"Update the ## Workflow section. Preserve ## Rules exactly."
        )
        updated_output = "---\nname: mcp-debugging\ndescription: Updated desc\n---\n\n## Workflow\nUpdated workflow\n\n## Rules\n- [mcp-debugging-00001] Old rule\n"
        key = cache_key("complete", messages=[{"role": "user", "content": prompt}], model=DEFAULT_MODEL, max_tokens=8192, system=SYNTHESIZE_UPDATE_SYSTEM)
        await cache_set(r, key, updated_output)
        seeded_keys.append(key)

        result = await synthesize_skill(cluster_id=1, goals=self._GOALS, thread_summaries=self._SUMMARIES, existing_skill=existing)
        assert "Updated workflow" in result
        assert "Old rule" in result
