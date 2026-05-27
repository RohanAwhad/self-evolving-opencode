"""Smoke tests for the skill evolution pipeline.

Marked @pytest.mark.live — run with:
    DRY_RUN=1 uv run pytest tests/test_skill_evolution.py -m live -v

All use real LLM APIs. Pre-seeded Redis cache ensures deterministic results
for the synthesizer and curator (the heaviest calls).
"""

import json

import pytest

from src.llm.cache import cache_key, cache_set
from src.skill_registry import (
    SkillDecision,
    decide_new_or_update,
)

# ---------------------------------------------------------------------------
# Smoke 1: synthesize-bootstrap — no existing skills, create new
# ---------------------------------------------------------------------------


@pytest.mark.redis
@pytest.mark.live
class TestSynthesizeBootstrap:
    async def test_creates_new_skill_for_cluster(self, redis_client, temp_skills_dir, skills_db_path, db_path):
        """Synthesizer creates a new skill when no existing skill matches."""
        r, seeded_keys = redis_client

        # The synthesizer needs: extract_goals (already in DB), summarize, decide, synthesize
        # Sessions s1-s5 in fixture have simple messages. s1 has a "Fix the bug" goal.
        # We pre-seed the decide_new_or_update and synthesize_skill calls.

        from src.skill_synthesizer import SYNTHESIZE_NEW_SYSTEM, synthesize_skill
        from src.goal_extractor import Goal
        from src.llm import DEFAULT_MODEL

        # Pre-seed a seed skill SKILL.md
        skill_output = "\n".join([
            "---",
            "name: fix-the-bug",
            "description: Diagnose and fix bugs in the codebase",
            "---",
            "",
            "## Workflow",
            "### Phase 1: Reproduce",
            "1. Read error logs and stack traces",
            "2. Identify the failing code path",
            "",
            "## Rules",
            "- [fix-the-bug-00001] Always read the full error log before editing any file",
        ])

        goals = [Goal(title="Fix the bug", description="Fix the bug in the application", message_range="msgs 1-10")]
        summaries = ["User reported a bug. Agent read logs, identified issue, fixed file."]

        from src.skill_synthesizer import synthesize_skill
        result = await synthesize_skill(cluster_id=1, goals=goals, thread_summaries=summaries)
        assert "---" in result
        assert "name:" in result

    async def test_pipeline_synthesizer_standalone(self, redis_client, temp_skills_dir, skills_db_path, db_path):
        """Verify the synthesizer's core functions work together."""
        r, seeded_keys = redis_client
        from src.llm import DEFAULT_MODEL
        from src.skill_synthesizer import synthesize_skill
        from src.goal_extractor import Goal

        # Pre-seed synthesize for new skill
        goals = [Goal(title="Test bug", description="Find and fix testing bugs", message_range="msgs 1-5")]
        summaries = ["Session about fixing tests."]

        from src.skill_synthesizer import SYNTHESIZE_NEW_SYSTEM
        goals_text = f"- {goals[0].title}: {goals[0].description}"
        threads_text = f"## Thread 1\n{summaries[0]}"
        prompt = f"Goals (cluster 1):\n{goals_text}\n\nThread summaries:\n{threads_text}\n\nSynthesize a complete SKILL.md."
        output = "---\nname: test-bug\ndescription: Fix testing bugs\n---\n\n## Workflow\n1. Check test output\n\n## Rules\n- [test-bug-00001] Read error output first\n"
        key = cache_key("complete", messages=[{"role": "user", "content": prompt}], model=DEFAULT_MODEL, max_tokens=8192, system=SYNTHESIZE_NEW_SYSTEM)
        await cache_set(r, key, output)
        seeded_keys.append(key)

        result = await synthesize_skill(cluster_id=1, goals=goals, thread_summaries=summaries)
        assert "test-bug" in result
        assert "## Workflow" in result
        assert "## Rules" in result


# ---------------------------------------------------------------------------
# Smoke 2: synthesize-existing — match existing, update workflow
# ---------------------------------------------------------------------------


@pytest.mark.redis
@pytest.mark.live
class TestSynthesizeExisting:
    async def test_updates_workflow_preserves_rules(self, redis_client, temp_skills_dir):
        """Synthesizer updates workflow without touching the ## Rules section."""
        r, seeded_keys = redis_client
        from src.llm import DEFAULT_MODEL
        from src.skill_registry import SkillInfo
        from src.skill_synthesizer import SYNTHESIZE_UPDATE_SYSTEM, synthesize_skill
        from src.goal_extractor import Goal
        from pathlib import Path

        existing = SkillInfo(
            name="existing-skill",
            description="Existing skill",
            path=Path("existing-skill/SKILL.md"),
            content="---\nname: existing-skill\ndescription: Old\n---\n\n## Workflow\nOld\n\n## Rules\n- [existing-skill-00001] Preserve me\n",
            has_rules=True,
        )
        goals = [Goal(title="Refine workflow", description="Improve existing skill workflow", message_range="msgs 1-5")]
        summaries = ["Session refined the workflow with better steps."]

        threads_text = "## Thread 1\nSession refined the workflow with better steps."
        prompt = f"Existing skill:\n```markdown\n{existing.content}\n```\n\nNew threads:\n{threads_text}\n\nUpdate the ## Workflow section. Preserve ## Rules exactly."
        updated = "---\nname: existing-skill\ndescription: Improved\n---\n\n## Workflow\nBetter workflow\n\n## Rules\n- [existing-skill-00001] Preserve me\n"
        key = cache_key("complete", messages=[{"role": "user", "content": prompt}], model=DEFAULT_MODEL, max_tokens=8192, system=SYNTHESIZE_UPDATE_SYSTEM)
        await cache_set(r, key, updated)
        seeded_keys.append(key)

        result = await synthesize_skill(cluster_id=1, goals=goals, thread_summaries=summaries, existing_skill=existing)
        assert "Better workflow" in result
        assert "Preserve me" in result


# ---------------------------------------------------------------------------
# Smoke 3: evolve-bootstrap — no rules, insight-only mode
# ---------------------------------------------------------------------------


@pytest.mark.redis
@pytest.mark.live
class TestEvolveBootstrap:
    async def test_insight_only_no_rules(self, redis_client):
        """Reflector in insight-only mode when skill has no rules."""
        r, seeded_keys = redis_client
        from src.llm import DEFAULT_MODEL
        from src.reflector import REFLECT_INSIGHT_SYSTEM, reflect_insight_only

        session_id = "s1"
        summary = "User debugged a failing test. Agent ran tests, found error, fixed assertion."
        skill_names = ["test-skill"]

        skills_text = "- test-skill"
        prompt = f"Session: {session_id}\n\nConversation summary:\n{summary}\n\nActive skills:\n{skills_text}\n\nExtract new insights grouped by skill."
        response = json.dumps({"insights_by_skill": {"test-skill": ["Always check assertion output first"]}})
        key = cache_key("complete", messages=[{"role": "user", "content": prompt}], model=DEFAULT_MODEL, max_tokens=4096, system=REFLECT_INSIGHT_SYSTEM)
        await cache_set(r, key, response)
        seeded_keys.append(key)

        result = await reflect_insight_only(session_id=session_id, thread_summary=summary, skill_names=skill_names)
        assert result.rule_tags == []
        assert "test-skill" in result.insights_by_skill


# ---------------------------------------------------------------------------
# Smoke 4: evolve-existing — rules exist, tag + curate
# ---------------------------------------------------------------------------


@pytest.mark.redis
@pytest.mark.live
class TestEvolveExisting:
    async def test_tag_and_curate(self, redis_client):
        """Full reflector → curator flow with existing rules."""
        r, seeded_keys = redis_client
        from src.llm import DEFAULT_MODEL
        from src.reflector import REFLECT_TAG_SYSTEM, _format_rules_for_prompt, reflect_on_thread
        from src.curator import CURATOR_SYSTEM, curate_skill
        from src.skill_registry import SkillInfo
        from src.skill_rules import RuleRow, RuleStats
        from pathlib import Path

        session_id = "s1"
        summary = "User fixed a bug in CI pipeline. Agent ran git status, checked branch, made fix, pushed."
        skills = [
            ("git-api", [
                RuleRow(id="git-api-00001", skill_name="git-api", content="Always verify repo context before editing", helpful_count=3, harmful_count=1),
                RuleRow(id="git-api-00002", skill_name="git-api", content="Use minimal diffs", helpful_count=5, harmful_count=0),
            ]),
        ]

        # Pre-seed reflector
        rules_text = _format_rules_for_prompt(skills)
        reflect_prompt = f"Session: {session_id}\n\nConversation summary:\n{summary}\n\nRules:\n{rules_text}\n\nEvaluate each rule and extract new insights."
        reflect_response = json.dumps({
            "rule_tags": [
                {"rule_id": "git-api-00001", "tag": "followed_helpful"},
                {"rule_id": "git-api-00002", "tag": "irrelevant"},
            ],
            "insights_by_skill": {
                "git-api": ["Push to correct branch after fix, verify it works in CI"],
            },
        })
        key = cache_key("complete", messages=[{"role": "user", "content": reflect_prompt}], model=DEFAULT_MODEL, max_tokens=4096, system=REFLECT_TAG_SYSTEM)
        await cache_set(r, key, reflect_response)
        seeded_keys.append(key)

        reflection = await reflect_on_thread(session_id=session_id, thread_summary=summary, skills=skills)
        assert len(reflection.rule_tags) == 2
        assert "git-api" in reflection.insights_by_skill

        # Pre-seed curator
        current = SkillInfo(
            name="git-api",
            description="Git API interactions",
            path=Path("git-api/SKILL.md"),
            content="---\nname: git-api\n---\n\n## Workflow\n1. Check branch\n\n## Rules\n- [git-api-00001] Always verify repo context\n- [git-api-00002] Use minimal diffs\n",
            has_rules=True,
        )
        stats = RuleStats(total=2, high_performing=1, suspicious=0, unused=1, average_helpful=4.0, average_harmful=0.5)
        insights = ["Push to correct branch after fix, verify it works in CI"]
        insights_text = f"- {insights[0]}"
        stats_text = "Rule stats: 2 total, 1 high-performing, 0 suspicious, 1 unused, avg helpful=4.0, avg harmful=0.5"
        curator_prompt = (
            f"Skill: git-api\n\n"
            f"Current skill content:\n```markdown\n{current.content}\n```\n\n"
            f"New insights:\n{insights_text}\n\n"
            f"{stats_text}\n\n"
            f"Synthesize new rules from these insights."
        )
        curator_response = json.dumps([
            {"type": "ADD_RULE", "content": "Verify the fix in CI before closing the issue", "reasoning": "Agent pushed a fix without checking CI, causing regression"},
        ])
        key2 = cache_key("complete", messages=[{"role": "user", "content": curator_prompt}], model=DEFAULT_MODEL, max_tokens=4096, system=CURATOR_SYSTEM)
        await cache_set(r, key2, curator_response)
        seeded_keys.append(key2)

        ops = await curate_skill(skill_name="git-api", insights=insights, current_skill=current, rule_stats=stats)
        assert len(ops) == 1
        assert ops[0].target_skill == "git-api"
        assert "CI" in ops[0].content
