"""Tests for src/reflector.py."""

import json

import pytest

from src.llm.cache import cache_key, cache_set
from src.skill_rules import RuleRow


@pytest.mark.redis
class TestReflectOnThread:
    async def test_tags_and_insights(self, redis_client):
        r, seeded_keys = redis_client
        from src.llm import DEFAULT_MODEL
        from src.reflector import REFLECT_TAG_SYSTEM, REFLECT_TAG_TOOL, _format_rules_for_prompt, reflect_on_thread

        session_id = "s1"
        summary = "User asked to fix a bug. Agent used git status, made edits, tested."
        skills = [
            ("test-skill", [
                RuleRow(id="test-skill-00001", skill_name="test-skill", content="Always show git status first", helpful_count=0, harmful_count=0),
                RuleRow(id="test-skill-00002", skill_name="test-skill", content="Run tests after changes", helpful_count=0, harmful_count=0),
            ]),
            ("other-skill", [
                RuleRow(id="other-skill-00001", skill_name="other-skill", content="Use proper indentation", helpful_count=0, harmful_count=0),
            ]),
        ]

        rules_text = _format_rules_for_prompt(skills)
        prompt = (
            f"Session: {session_id}\n\n"
            f"Conversation summary:\n{summary}\n\n"
            f"Rules:\n{rules_text}\n\n"
            f"Evaluate each rule and extract new insights."
        )
        response = {
            "rule_tags": [
                {"rule_id": "test-skill-00001", "tag": "followed_helpful"},
                {"rule_id": "test-skill-00002", "tag": "not_followed"},
                {"rule_id": "other-skill-00001", "tag": "irrelevant"},
            ],
            "insights_by_skill": {
                "test-skill": ["Always check assertion output first"],
                "other-skill": [],
            },
        }
        key = cache_key("complete_tool", messages=[{"role": "user", "content": prompt}], tool=REFLECT_TAG_TOOL, model=DEFAULT_MODEL, max_tokens=4096, system=REFLECT_TAG_SYSTEM)
        await cache_set(r, key, json.dumps(response, sort_keys=True))
        seeded_keys.append(key)

        result = await reflect_on_thread(session_id=session_id, thread_summary=summary, skills=skills)
        assert len(result.rule_tags) == 3
        assert result.rule_tags[0].tag == "followed_helpful"
        assert "test-skill" in result.insights_by_skill
        assert len(result.insights_by_skill["test-skill"]) == 1

    async def test_empty_skills(self, redis_client):
        r, seeded_keys = redis_client
        from src.llm import DEFAULT_MODEL
        from src.reflector import REFLECT_TAG_SYSTEM, REFLECT_TAG_TOOL, _format_rules_for_prompt, reflect_on_thread

        session_id = "s2"
        summary = "Simple coding session."
        rules_text = _format_rules_for_prompt([])
        prompt = (
            f"Session: {session_id}\n\n"
            f"Conversation summary:\n{summary}\n\n"
            f"Rules:\n{rules_text}\n\n"
            f"Evaluate each rule and extract new insights."
        )
        response = {"rule_tags": [], "insights_by_skill": {}}
        key = cache_key("complete_tool", messages=[{"role": "user", "content": prompt}], tool=REFLECT_TAG_TOOL, model=DEFAULT_MODEL, max_tokens=4096, system=REFLECT_TAG_SYSTEM)
        await cache_set(r, key, json.dumps(response, sort_keys=True))
        seeded_keys.append(key)

        result = await reflect_on_thread(session_id=session_id, thread_summary=summary, skills=[])
        assert result.rule_tags == []
        assert result.insights_by_skill == {}


@pytest.mark.redis
class TestReflectInsightOnly:
    async def test_insights_extracted(self, redis_client):
        r, seeded_keys = redis_client
        from src.llm import DEFAULT_MODEL
        from src.reflector import REFLECT_INSIGHT_SYSTEM, REFLECT_INSIGHT_TOOL, reflect_insight_only

        session_id = "s1"
        summary = "User asked complex queries. Agent cached responses."
        skill_names = ["test-skill", "other-skill"]

        skills_text = "\n".join(f"- {s}" for s in skill_names)
        prompt = (
            f"Session: {session_id}\n\n"
            f"Conversation summary:\n{summary}\n\n"
            f"Active skills:\n{skills_text}\n\n"
            f"Extract new insights grouped by skill."
        )
        response = {
            "insights_by_skill": {
                "test-skill": ["Found that caching responses avoids repeated API calls"],
                "other-skill": [],
            },
        }
        key = cache_key("complete_tool", messages=[{"role": "user", "content": prompt}], tool=REFLECT_INSIGHT_TOOL, model=DEFAULT_MODEL, max_tokens=4096, system=REFLECT_INSIGHT_SYSTEM)
        await cache_set(r, key, json.dumps(response, sort_keys=True))
        seeded_keys.append(key)

        result = await reflect_insight_only(session_id=session_id, thread_summary=summary, skill_names=skill_names)
        assert result.rule_tags == []
        assert "test-skill" in result.insights_by_skill
        assert result.insights_by_skill["test-skill"][0] == "Found that caching responses avoids repeated API calls"
