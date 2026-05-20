"""Tests for src/goal_extractor.py"""

import pytest

from src.goal_extractor import Goal, EXTRACT_GOALS_TOOL, extract_goals
from src.llm import DEFAULT_MODEL
from src.opencode_db import get_conversation_transcript
from tests.helpers import preseed_complete_tool


def _build_extract_goals_prompt(transcript: str) -> str:
    """Reproduce the exact prompt template from extract_goals."""
    return f"""Analyze this OpenCode conversation transcript and identify the distinct goals/tasks the user was trying to accomplish.

The conversation may contain multiple threads or phases, each pointing to a separate goal.

For each goal, provide:
1. A short title (title)
2. A one-line description (description)
3. Which messages (by number) relate to this goal (message_range, e.g. "msgs 1-8")

--- TRANSCRIPT ---
{transcript}"""


class TestExtractGoals:
    """Integration tests: fixture DB + pre-seeded Redis cache."""

    async def _preseed(self, redis_client, db_path, session_id, response):
        """Pre-seed cache for an extract_goals call on the given session."""
        r, seeded_keys = redis_client
        transcript = await get_conversation_transcript(session_id, db_path=db_path)
        prompt = _build_extract_goals_prompt(transcript)

        await preseed_complete_tool(
            r, seeded_keys,
            messages=[{"role": "user", "content": prompt}],
            tool=EXTRACT_GOALS_TOOL,
            model=DEFAULT_MODEL,
            max_tokens=4096,
            system=None,
            response=response,
        )

    @pytest.mark.redis
    async def test_single_goal(self, redis_client, db_path):
        response = {
            "goals": [
                {
                    "title": "Fix login bug",
                    "description": "User wants the login bug fixed",
                    "message_range": "msgs 1-4",
                },
            ]
        }
        await self._preseed(redis_client, db_path, "s1", response)

        goals = await extract_goals("s1", db_path=db_path)

        assert len(goals) == 1
        assert isinstance(goals[0], Goal)
        assert goals[0].title == "Fix login bug"
        assert goals[0].description == "User wants the login bug fixed"
        assert goals[0].message_range == "msgs 1-4"

    @pytest.mark.redis
    async def test_multiple_goals(self, redis_client, db_path):
        response = {
            "goals": [
                {
                    "title": "Add dark mode",
                    "description": "Implement dark mode styling",
                    "message_range": "msgs 1-4",
                },
                {
                    "title": "Add toggle",
                    "description": "Create a dark mode toggle component",
                    "message_range": "msgs 5-6",
                },
            ]
        }
        await self._preseed(redis_client, db_path, "s2", response)

        goals = await extract_goals("s2", db_path=db_path)

        assert len(goals) == 2
        assert goals[0].title == "Add dark mode"
        assert goals[1].title == "Add toggle"
        for g in goals:
            assert g.title
            assert g.description
            assert g.message_range

    @pytest.mark.redis
    async def test_empty_transcript_returns_empty_goals(self, redis_client, db_path):
        """Session s5 has no messages -> empty transcript -> no goals."""
        response = {"goals": []}
        await self._preseed(redis_client, db_path, "s5", response)

        goals = await extract_goals("s5", db_path=db_path)

        assert goals == []

    @pytest.mark.redis
    async def test_goal_missing_message_range_defaults_to_empty(self, redis_client, db_path):
        """Goals without message_range use .get() fallback to empty string."""
        response = {
            "goals": [
                {
                    "title": "Some goal",
                    "description": "Description",
                    # no message_range key
                },
            ]
        }
        await self._preseed(redis_client, db_path, "s1", response)

        goals = await extract_goals("s1", db_path=db_path)

        assert len(goals) == 1
        assert goals[0].message_range == ""
