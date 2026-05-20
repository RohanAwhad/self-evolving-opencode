"""Tests for src/goal_checker.py"""

import json

import pytest

from src.goal_checker import (
    GoalResult,
    GOAL_RESULT_TOOL,
    SYSTEM_PROMPT,
    _format_messages_for_prompt,
    check_goal_achieved,
)
from src.llm import DEFAULT_MODEL
from tests.conftest import preseed_complete_tool


# ── Unit tests: _format_messages_for_prompt ──────────────────────────


class TestFormatMessagesForPrompt:
    def test_string_content(self):
        msgs = [{"role": "user", "content": "hello world"}]
        result = _format_messages_for_prompt(msgs)
        assert result == "--- Message 1 (user) ---\nhello world"

    def test_multiple_messages_joined_by_double_newline(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = _format_messages_for_prompt(msgs)
        parts = result.split("\n\n")
        assert len(parts) == 2
        assert parts[0] == "--- Message 1 (user) ---\nhello"
        assert parts[1] == "--- Message 2 (assistant) ---\nhi there"

    def test_list_content_text_blocks(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ],
            }
        ]
        result = _format_messages_for_prompt(msgs)
        assert "first\nsecond" in result

    def test_list_content_plain_strings(self):
        msgs = [{"role": "user", "content": ["line one", "line two"]}]
        result = _format_messages_for_prompt(msgs)
        assert "line one\nline two" in result

    def test_list_content_non_text_blocks_skipped(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "visible"},
                    {"type": "tool_use", "name": "bash", "input": {}},
                ],
            }
        ]
        result = _format_messages_for_prompt(msgs)
        assert "visible" in result
        assert "bash" not in result

    def test_list_content_mixed_dicts_and_strings(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "from dict"},
                    "plain string",
                ],
            }
        ]
        result = _format_messages_for_prompt(msgs)
        assert "from dict\nplain string" in result

    def test_empty_messages_list(self):
        assert _format_messages_for_prompt([]) == ""

    def test_missing_role_defaults_to_unknown(self):
        msgs = [{"content": "text"}]
        result = _format_messages_for_prompt(msgs)
        assert "(unknown)" in result

    def test_missing_content_defaults_to_empty_string(self):
        msgs = [{"role": "user"}]
        result = _format_messages_for_prompt(msgs)
        assert result == "--- Message 1 (user) ---\n"

    def test_empty_string_content(self):
        msgs = [{"role": "user", "content": ""}]
        result = _format_messages_for_prompt(msgs)
        assert result == "--- Message 1 (user) ---\n"

    def test_empty_list_content(self):
        msgs = [{"role": "user", "content": []}]
        result = _format_messages_for_prompt(msgs)
        assert result == "--- Message 1 (user) ---\n"


# ── Integration tests: check_goal_achieved ───────────────────────────


class TestCheckGoalAchieved:
    """Pre-seed Redis cache so complete_tool returns without hitting the LLM."""

    @pytest.fixture
    def sample_messages(self):
        return [
            {"role": "user", "content": "Write a quicksort function in Python"},
            {"role": "assistant", "content": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    ..."},
        ]

    def _build_llm_messages(self, messages: list[dict], goal: str) -> list[dict]:
        """Reproduce the exact messages that check_goal_achieved sends to complete_tool."""
        transcript = _format_messages_for_prompt(messages)
        return [
            {"role": "user", "content": f"## Goal\n{goal}\n\n## Conversation\n{transcript}"}
        ]

    @pytest.mark.redis
    async def test_achieved_true(self, redis_client, sample_messages):
        r, seeded_keys = redis_client
        goal = "Write a quicksort function"
        response = {"achieved": True, "reasoning": "A quicksort implementation was provided."}

        await preseed_complete_tool(
            r, seeded_keys,
            messages=self._build_llm_messages(sample_messages, goal),
            tool=GOAL_RESULT_TOOL,
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            response=response,
        )

        result = await check_goal_achieved(sample_messages, goal)

        assert isinstance(result, GoalResult)
        assert result.achieved is True
        assert result.reasoning == "A quicksort implementation was provided."

    @pytest.mark.redis
    async def test_achieved_false(self, redis_client, sample_messages):
        r, seeded_keys = redis_client
        goal = "Deploy the application to production"
        response = {"achieved": False, "reasoning": "The conversation only discusses sorting, not deployment."}

        await preseed_complete_tool(
            r, seeded_keys,
            messages=self._build_llm_messages(sample_messages, goal),
            tool=GOAL_RESULT_TOOL,
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            response=response,
        )

        result = await check_goal_achieved(sample_messages, goal)

        assert isinstance(result, GoalResult)
        assert result.achieved is False
        assert "sorting" in result.reasoning
