"""Tests for src/conversation_summarizer."""

import pytest

from src.conversation_summarizer import (
    SUMMARIZE_TOOL,
    SYSTEM_PROMPT,
    _build_markdown,
    _format_rich_messages,
    summarize_conversation,
)
from src.llm import DEFAULT_MODEL
from tests.helpers import preseed_complete_tool


# ---------------------------------------------------------------------------
# _format_rich_messages
# ---------------------------------------------------------------------------


class TestFormatRichMessages:
    def test_single_text_message(self):
        msgs = [{"role": "user", "parts": [{"type": "text", "text": "Hello"}]}]
        result = _format_rich_messages(msgs)
        assert "--- Message 1 (user) ---" in result
        assert "Hello" in result

    def test_multiple_messages(self):
        msgs = [
            {"role": "user", "parts": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "Hey"}]},
        ]
        result = _format_rich_messages(msgs)
        assert "--- Message 1 (user) ---" in result
        assert "--- Message 2 (assistant) ---" in result
        assert "Hi" in result
        assert "Hey" in result

    def test_tool_part_full_fields(self):
        msgs = [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool",
                        "tool": "bash",
                        "title": "Run tests",
                        "status": "success",
                        "input": {"command": "pytest"},
                        "output": "3 passed",
                    }
                ],
            }
        ]
        result = _format_rich_messages(msgs)
        assert "[tool: bash]" in result
        assert "Run tests" in result
        assert "(success)" in result
        assert "command: pytest" in result
        assert "Output: 3 passed" in result

    def test_tool_part_minimal_fields(self):
        msgs = [
            {
                "role": "assistant",
                "parts": [{"type": "tool", "tool": "read"}],
            }
        ]
        result = _format_rich_messages(msgs)
        assert "[tool: read]" in result
        # No title/status/input/output — should not crash
        assert "()" not in result  # empty status shouldn't produce parens

    def test_reasoning_part(self):
        msgs = [
            {
                "role": "assistant",
                "parts": [{"type": "reasoning", "text": "Let me think..."}],
            }
        ]
        result = _format_rich_messages(msgs)
        assert "[reasoning] Let me think..." in result

    def test_mixed_parts(self):
        msgs = [
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "text": "I'll run a command"},
                    {"type": "tool", "tool": "bash", "input": {"cmd": "ls"}, "output": "file.py"},
                    {"type": "reasoning", "text": "Looks good"},
                ],
            }
        ]
        result = _format_rich_messages(msgs)
        assert "I'll run a command" in result
        assert "[tool: bash]" in result
        assert "[reasoning] Looks good" in result

    def test_empty_parts_list(self):
        msgs = [{"role": "user", "parts": []}]
        result = _format_rich_messages(msgs)
        assert "--- Message 1 (user) ---" in result

    def test_no_parts_key(self):
        msgs = [{"role": "user"}]
        result = _format_rich_messages(msgs)
        assert "--- Message 1 (user) ---" in result

    def test_missing_role_defaults_to_unknown(self):
        msgs = [{"parts": [{"type": "text", "text": "orphan"}]}]
        result = _format_rich_messages(msgs)
        assert "(unknown)" in result

    def test_tool_input_value_truncation(self):
        long_value = "x" * 300
        msgs = [
            {
                "role": "assistant",
                "parts": [
                    {"type": "tool", "tool": "write", "input": {"content": long_value}},
                ],
            }
        ]
        result = _format_rich_messages(msgs)
        # Value should be truncated to 200 chars + "..."
        assert "x" * 200 + "..." in result
        assert "x" * 201 not in result.replace("...", "")

    def test_tool_output_truncation(self):
        long_output = "y" * 600
        msgs = [
            {
                "role": "assistant",
                "parts": [
                    {"type": "tool", "tool": "bash", "output": long_output},
                ],
            }
        ]
        result = _format_rich_messages(msgs)
        assert "y" * 500 + "..." in result
        assert "y" * 501 not in result.replace("...", "")


# ---------------------------------------------------------------------------
# _build_markdown
# ---------------------------------------------------------------------------


class TestBuildMarkdown:
    def test_all_sections(self):
        data = {
            "goal": "Fix login bug",
            "intent": "Users can't log in",
            "what_happened": "Traced through auth flow",
            "user_messages": ["Fix the login", "Try the staging server"],
            "assistant_actions": ["Read logs", "Fixed the handler"],
            "tool_usage": ["bash: ran pytest — 3 passed"],
            "outcome": "Fixed successfully",
            "evaluation_criteria": ["Login works", "No regressions"],
        }
        result = _build_markdown(data)
        assert "## Goal\nFix login bug" in result
        assert "## Intent\nUsers can't log in" in result
        assert "## What Happened\nTraced through auth flow" in result
        assert "1. Fix the login" in result
        assert "2. Try the staging server" in result
        assert "1. Read logs" in result
        assert "## Outcome\nFixed successfully" in result
        assert "- Login works" in result

    def test_empty_lists(self):
        data = {
            "goal": "Test",
            "intent": "Test intent",
            "what_happened": "Nothing",
            "user_messages": [],
            "assistant_actions": [],
            "tool_usage": [],
            "outcome": "N/A",
            "evaluation_criteria": [],
        }
        result = _build_markdown(data)
        assert "## Goal\nTest" in result
        assert "## User Messages" not in result
        assert "## Tool Usage" not in result


# ---------------------------------------------------------------------------
# summarize_conversation (integration -- requires Redis)
# ---------------------------------------------------------------------------


class TestSummarizeConversation:
    @pytest.mark.redis
    async def test_returns_structured_summary(self, redis_client):
        r, seeded_keys = redis_client

        messages = [
            {"role": "user", "parts": [{"type": "text", "text": "Fix the login bug"}]},
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "text": "I'll look into it"},
                    {"type": "tool", "tool": "bash", "input": {"cmd": "grep -r login"}, "output": "found 3 matches"},
                ],
            },
        ]

        # Build the exact prompt that summarize_conversation will produce
        transcript = _format_rich_messages(messages)
        llm_messages = [{"role": "user", "content": f"Summarize this conversation:\n\n{transcript}"}]

        fake_response = {
            "goal": "Fix login bug",
            "intent": "Users can't log in",
            "what_happened": "Traced through auth flow",
            "user_messages": ["Fix the login bug"],
            "assistant_actions": ["Searched for login code"],
            "tool_usage": ["bash: grep -r login — found 3 matches"],
            "outcome": "Fixed successfully",
            "evaluation_criteria": ["Login works"],
        }

        await preseed_complete_tool(
            r,
            seeded_keys,
            messages=llm_messages,
            tool=SUMMARIZE_TOOL,
            model=DEFAULT_MODEL,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            response=fake_response,
        )

        result = await summarize_conversation(messages)
        assert "## Goal\nFix login bug" in result
        assert "## Outcome\nFixed successfully" in result

    @pytest.mark.redis
    async def test_custom_model_and_max_tokens(self, redis_client):
        r, seeded_keys = redis_client

        messages = [
            {"role": "user", "parts": [{"type": "text", "text": "Hello"}]},
        ]

        transcript = _format_rich_messages(messages)
        llm_messages = [{"role": "user", "content": f"Summarize this conversation:\n\n{transcript}"}]

        custom_model = "claude-sonnet-4-20250514"
        custom_max_tokens = 2048

        fake_response = {
            "goal": "Greeting",
            "intent": "Say hello",
            "what_happened": "User said hello",
            "user_messages": ["Hello"],
            "assistant_actions": [],
            "tool_usage": [],
            "outcome": "Greeted",
            "evaluation_criteria": [],
        }

        await preseed_complete_tool(
            r,
            seeded_keys,
            messages=llm_messages,
            tool=SUMMARIZE_TOOL,
            model=custom_model,
            max_tokens=custom_max_tokens,
            system=SYSTEM_PROMPT,
            response=fake_response,
        )

        result = await summarize_conversation(
            messages, model=custom_model, max_tokens=custom_max_tokens
        )
        assert "## Goal\nGreeting" in result
