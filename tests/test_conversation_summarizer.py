"""Tests for src/conversation_summarizer."""

import pytest

from src.conversation_summarizer import (
    SUMMARY_END_TAG,
    SUMMARY_START_TAG,
    SYSTEM_PROMPT,
    _extract_summary,
    _format_rich_messages,
    summarize_conversation,
)
from src.llm import DEFAULT_MODEL
from tests.helpers import preseed_complete


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
# _extract_summary
# ---------------------------------------------------------------------------


class TestExtractSummary:
    def test_extracts_between_tags(self):
        response = f"preamble\n{SUMMARY_START_TAG}\n## Goal\nDo stuff\n{SUMMARY_END_TAG}\npostamble"
        result = _extract_summary(response)
        assert result == "## Goal\nDo stuff"

    def test_strips_whitespace(self):
        response = f"{SUMMARY_START_TAG}\n  content  \n{SUMMARY_END_TAG}"
        result = _extract_summary(response)
        assert result == "content"

    def test_missing_tags_returns_raw(self):
        response = "Just some text without tags"
        result = _extract_summary(response)
        assert result == "Just some text without tags"

    def test_empty_between_tags(self):
        response = f"{SUMMARY_START_TAG}\n\n{SUMMARY_END_TAG}"
        result = _extract_summary(response)
        assert result == ""

    def test_only_start_tag_returns_raw(self):
        response = f"{SUMMARY_START_TAG}\ncontent without end tag"
        result = _extract_summary(response)
        # No end tag → fallback to raw
        assert SUMMARY_START_TAG in result

    def test_multiline_content(self):
        content = "## Goal\nFix bug\n\n## Outcome\nFixed"
        response = f"{SUMMARY_START_TAG}\n{content}\n{SUMMARY_END_TAG}"
        result = _extract_summary(response)
        assert result == content


# ---------------------------------------------------------------------------
# summarize_conversation (integration -- requires Redis)
# ---------------------------------------------------------------------------


class TestSummarizeConversation:
    @pytest.mark.redis
    async def test_returns_extracted_summary(self, redis_client):
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

        fake_summary = "## Goal\nFix login bug\n\n## Outcome\nFixed successfully"
        fake_response = f"{SUMMARY_START_TAG}\n{fake_summary}\n{SUMMARY_END_TAG}"

        await preseed_complete(
            r,
            seeded_keys,
            messages=llm_messages,
            model=DEFAULT_MODEL,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            response=fake_response,
        )

        result = await summarize_conversation(messages)
        assert result == fake_summary

    @pytest.mark.redis
    async def test_custom_model_and_max_tokens(self, redis_client):
        r, seeded_keys = redis_client

        messages = [
            {"role": "user", "parts": [{"type": "text", "text": "Hello"}]},
        ]

        transcript = _format_rich_messages(messages)
        llm_messages = [{"role": "user", "content": f"Summarize this conversation:\n\n{transcript}"}]

        fake_response = f"{SUMMARY_START_TAG}\nshort summary\n{SUMMARY_END_TAG}"

        custom_model = "claude-sonnet-4-20250514"
        custom_max_tokens = 2048

        await preseed_complete(
            r,
            seeded_keys,
            messages=llm_messages,
            model=custom_model,
            max_tokens=custom_max_tokens,
            system=SYSTEM_PROMPT,
            response=fake_response,
        )

        result = await summarize_conversation(
            messages, model=custom_model, max_tokens=custom_max_tokens
        )
        assert result == "short summary"
