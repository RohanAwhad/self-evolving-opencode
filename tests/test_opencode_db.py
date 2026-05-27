"""Tests for src/opencode_db.py -- SQLite access layer."""

from src.opencode_db import (
    get_conversation_transcript,
    get_messages_for_session,
    get_sessions,
    get_skills_for_session,
    parse_message_range,
    slice_messages,
)


# -- parse_message_range (pure) -----------------------------------------------


class TestParseMessageRange:
    def test_range_string(self):
        assert parse_message_range("msgs 1-8") == (0, 8)

    def test_same_endpoints(self):
        assert parse_message_range("msgs 3-3") == (2, 3)

    def test_single_number(self):
        assert parse_message_range("5") == (4, 5)

    def test_garbage_input(self):
        assert parse_message_range("foo") == (0, 999999)

    def test_empty_string(self):
        assert parse_message_range("") == (0, 999999)


# -- slice_messages (pure) ----------------------------------------------------


class TestSliceMessages:
    _MSGS = [{"role": "user", "content": f"msg{i}"} for i in range(10)]

    def test_basic_slice(self):
        result = slice_messages(self._MSGS, "msgs 2-5")
        assert len(result) == 4
        assert result[0]["content"] == "msg1"
        assert result[-1]["content"] == "msg4"

    def test_empty_input(self):
        assert slice_messages([], "msgs 1-5") == []

    def test_range_exceeds_length(self):
        result = slice_messages(self._MSGS[:3], "msgs 1-100")
        assert len(result) == 3

    def test_single_message_slice(self):
        result = slice_messages(self._MSGS, "3")
        assert len(result) == 1
        assert result[0]["content"] == "msg2"


# -- get_sessions (async, needs DB) -------------------------------------------


class TestGetSessions:
    async def test_ordered_by_time_updated_desc(self, db_path):
        sessions = await get_sessions(db_path=db_path)
        times = [s.time_updated for s in sessions]
        assert times == sorted(times, reverse=True)

    async def test_limit(self, db_path):
        sessions = await get_sessions(limit=2, db_path=db_path)
        assert len(sessions) == 2
        assert sessions[0].id == "s1"
        assert sessions[1].id == "s2"

    async def test_returns_all(self, db_path):
        sessions = await get_sessions(db_path=db_path)
        assert len(sessions) == 5

    async def test_zero_message_count(self, db_path):
        sessions = await get_sessions(db_path=db_path)
        s4 = next(s for s in sessions if s.id == "s4")
        assert s4.message_count == 0

    async def test_valid_model_json(self, db_path):
        sessions = await get_sessions(db_path=db_path)
        s1 = next(s for s in sessions if s.id == "s1")
        assert s1.model_id == "claude-3"

    async def test_null_model_returns_unknown(self, db_path):
        sessions = await get_sessions(db_path=db_path)
        s3 = next(s for s in sessions if s.id == "s3")
        assert s3.model_id == "unknown"

    # TODO: malformed model JSON -> "unknown" (requires try/except in _get_sessions_sync)
    # Currently json.loads raises JSONDecodeError on bad model strings.

    async def test_nullable_cost_and_tokens(self, db_path):
        sessions = await get_sessions(db_path=db_path)
        s3 = next(s for s in sessions if s.id == "s3")
        assert s3.cost == 0.0
        assert s3.tokens_input == 0
        assert s3.tokens_output == 0

    async def test_null_title_becomes_untitled(self, db_path):
        sessions = await get_sessions(db_path=db_path)
        s3 = next(s for s in sessions if s.id == "s3")
        assert s3.title == "(untitled)"


# -- get_messages_for_session (async, needs DB) --------------------------------


class TestGetMessagesForSession:
    async def test_ordering_by_time_created(self, db_path):
        msgs = await get_messages_for_session("s1", db_path=db_path)
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "user", "assistant", "assistant"]

    async def test_text_extracted(self, db_path):
        msgs = await get_messages_for_session("s1", db_path=db_path)
        assert msgs[0]["content"] == "Hello"

    async def test_multi_part_concatenated(self, db_path):
        msgs = await get_messages_for_session("s1", db_path=db_path)
        assert msgs[1]["content"] == "Hi there\nHow can I help?"

    async def test_tool_part_formatted(self, db_path):
        msgs = await get_messages_for_session("s1", db_path=db_path)
        assert msgs[3]["content"] == "[tool: bash]\nDone fixing"

    async def test_empty_session_returns_empty(self, db_path):
        msgs = await get_messages_for_session("s4", db_path=db_path)
        assert msgs == []

    async def test_nonexistent_session_returns_empty(self, db_path):
        msgs = await get_messages_for_session("nonexistent", db_path=db_path)
        assert msgs == []

    async def test_message_with_no_parts_skipped(self, db_path):
        msgs = await get_messages_for_session("s5", db_path=db_path)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "I'll help"


# -- get_conversation_transcript (async, needs DB) ----------------------------


class TestGetConversationTranscript:
    async def test_header_format(self, db_path):
        t = await get_conversation_transcript("s2", db_path=db_path)
        assert "--- Message 1 (user) ---" in t
        assert "--- Message 2 (assistant) ---" in t
        assert "--- Message 3 (user) ---" in t

    async def test_content_present(self, db_path):
        t = await get_conversation_transcript("s2", db_path=db_path)
        assert "Review this code" in t
        assert "LGTM" in t
        assert "Thanks" in t

    async def test_multi_part_concat(self, db_path):
        t = await get_conversation_transcript("s1", db_path=db_path)
        assert "Hi there\nHow can I help?" in t

    async def test_tool_in_transcript(self, db_path):
        t = await get_conversation_transcript("s1", db_path=db_path)
        assert "[tool: bash]" in t

    async def test_empty_session_returns_empty_string(self, db_path):
        t = await get_conversation_transcript("s4", db_path=db_path)
        assert t == ""

    async def test_numbering_counts_all_messages(self, db_path):
        t = await get_conversation_transcript("s5", db_path=db_path)
        # m10 is enumerated as 1, m11 as 2 but skipped (no parts)
        assert "--- Message 1 (assistant) ---" in t
        assert "Message 2" not in t


# -- get_skills_for_session (async, needs DB) ----------------------------------


class TestGetSkillsForSession:
    async def test_returns_skill_names(self, db_path):
        skills = await get_skills_for_session("s1", db_path=db_path)
        assert set(skills) == {"gitlab-api", "code-review"}

    async def test_single_skill_session(self, db_path):
        skills = await get_skills_for_session("s2", db_path=db_path)
        assert skills == ["code-review"]

    async def test_no_skills_returns_empty(self, db_path):
        skills = await get_skills_for_session("s3", db_path=db_path)
        assert skills == []

    async def test_nonexistent_session_returns_empty(self, db_path):
        skills = await get_skills_for_session("nonexistent", db_path=db_path)
        assert skills == []

    async def test_empty_session_returns_empty(self, db_path):
        skills = await get_skills_for_session("s4", db_path=db_path)
        assert skills == []
