"""Tests for src/llm/__init__.py — pure functions + mocked async calls."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import AsyncAnthropicVertex

from src.llm import (
    _call_complete,
    _call_complete_tool,
    complete,
    complete_tool,
    get_client,
    strip_fences,
)
from src.llm.cache import cache_key


# ── fixtures ──────────────────────────────────────────────


@pytest.fixture
def mock_client():
    """Mock AsyncAnthropicVertex with async messages.create."""
    client = MagicMock(spec=AsyncAnthropicVertex)
    client.messages = MagicMock()
    client.messages.create = AsyncMock()
    return client


@pytest.fixture
def text_response():
    """Mock LLM response containing a single text block."""
    block = MagicMock()
    block.type = "text"
    block.text = "Hello from LLM"
    response = MagicMock()
    response.content = [block]
    return response


@pytest.fixture
def tool_response():
    """Mock LLM response containing a tool_use block (preceded by text)."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "thinking..."
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"goals": ["goal1", "goal2"]}
    response = MagicMock()
    response.content = [text_block, tool_block]
    return response


@pytest.fixture
def sample_messages():
    return [{"role": "user", "content": "hello"}]


@pytest.fixture
def sample_tool():
    return {
        "name": "extract_goals",
        "description": "Extract goals",
        "input_schema": {"type": "object", "properties": {"goals": {"type": "array"}}},
    }


# ── strip_fences ──────────────────────────────────────────────


def test_strip_fences_with_lang_tag():
    text = "```python\nprint('hello')\n```"
    assert strip_fences(text) == "print('hello')"


def test_strip_fences_no_lang_tag():
    text = "```\nsome code\n```"
    assert strip_fences(text) == "some code"


def test_strip_fences_no_fences():
    text = "just plain text"
    assert strip_fences(text) == "just plain text"


def test_strip_fences_only_opening_fence():
    text = "```python\ncode without closing"
    assert strip_fences(text) == "code without closing"


def test_strip_fences_nested_fences():
    text = "```markdown\nhere is a block:\n```python\nx = 1\n```\ndone\n```"
    result = strip_fences(text)
    # outer fences stripped; inner content preserved up to last ```
    assert "here is a block:" in result
    assert result.startswith("here is a block:")


# ── get_client ────────────────────────────────────────────────


def test_get_client_returns_async_vertex():
    client = get_client()
    assert isinstance(client, AsyncAnthropicVertex)


def test_get_client_custom_params():
    client = get_client(project_id="my-project", region="us-east5")
    assert isinstance(client, AsyncAnthropicVertex)


# ── cache_key determinism ────────────────────────────────────


def test_cache_key_same_inputs_same_key():
    msgs = [{"role": "user", "content": "hello"}]
    k1 = cache_key("complete", messages=msgs, model="m", max_tokens=100, system=None)
    k2 = cache_key("complete", messages=msgs, model="m", max_tokens=100, system=None)
    assert k1 == k2


def test_cache_key_different_inputs_different_key():
    msgs = [{"role": "user", "content": "hello"}]
    k1 = cache_key("complete", messages=msgs, model="m", max_tokens=100, system=None)
    k2 = cache_key("complete", messages=msgs, model="m", max_tokens=200, system=None)
    assert k1 != k2


def test_cache_key_complete_vs_complete_tool_different_prefix():
    msgs = [{"role": "user", "content": "hello"}]
    k1 = cache_key("complete", messages=msgs, model="m", max_tokens=100, system=None)
    k2 = cache_key("complete_tool", messages=msgs, model="m", max_tokens=100, system=None)
    assert k1 != k2


def test_cache_key_complete_tool_with_tool_dict():
    msgs = [{"role": "user", "content": "extract"}]
    tool = {"name": "extract_goals", "description": "...", "input_schema": {"type": "object"}}
    k1 = cache_key("complete_tool", messages=msgs, tool=tool, model="m", max_tokens=100, system=None)
    k2 = cache_key("complete_tool", messages=msgs, tool=tool, model="m", max_tokens=100, system=None)
    assert k1 == k2


def test_cache_key_kwarg_order_irrelevant():
    k1 = cache_key("complete", model="m", max_tokens=100)
    k2 = cache_key("complete", max_tokens=100, model="m")
    assert k1 == k2


# ── _call_complete / _call_complete_tool ──────────────────


async def test_call_complete_extracts_text(mock_client, text_response):
    mock_client.messages.create.return_value = text_response
    result = await _call_complete(mock_client, {"model": "m", "messages": []})
    assert result == "Hello from LLM"
    mock_client.messages.create.assert_awaited_once_with(model="m", messages=[])


async def test_call_complete_tool_extracts_tool_input(mock_client, tool_response):
    mock_client.messages.create.return_value = tool_response
    result = await _call_complete_tool(mock_client, {"model": "m", "messages": []})
    assert result == {"goals": ["goal1", "goal2"]}


# ── complete ──────────────────────────────────────────────


async def test_complete_returns_text(sample_messages):
    with patch("src.llm._call_complete", new_callable=AsyncMock, return_value="LLM says hi"):
        result = await complete(sample_messages, use_cache=False)
    assert result == "LLM says hi"


async def test_complete_passes_system_kwarg(sample_messages):
    with patch("src.llm._call_complete", new_callable=AsyncMock, return_value="ok") as mock:
        await complete(sample_messages, system="Be helpful", use_cache=False)
    kwargs_passed = mock.call_args[0][1]
    assert kwargs_passed["system"] == "Be helpful"


async def test_complete_no_system_kwarg_by_default(sample_messages):
    with patch("src.llm._call_complete", new_callable=AsyncMock, return_value="ok") as mock:
        await complete(sample_messages, use_cache=False)
    kwargs_passed = mock.call_args[0][1]
    assert "system" not in kwargs_passed


async def test_complete_cache_hit_skips_api(sample_messages):
    with (
        patch("src.llm.cache_get", new_callable=AsyncMock, return_value="cached"),
        patch("src.llm._call_complete", new_callable=AsyncMock) as mock_call,
    ):
        result = await complete(sample_messages, use_cache=True)
    assert result == "cached"
    mock_call.assert_not_awaited()


async def test_complete_cache_miss_calls_api_and_stores(sample_messages):
    with (
        patch("src.llm.cache_get", new_callable=AsyncMock, return_value=None),
        patch("src.llm.cache_set", new_callable=AsyncMock) as mock_set,
        patch("src.llm._call_complete", new_callable=AsyncMock, return_value="fresh"),
    ):
        result = await complete(sample_messages, use_cache=True)
    assert result == "fresh"
    mock_set.assert_awaited_once()


# ── complete_tool ─────────────────────────────────────────


async def test_complete_tool_returns_dict(sample_messages, sample_tool):
    with patch("src.llm._call_complete_tool", new_callable=AsyncMock, return_value={"goals": ["a"]}):
        result = await complete_tool(sample_messages, tool=sample_tool, use_cache=False)
    assert result == {"goals": ["a"]}


async def test_complete_tool_passes_tools_and_tool_choice(sample_messages, sample_tool):
    with patch("src.llm._call_complete_tool", new_callable=AsyncMock, return_value={}) as mock:
        await complete_tool(sample_messages, tool=sample_tool, use_cache=False)
    kwargs_passed = mock.call_args[0][1]
    assert kwargs_passed["tools"] == [sample_tool]
    assert kwargs_passed["tool_choice"] == {"type": "tool", "name": "extract_goals"}


async def test_complete_tool_passes_system_kwarg(sample_messages, sample_tool):
    with patch("src.llm._call_complete_tool", new_callable=AsyncMock, return_value={}) as mock:
        await complete_tool(sample_messages, tool=sample_tool, system="sys", use_cache=False)
    kwargs_passed = mock.call_args[0][1]
    assert kwargs_passed["system"] == "sys"


async def test_complete_tool_cache_hit_returns_parsed_json(sample_messages, sample_tool):
    cached = json.dumps({"goals": ["cached"]})
    with (
        patch("src.llm.cache_get", new_callable=AsyncMock, return_value=cached),
        patch("src.llm._call_complete_tool", new_callable=AsyncMock) as mock_call,
    ):
        result = await complete_tool(sample_messages, tool=sample_tool, use_cache=True)
    assert result == {"goals": ["cached"]}
    mock_call.assert_not_awaited()


async def test_complete_tool_cache_miss_stores_json_string(sample_messages, sample_tool):
    with (
        patch("src.llm.cache_get", new_callable=AsyncMock, return_value=None),
        patch("src.llm.cache_set", new_callable=AsyncMock) as mock_set,
        patch("src.llm._call_complete_tool", new_callable=AsyncMock, return_value={"x": 1}),
    ):
        result = await complete_tool(sample_messages, tool=sample_tool, use_cache=True)
    assert result == {"x": 1}
    stored_value = mock_set.call_args[0][2]
    assert json.loads(stored_value) == {"x": 1}
