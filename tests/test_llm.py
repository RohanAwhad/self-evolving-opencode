"""Tests for src/llm/__init__.py — pure functions, no LLM calls needed."""

from anthropic import AsyncAnthropicVertex

from src.llm import get_client, strip_fences
from src.llm.cache import cache_key


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
