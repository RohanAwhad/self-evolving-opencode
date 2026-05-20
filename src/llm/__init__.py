"""Thin abstraction over AnthropicVertex for LLM completions."""

import json
import os

from anthropic import AnthropicVertex
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, before_sleep_log

from .cache import cache_get, cache_key, cache_set, get_redis

DEFAULT_PROJECT = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "itpc-gcp-ai-eng-claude")
DEFAULT_REGION = os.environ.get("GOOGLE_VERTEX_LOCATION", "global")
DEFAULT_MODEL = "claude-opus-4-6@default"

_redis = get_redis()


_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=1, max=60),
    before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
    reraise=True,
)


def get_client(
    project_id: str = DEFAULT_PROJECT,
    region: str = DEFAULT_REGION,
) -> AnthropicVertex:
    """Create an AnthropicVertex client."""
    return AnthropicVertex(project_id=project_id, region=region)


@_retry
def _call_complete(client: AnthropicVertex, kwargs: dict) -> str:
    response = client.messages.create(**kwargs)
    return response.content[0].text


@_retry
def _call_complete_tool(client: AnthropicVertex, kwargs: dict) -> dict:
    response = client.messages.create(**kwargs)
    tool_block = next(b for b in response.content if b.type == "tool_use")
    return tool_block.input


def complete(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    system: str | None = None,
    project_id: str = DEFAULT_PROJECT,
    region: str = DEFAULT_REGION,
    use_cache: bool = True,
) -> str:
    """Send a completion request and return the raw text response."""
    if use_cache:
        key = cache_key(
            "complete",
            messages=messages, model=model, max_tokens=max_tokens, system=system,
        )
        hit = cache_get(_redis, key)
        if hit is not None:
            logger.debug("Cache hit for complete (key={})", key[:40])
            return hit
        logger.debug("Cache miss for complete (key={})", key[:40])

    client = get_client(project_id=project_id, region=region)
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system is not None:
        kwargs["system"] = system
    result = _call_complete(client, kwargs)

    if use_cache:
        cache_set(_redis, key, result)

    return result


def complete_tool(
    messages: list[dict],
    tool: dict,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    system: str | None = None,
    project_id: str = DEFAULT_PROJECT,
    region: str = DEFAULT_REGION,
    use_cache: bool = True,
) -> dict:
    """Send a completion forcing a specific tool call, return the parsed input dict.

    Args:
        tool: Tool definition dict with "name", "description", "input_schema".
    """
    if use_cache:
        key = cache_key(
            "complete_tool",
            messages=messages, tool=tool, model=model,
            max_tokens=max_tokens, system=system,
        )
        hit = cache_get(_redis, key)
        if hit is not None:
            logger.debug("Cache hit for complete_tool (key={})", key[:40])
            return json.loads(hit)
        logger.debug("Cache miss for complete_tool (key={})", key[:40])

    client = get_client(project_id=project_id, region=region)
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": tool["name"]},
    }
    if system is not None:
        kwargs["system"] = system
    result = _call_complete_tool(client, kwargs)

    if use_cache:
        cache_set(_redis, key, json.dumps(result, sort_keys=True))

    return result


def strip_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()
