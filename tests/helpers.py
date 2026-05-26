"""Shared test helpers (cache pre-seeding, etc.)."""

import json

from src.llm.cache import cache_key, cache_set


async def preseed_complete_tool(
    r,
    seeded_keys: list[str],
    *,
    messages: list[dict],
    tool: dict,
    model: str,
    max_tokens: int,
    system: str | None,
    response: dict,
) -> str:
    """Pre-seed Redis cache for a complete_tool call. Returns the cache key."""
    key = cache_key(
        "complete_tool",
        messages=messages, tool=tool, model=model,
        max_tokens=max_tokens, system=system,
    )
    await cache_set(r, key, json.dumps(response, sort_keys=True))
    seeded_keys.append(key)
    return key


async def preseed_complete(
    r,
    seeded_keys: list[str],
    *,
    messages: list[dict],
    model: str,
    max_tokens: int,
    system: str | None,
    response: str,
) -> str:
    """Pre-seed Redis cache for a complete call. Returns the cache key."""
    key = cache_key(
        "complete",
        messages=messages, model=model,
        max_tokens=max_tokens, system=system,
    )
    await cache_set(r, key, response)
    seeded_keys.append(key)
    return key
