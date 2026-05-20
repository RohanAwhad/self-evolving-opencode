"""Redis-backed cache for LLM call results."""

import hashlib
import json
import os

import redis.asyncio as aioredis
from loguru import logger

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6380"))


def get_redis(
    host: str = REDIS_HOST,
    port: int = REDIS_PORT,
) -> aioredis.Redis:
    """Return an async Redis client."""
    return aioredis.Redis(host=host, port=port, decode_responses=True)


def cache_key(prefix: str, **kwargs: object) -> str:
    """Build a deterministic cache key from arbitrary kwargs."""
    blob = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    return f"{prefix}:{digest}"


async def cache_get(r: aioredis.Redis, key: str) -> str | None:
    """Fetch a cached value (None on miss). Returns None if Redis is down."""
    try:
        return await r.get(key)
    except aioredis.RedisError as e:
        logger.error("Redis cache_get failed (key={}): {}", key[:40], e)
        return None


async def cache_set(r: aioredis.Redis, key: str, value: str, ttl: int | None = None) -> None:
    """Store a value in the cache. No-op if Redis is down."""
    try:
        await r.set(key, value, ex=ttl)
    except aioredis.RedisError as e:
        logger.error("Redis cache_set failed (key={}): {}", key[:40], e)
