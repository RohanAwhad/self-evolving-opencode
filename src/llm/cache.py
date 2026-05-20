"""Redis-backed cache for LLM call results."""

import hashlib
import json
import os

import redis
from loguru import logger

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6380"))


def get_redis(
    host: str = REDIS_HOST,
    port: int = REDIS_PORT,
) -> redis.Redis:
    """Return a Redis client."""
    return redis.Redis(host=host, port=port, decode_responses=True)


def cache_key(prefix: str, **kwargs: object) -> str:
    """Build a deterministic cache key from arbitrary kwargs."""
    blob = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    return f"{prefix}:{digest}"


def cache_get(r: redis.Redis, key: str) -> str | None:
    """Fetch a cached value (None on miss). Returns None if Redis is down."""
    try:
        return r.get(key)
    except redis.RedisError as e:
        logger.error("Redis cache_get failed (key={}): {}", key[:40], e)
        return None


def cache_set(r: redis.Redis, key: str, value: str, ttl: int | None = None) -> None:
    """Store a value in the cache. No-op if Redis is down."""
    try:
        r.set(key, value, ex=ttl)
    except redis.RedisError as e:
        logger.error("Redis cache_set failed (key={}): {}", key[:40], e)
