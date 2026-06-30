"""Async Redis client for the Recommendation Service.

Thin lazy singleton over redis.asyncio, sharing the same Redis instance as the
Hono backend (via REDIS_URL). Used by the event-stream consumer. No keys are
read or written at import time, and the FastAPI app does not import this module
— so the web process needs no Redis connection to serve health checks.
"""

from __future__ import annotations

import redis.asyncio as aioredis

from app.config import get_settings

settings = get_settings()

_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Return a lazily-created shared async Redis client."""
    global _client
    if _client is None:
        _client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _client


async def close_redis() -> None:
    """Close the shared client (call on shutdown)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
