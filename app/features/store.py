"""Redis feature store: read/write abstractions, TTL strategy, compute-on-miss,
and invalidation helpers for user and reel feature vectors.

CODE ONLY (Batch 6). Nothing here is activated: not called from any API route,
not started by any background job, not wired into the gateway.

Layout: each feature vector is a Redis hash at `rec:feat:user:{id}` /
`rec:feat:reel:{id}`. Values are JSON-encoded per field so scalar features and
nested affinity maps coexist. A TTL is applied on write.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from app.config import get_settings
from app.core.redis import get_redis

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.features")


def user_feature_key(user_id: str) -> str:
    return f"{settings.feat_user_prefix}{user_id}"


def reel_feature_key(reel_id: str) -> str:
    return f"{settings.feat_reel_prefix}{reel_id}"


def _encode(features: dict[str, Any]) -> dict[str, str]:
    return {k: json.dumps(v) for k, v in features.items()}


def _decode(raw: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            out[key] = json.loads(value)
        except (ValueError, TypeError):
            out[key] = value
    return out


async def read_features(key: str) -> dict[str, Any] | None:
    """Return the cached feature hash, or None on a cache miss."""
    raw = await get_redis().hgetall(key)
    if not raw:
        return None
    return _decode(raw)


async def write_features(key: str, features: dict[str, Any], ttl: int) -> None:
    """Replace a feature hash and (re)apply its TTL atomically."""
    client = get_redis()
    pipe = client.pipeline()
    pipe.delete(key)
    if features:
        pipe.hset(key, mapping=_encode(features))
        pipe.expire(key, ttl)
    await pipe.execute()


async def get_or_compute(
    key: str,
    ttl: int,
    compute: Callable[[], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Compute-on-miss: return cached features, else compute → cache → return."""
    cached = await read_features(key)
    if cached is not None:
        return cached
    features = await compute()
    await write_features(key, features, ttl)
    return features


async def invalidate(key: str) -> None:
    """Drop a single feature entry."""
    await get_redis().delete(key)


async def invalidate_user(user_id: str) -> None:
    await invalidate(user_feature_key(user_id))


async def invalidate_reel(reel_id: str) -> None:
    await invalidate(reel_feature_key(reel_id))
