"""Reel feature computation (compute-on-miss source for the feature store).

Derives engagement counters plus freshness and velocity from a reel document.
CODE ONLY — not activated anywhere. No candidate generation, ranking, or ML.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from app.config import get_settings
from app.core.mongo import get_collection
from app.features import store

settings = get_settings()

# Freshness decays linearly to zero over this horizon.
_FRESHNESS_HORIZON_HOURS = 24.0 * 7.0


def _safe_object_id(value: str) -> ObjectId | str:
    try:
        return ObjectId(value)
    except Exception:
        return value


def _age_hours(created_at: Any) -> float:
    if not isinstance(created_at, datetime):
        return 0.0
    created = created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    delta = datetime.now(tz=timezone.utc) - created
    return max(0.0, delta.total_seconds() / 3600.0)


def _empty() -> dict[str, Any]:
    return {
        "viewCount": 0,
        "likeCount": 0,
        "shareCount": 0,
        "commentCount": 0,
        "freshness": 0.0,
        "velocity": 0.0,
    }


async def compute_reel_features(reel_id: str) -> dict[str, Any]:
    """Derive the stored feature set from the reel document."""
    reels_col = get_collection(settings.reels_collection)
    reel = await reels_col.find_one({"_id": _safe_object_id(reel_id)})
    if not reel:
        return _empty()

    view_count = reel.get("viewCount", 0) or 0
    like_count = reel.get("likeCount", 0) or 0
    share_count = reel.get("shareCount", 0) or 0
    comment_count = reel.get("commentCount", 0) or 0

    age_hours = _age_hours(reel.get("created_at"))
    freshness = max(0.0, 1.0 - (age_hours / _FRESHNESS_HORIZON_HOURS))
    engagement = view_count + like_count + share_count + comment_count
    velocity = engagement / max(1.0, age_hours)

    return {
        "viewCount": view_count,
        "likeCount": like_count,
        "shareCount": share_count,
        "commentCount": comment_count,
        "freshness": freshness,
        "velocity": velocity,
    }


async def get_reel_features(reel_id: str) -> dict[str, Any]:
    """Compute-on-miss accessor for a reel's features."""
    return await store.get_or_compute(
        store.reel_feature_key(reel_id),
        settings.feat_reel_ttl,
        lambda: compute_reel_features(reel_id),
    )
