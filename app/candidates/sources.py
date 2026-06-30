"""Recall sources for candidate generation.

Each function runs one Mongo query against the reels collection and returns a
list of reel-ID strings. Exclusion of deleted reels, already-seen reels, and
muted creators is applied at query time. There is no scoring or ordering beyond
the per-source recency/popularity sort. CODE ONLY — not activated anywhere.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection


async def _reel_ids(cursor: Any) -> list[str]:
    docs = await cursor.to_list(length=None)
    return [str(d["_id"]) for d in docs]


def _base_filter(exclude_ids: list[ObjectId]) -> dict[str, Any]:
    query: dict[str, Any] = {"isDeleted": False}
    if exclude_ids:
        query["_id"] = {"$nin": exclude_ids}
    return query


async def from_tag_affinity(
    reels: AsyncIOMotorCollection,
    tags: list[str],
    exclude_ids: list[ObjectId],
    muted_creator_ids: list[ObjectId],
    limit: int,
) -> list[str]:
    if not tags or limit <= 0:
        return []
    query = _base_filter(exclude_ids)
    query["tags"] = {"$in": tags}
    if muted_creator_ids:
        query["creator_id"] = {"$nin": muted_creator_ids}
    cursor = reels.find(query, {"_id": 1}).sort("created_at", -1).limit(limit)
    return await _reel_ids(cursor)


async def from_creator_affinity(
    reels: AsyncIOMotorCollection,
    creator_ids: list[ObjectId],
    exclude_ids: list[ObjectId],
    limit: int,
) -> list[str]:
    # creator_ids are pre-filtered by the caller to exclude muted creators.
    if not creator_ids or limit <= 0:
        return []
    query = _base_filter(exclude_ids)
    query["creator_id"] = {"$in": creator_ids}
    cursor = reels.find(query, {"_id": 1}).sort("created_at", -1).limit(limit)
    return await _reel_ids(cursor)


async def from_preferred_audio(
    reels: AsyncIOMotorCollection,
    audio_ids: list[str],
    exclude_ids: list[ObjectId],
    muted_creator_ids: list[ObjectId],
    limit: int,
) -> list[str]:
    if not audio_ids or limit <= 0:
        return []
    query = _base_filter(exclude_ids)
    query["audio_id"] = {"$in": audio_ids}
    if muted_creator_ids:
        query["creator_id"] = {"$nin": muted_creator_ids}
    cursor = reels.find(query, {"_id": 1}).sort("created_at", -1).limit(limit)
    return await _reel_ids(cursor)


async def from_trending(
    reels: AsyncIOMotorCollection,
    exclude_ids: list[ObjectId],
    muted_creator_ids: list[ObjectId],
    limit: int,
    window_hours: int,
) -> list[str]:
    if limit <= 0:
        return []
    since = datetime.now(tz=timezone.utc) - timedelta(hours=window_hours)
    query = _base_filter(exclude_ids)
    query["created_at"] = {"$gte": since}
    if muted_creator_ids:
        query["creator_id"] = {"$nin": muted_creator_ids}
    cursor = (
        reels.find(query, {"_id": 1})
        .sort([("viewCount", -1), ("shareCount", -1)])
        .limit(limit)
    )
    return await _reel_ids(cursor)


async def collaborative_filtering_stub(
    user_id: str,
    exclude_ids: list[ObjectId],
    limit: int,
) -> list[str]:
    """Placeholder for item-item / user-user collaborative filtering.

    Intentionally returns no candidates in this batch — a real neighbor model
    is wired in a later batch. Kept as a source so the breakdown and blending
    logic already account for it.
    """
    return []
