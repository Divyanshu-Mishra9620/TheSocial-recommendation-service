"""User feature computation (compute-on-miss source for the feature store).

Aggregates a user's recent reel interactions into engagement-rate features plus
creator/tag affinity maps. CODE ONLY — not activated anywhere. No candidate
generation, ranking, or ML lives here.
"""

from __future__ import annotations

from typing import Any

from bson import ObjectId

from app.config import get_settings
from app.core.mongo import get_collection
from app.features import store

settings = get_settings()

# Cap how many recent interactions feed the aggregates.
_MAX_INTERACTIONS = 200
# Top-K affinity entries retained per map.
_AFFINITY_TOP_K = 20


def _safe_object_id(value: str) -> ObjectId | str:
    try:
        return ObjectId(value)
    except Exception:
        return value


def _top_n(scores: dict[str, float], n: int) -> dict[str, float]:
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n])


def _empty() -> dict[str, Any]:
    return {
        "like_rate": 0.0,
        "share_rate": 0.0,
        "comment_rate": 0.0,
        "skip_rate": 0.0,
        "completion_rate": 0.0,
        "save_rate": 0.0,
        "avg_watch_ratio": 0.0,
        "creator_affinity": {},
        "tag_affinity": {},
    }


async def compute_user_features(user_id: str) -> dict[str, Any]:
    """Aggregate the user's interactions into the stored feature set."""
    interactions_col = get_collection(settings.user_interactions_collection)
    reels_col = get_collection(settings.reels_collection)

    interactions = (
        await interactions_col.find({"user_id": _safe_object_id(user_id)})
        .sort("last_interaction_at", -1)
        .limit(_MAX_INTERACTIONS)
        .to_list(length=_MAX_INTERACTIONS)
    )

    total = len(interactions)
    if total == 0:
        return _empty()

    liked = sum(1 for i in interactions if i.get("liked"))
    shared = sum(1 for i in interactions if i.get("shared"))
    commented = sum(1 for i in interactions if i.get("commented"))
    skipped = sum(1 for i in interactions if i.get("skipped"))
    completed = sum(1 for i in interactions if i.get("completed"))
    saved = sum(1 for i in interactions if i.get("saved"))

    # Reel metadata for watch-ratio + affinity.
    reel_ids = [i["reel_id"] for i in interactions if i.get("reel_id") is not None]
    reels = await reels_col.find(
        {"_id": {"$in": reel_ids}},
        {"duration": 1, "tags": 1, "creator_id": 1},
    ).to_list(length=len(reel_ids))
    reel_map = {str(r["_id"]): r for r in reels}

    watch_ratios: list[float] = []
    tag_affinity: dict[str, float] = {}
    creator_affinity: dict[str, float] = {}

    for i in interactions:
        reel = reel_map.get(str(i.get("reel_id")))
        if not reel:
            continue

        duration = reel.get("duration") or 0
        watch_time = i.get("watch_time") or 0
        if duration > 0:
            watch_ratios.append(min(1.0, watch_time / duration))

        # Positive-engagement weight drives affinity (mirrors the heuristic).
        weight = (
            (3 if i.get("liked") else 0)
            + (5 if i.get("shared") else 0)
            + (4 if i.get("commented") else 0)
            + (2 if i.get("saved") else 0)
            + (1 if i.get("completed") else 0)
        )
        if weight <= 0:
            continue

        for tag in reel.get("tags", []) or []:
            tag_affinity[tag] = tag_affinity.get(tag, 0.0) + weight
        creator = reel.get("creator_id")
        if creator is not None:
            cid = str(creator)
            creator_affinity[cid] = creator_affinity.get(cid, 0.0) + weight

    avg_watch_ratio = (
        sum(watch_ratios) / len(watch_ratios) if watch_ratios else 0.0
    )

    return {
        "like_rate": liked / total,
        "share_rate": shared / total,
        "comment_rate": commented / total,
        "skip_rate": skipped / total,
        "completion_rate": completed / total,
        "save_rate": saved / total,
        "avg_watch_ratio": avg_watch_ratio,
        "creator_affinity": _top_n(creator_affinity, _AFFINITY_TOP_K),
        "tag_affinity": _top_n(tag_affinity, _AFFINITY_TOP_K),
    }


async def get_user_features(user_id: str) -> dict[str, Any]:
    """Compute-on-miss accessor for a user's features."""
    return await store.get_or_compute(
        store.user_feature_key(user_id),
        settings.feat_user_ttl,
        lambda: compute_user_features(user_id),
    )
