"""Candidate generation (recall layer).

Produces ~300–500 candidate reel IDs for a user by unioning several recall
sources, de-duplicating, and applying exclusion rules. Returns IDs only — no
scoring, ranking, or reranking (those consume this output in later batches).

Public entrypoint:

    await generate_candidates(user_id, limit=500)
        -> {"reel_ids": [...], "source_breakdown": {...}}

CODE ONLY — not activated anywhere.
"""

from __future__ import annotations

import asyncio
from typing import Any

from bson import ObjectId

from app.candidates import sources
from app.config import get_settings
from app.core.mongo import get_collection
from app.features.user_features import get_user_features

settings = get_settings()

# Recall tuning.
_TOP_TAGS = 10
_SEEN_CAP = 1000
_TRENDING_WINDOW_HOURS = 24
_DEFAULT_LIMIT = 500
# Per-source fetch caps (pre-dedup). The sum slightly exceeds the cap so that
# de-duplication still yields a full candidate set.
_SOURCE_LIMITS = {
    "tag_affinity": 200,
    "creator_affinity": 150,
    "preferred_audio": 100,
    "trending": 150,
    "collaborative": 0,
}


def _to_object_id(value: Any) -> ObjectId | None:
    try:
        return ObjectId(str(value))
    except Exception:
        return None


def _to_object_ids(values: list[Any]) -> list[ObjectId]:
    out: list[ObjectId] = []
    for value in values:
        oid = value if isinstance(value, ObjectId) else _to_object_id(value)
        if oid is not None:
            out.append(oid)
    return out


def _unique_object_ids(values: list[ObjectId]) -> list[ObjectId]:
    return list({str(v): v for v in values}.values())


async def generate_candidates(
    user_id: str, limit: int = _DEFAULT_LIMIT, seen: list[str] | None = None
) -> dict[str, Any]:
    """Generate a de-duplicated, filtered candidate set (reel IDs only).

    `seen` is an optional extra exclusion set (e.g. reels already shown this
    session) merged with the user's interaction-derived seen reels.
    """
    reels = get_collection(settings.reels_collection)
    prefs_col = get_collection(settings.user_preferences_collection)
    interactions_col = get_collection(settings.user_interactions_collection)
    follows_col = get_collection(settings.follows_collection)

    # Affinity from the feature store (compute-on-miss).
    features = await get_user_features(user_id)
    top_tags = list(features.get("tag_affinity", {}).keys())[:_TOP_TAGS]
    affinity_creators = list(features.get("creator_affinity", {}).keys())

    # Preferences (audio, muted creators) + the real follow graph (Follow.ts —
    # userreelpreferences.followed_creators was a dead field with no writer).
    prefs = await prefs_col.find_one({"user_id": _to_object_id(user_id)}) or {}
    audio_ids = prefs.get("preferred_audio_ids") or []
    muted_oids = _to_object_ids(prefs.get("muted_creators") or [])
    muted_str = {str(m) for m in muted_oids}

    follow_docs = await follows_col.find(
        {"follower": _to_object_id(user_id), "status": "accepted"},
        {"followee": 1},
    ).to_list(length=None)
    followed = [d["followee"] for d in follow_docs if d.get("followee") is not None]

    # Creator candidates = affinity ∪ followed, minus muted.
    creator_oids = _unique_object_ids(
        _to_object_ids(affinity_creators) + _to_object_ids(followed)
    )
    creator_oids = [c for c in creator_oids if str(c) not in muted_str]

    # Already-seen reels (exclusion set).
    seen_docs = (
        await interactions_col.find(
            {"user_id": _to_object_id(user_id)}, {"reel_id": 1}
        )
        .limit(_SEEN_CAP)
        .to_list(length=_SEEN_CAP)
    )
    seen_ids = [d["reel_id"] for d in seen_docs if d.get("reel_id") is not None]
    if seen:
        seen_ids.extend(_to_object_ids(seen))

    # Run recall sources concurrently.
    tag_c, creator_c, audio_c, trending_c, collab_c = await asyncio.gather(
        sources.from_tag_affinity(
            reels, top_tags, seen_ids, muted_oids, _SOURCE_LIMITS["tag_affinity"]
        ),
        sources.from_creator_affinity(
            reels, creator_oids, seen_ids, _SOURCE_LIMITS["creator_affinity"]
        ),
        sources.from_preferred_audio(
            reels, audio_ids, seen_ids, muted_oids, _SOURCE_LIMITS["preferred_audio"]
        ),
        sources.from_trending(
            reels,
            seen_ids,
            muted_oids,
            _SOURCE_LIMITS["trending"],
            _TRENDING_WINDOW_HOURS,
        ),
        sources.collaborative_filtering_stub(
            user_id, seen_ids, _SOURCE_LIMITS["collaborative"]
        ),
    )

    ordered = [
        ("tag_affinity", tag_c),
        ("creator_affinity", creator_c),
        ("preferred_audio", audio_c),
        ("trending", trending_c),
        ("collaborative", collab_c),
    ]

    # De-duplicate across sources (priority order), capping at `limit`.
    chosen: set[str] = set()
    reel_ids: list[str] = []
    breakdown = {name: 0 for name, _ in ordered}

    for name, ids in ordered:
        for rid in ids:
            if rid in chosen:
                continue
            chosen.add(rid)
            reel_ids.append(rid)
            breakdown[name] += 1
            if len(reel_ids) >= limit:
                break
        if len(reel_ids) >= limit:
            break

    return {"reel_ids": reel_ids, "source_breakdown": breakdown}
