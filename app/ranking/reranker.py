"""Diversified reranker.

Consumes rank_candidates(user_id) and diversifies it into the final feed order:
  - Maximum Marginal Relevance (MMR) to penalize redundancy,
  - hard caps on consecutive same-creator and identical-tag reels,
  - a small freshness boost,
  - a small, DETERMINISTIC exploration factor (stable per reel id).

Deterministic: no randomness; the exploration term and tie-breaks derive from a
stable SHA1 hash of the reel id, so the same inputs always yield the same order.

CODE ONLY — not activated. No API / Hono / background jobs / route changes. It
reads reel metadata from Mongo (no writes); the only transitive Redis writes are
the feature store's compute-on-miss caching (acknowledged behavior). No XGBoost.

Public entrypoint:

    await rerank_feed(user_id) -> {"final_reel_ids": [...], "metadata": {...}}
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from app.config import get_settings
from app.core.mongo import get_collection
from app.ranking.engine import rank_candidates

settings = get_settings()

# Reranker tuning (heuristic, deterministic).
_MMR_LAMBDA = 0.7  # relevance vs. diversity tradeoff
_MAX_CONSECUTIVE_CREATOR = 2  # no more than N in a row from one creator
_MAX_CONSECUTIVE_TAG = 2  # no more than N in a row with identical tag sets
_FRESHNESS_WEIGHT = 0.15  # small freshness boost
_EXPLORATION_EPSILON = 0.05  # small deterministic perturbation
_FRESHNESS_HORIZON_HOURS = 24.0 * 7.0
_DEFAULT_LIMIT = 20  # final feed size
_CANDIDATE_POOL = 500  # candidate/ranking pool size (decoupled from final size)


def _stable_unit(reel_id: str) -> float:
    """Deterministic value in [0, 1) derived from the reel id (not random)."""
    digest = hashlib.sha1(reel_id.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10_000) / 10_000.0


def _to_object_id(value: str) -> ObjectId | None:
    try:
        return ObjectId(value)
    except Exception:
        return None


def _freshness(created_at: Any) -> float:
    if not isinstance(created_at, datetime):
        return 0.0
    created = created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_hours = max(
        0.0, (datetime.now(tz=timezone.utc) - created).total_seconds() / 3600.0
    )
    return max(0.0, 1.0 - (age_hours / _FRESHNESS_HORIZON_HOURS))


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi <= lo:
        return {k: 0.5 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


def _tag_signature(tags: list[str]) -> tuple[str, ...]:
    return tuple(sorted(tags or []))


def _similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Redundancy similarity in [0, 1]: blend of same-creator and tag Jaccard."""
    creator_match = (
        1.0
        if a.get("creator_id") and a.get("creator_id") == b.get("creator_id")
        else 0.0
    )
    tags_a = set(a.get("tags") or [])
    tags_b = set(b.get("tags") or [])
    union = tags_a | tags_b
    jaccard = len(tags_a & tags_b) / len(union) if union else 0.0
    return 0.5 * creator_match + 0.5 * jaccard


def _violates_consecutive(
    selected_meta: list[dict[str, Any]], cand: dict[str, Any]
) -> bool:
    cid = cand.get("creator_id")
    if cid:
        run = 0
        for m in reversed(selected_meta):
            if m.get("creator_id") == cid:
                run += 1
            else:
                break
        if run >= _MAX_CONSECUTIVE_CREATOR:
            return True

    sig = cand.get("tag_sig")
    if sig:
        run = 0
        for m in reversed(selected_meta):
            if m.get("tag_sig") == sig:
                run += 1
            else:
                break
        if run >= _MAX_CONSECUTIVE_TAG:
            return True
    return False


async def _fetch_metadata(reel_ids: list[str]) -> dict[str, dict[str, Any]]:
    oids = [oid for oid in (_to_object_id(rid) for rid in reel_ids) if oid is not None]
    if not oids:
        return {}
    reels = get_collection(settings.reels_collection)
    docs = await reels.find(
        {"_id": {"$in": oids}},
        {"tags": 1, "creator_id": 1, "created_at": 1},
    ).to_list(length=len(oids))

    meta: dict[str, dict[str, Any]] = {}
    for doc in docs:
        rid = str(doc["_id"])
        tags = doc.get("tags") or []
        meta[rid] = {
            "creator_id": str(doc["creator_id"]) if doc.get("creator_id") else None,
            "tags": tags,
            "tag_sig": _tag_signature(tags),
            "freshness": _freshness(doc.get("created_at")),
        }
    return meta


async def rerank_feed(
    user_id: str, limit: int = _DEFAULT_LIMIT, seen: list[str] | None = None
) -> dict[str, Any]:
    """Diversify the ranked candidate list deterministically.

    Ranks a large candidate pool (`_CANDIDATE_POOL`) and returns the top `limit`
    after diversification, so a small requested size does not starve recall.
    """
    ranked = await rank_candidates(user_id, limit=_CANDIDATE_POOL, seen=seen)
    reel_ids: list[str] = ranked.get("ranked_reel_ids", [])
    scores: dict[str, float] = ranked.get("scores", {})
    source_breakdown: dict[str, Any] = ranked.get("source_breakdown", {})

    if not reel_ids:
        return {
            "final_reel_ids": [],
            "metadata": {
                "candidates_in": 0,
                "reranked_out": 0,
                "deterministic": True,
                "source_breakdown": source_breakdown,
            },
        }

    meta = await _fetch_metadata(reel_ids)
    norm = _normalize(scores)

    # Adjusted relevance = normalized score + freshness boost + exploration.
    relevance: dict[str, float] = {}
    for rid in reel_ids:
        m = meta.get(rid, {})
        relevance[rid] = (
            norm.get(rid, 0.0)
            + _FRESHNESS_WEIGHT * float(m.get("freshness", 0.0))
            + _EXPLORATION_EPSILON * _stable_unit(rid)
        )

    # Greedy MMR with incremental max-similarity tracking (O(n^2)).
    max_sim = {rid: 0.0 for rid in reel_ids}
    remaining = list(reel_ids)
    selected: list[str] = []
    selected_meta: list[dict[str, Any]] = []

    while remaining and len(selected) < limit:
        best: tuple[float, str] | None = None
        best_blocked: tuple[float, str] | None = None
        for rid in remaining:
            mmr = _MMR_LAMBDA * relevance[rid] - (1.0 - _MMR_LAMBDA) * max_sim[rid]
            cand_key = (mmr, rid)
            if _violates_consecutive(selected_meta, meta.get(rid, {})):
                if best_blocked is None or cand_key > best_blocked:
                    best_blocked = cand_key
                continue
            if best is None or cand_key > best:
                best = cand_key

        pick = best or best_blocked
        if pick is None:
            break
        chosen = pick[1]
        selected.append(chosen)
        selected_meta.append(meta.get(chosen, {}))
        remaining.remove(chosen)

        chosen_meta = meta.get(chosen, {})
        for rid in remaining:
            sim = _similarity(meta.get(rid, {}), chosen_meta)
            if sim > max_sim[rid]:
                max_sim[rid] = sim

    unique_creators = len(
        {m.get("creator_id") for m in selected_meta if m.get("creator_id")}
    )
    metadata = {
        "candidates_in": len(reel_ids),
        "reranked_out": len(selected),
        "unique_creators": unique_creators,
        "mmr_lambda": _MMR_LAMBDA,
        "max_consecutive_creator": _MAX_CONSECUTIVE_CREATOR,
        "max_consecutive_tag": _MAX_CONSECUTIVE_TAG,
        "freshness_weight": _FRESHNESS_WEIGHT,
        "exploration_epsilon": _EXPLORATION_EPSILON,
        "deterministic": True,
        "source_breakdown": source_breakdown,
    }
    return {"final_reel_ids": selected, "metadata": metadata}
