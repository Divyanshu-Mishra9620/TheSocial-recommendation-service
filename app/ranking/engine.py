"""Ranking engine orchestration.

`rank_candidates(user_id)` ties together candidate generation, the feature
store, and the heuristic prior scorer, returning candidates ordered by score.
The scorer is swappable (heuristic prior today, XGBoost later) behind
`score_features`.

CODE ONLY — not activated. No API, no background jobs, no Hono. The engine
performs no writes of its own; feature access goes through the feature-store
getters (whose compute-on-miss caching is the feature store's own behavior).
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.candidates import generate_candidates
from app.features.reel_features import get_reel_features
from app.features.user_features import get_user_features
from app.ranking.scorer import score_features

_DEFAULT_LIMIT = 500


async def rank_candidates(
    user_id: str, limit: int = _DEFAULT_LIMIT, seen: list[str] | None = None
) -> dict[str, Any]:
    """Rank a user's candidate reels by the heuristic prior score.

    Returns {"ranked_reel_ids": [...], "scores": {reel_id: score}}.
    """
    candidates = await generate_candidates(user_id, limit=limit, seen=seen)
    reel_ids: list[str] = candidates.get("reel_ids", [])
    if not reel_ids:
        return {"ranked_reel_ids": [], "scores": {}}

    user_features = await get_user_features(user_id)

    # Fetch reel features concurrently (compute-on-miss via the feature store).
    reel_feature_list = await asyncio.gather(
        *(get_reel_features(rid) for rid in reel_ids)
    )

    scores: dict[str, float] = {}
    for reel_id, reel_features in zip(reel_ids, reel_feature_list):
        merged = {**user_features, **reel_features}
        scores[reel_id] = score_features(merged).score

    ranked_reel_ids = sorted(reel_ids, key=lambda rid: scores[rid], reverse=True)

    return {"ranked_reel_ids": ranked_reel_ids, "scores": scores}
