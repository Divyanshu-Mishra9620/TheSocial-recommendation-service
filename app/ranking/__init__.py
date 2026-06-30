"""Ranking engine for the Recommendation Service.

Public boundary:

    from app.ranking import rank_candidates
    from app.ranking.scorer import score_features   # swappable; XGBoost later

CODE ONLY (Batch 8): not activated — no API, no background jobs, no Hono, no
reranking, no XGBoost. The heuristic prior here is intended to be replaced by a
trained model behind the same `score_features` interface.
"""

from app.ranking.engine import rank_candidates
from app.ranking.reranker import rerank_feed

__all__ = ["rank_candidates", "rerank_feed"]
