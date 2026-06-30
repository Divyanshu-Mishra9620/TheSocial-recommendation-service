"""Heuristic prior scorer (ranking engine foundation).

Pure functions, zero I/O. Scores a single candidate from a merged feature
vector (user features + reel features). This is the interface a trained XGBoost
model will later replace: same input (a feature dict) → same output (a float
score + per-component breakdown). No reranking, no XGBoost here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Heuristic prior weights. Positive terms reward engagement propensity and
# freshness; the skip term penalizes. Chosen to mirror the existing heuristic
# recommender's intent (reelRecommendation.ts) — NOT a learned model. All
# feature inputs are normalized ~0..1, so weights set relative importance.
WEIGHTS: dict[str, float] = {
    "watch_ratio": 0.4,
    "like": 3.0,
    "share": 5.0,
    "comment": 4.0,
    "save": 4.0,
    "completed": 1.0,
    "freshness": 2.0,
    "skip": -2.0,
}

# Maps each score component to the feature key it reads from the merged vector.
FEATURE_KEYS: dict[str, str] = {
    "watch_ratio": "avg_watch_ratio",
    "like": "like_rate",
    "share": "share_rate",
    "comment": "comment_rate",
    "save": "save_rate",
    "completed": "completion_rate",
    "freshness": "freshness",
    "skip": "skip_rate",
}


@dataclass
class ScoreResult:
    score: float
    components: dict[str, float] = field(default_factory=dict)


def _num(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    # Guard against NaN / Infinity / -Infinity poisoning the score.
    return result if math.isfinite(result) else 0.0


def score_features(features: dict[str, Any]) -> ScoreResult:
    """Compute the heuristic prior score for one merged feature vector.

    Swappable interface: a trained model implements the same signature
    (feature dict in, ScoreResult out).
    """
    components: dict[str, float] = {}
    total = 0.0
    for component, weight in WEIGHTS.items():
        raw = _num(features.get(FEATURE_KEYS[component], 0.0))
        contribution = weight * raw
        components[component] = contribution
        total += contribution
    return ScoreResult(score=total, components=components)
