"""Offline evaluation metrics: AUC, NDCG@10, MAP.

OFFLINE ONLY. Heavy deps (numpy/scikit-learn) are imported lazily.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence


def compute_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    groups: Sequence[Any] | None = None,
    k: int = 10,
) -> dict[str, float]:
    """Return {auc, map, ndcg@k}. `groups` (e.g. user ids) define NDCG queries."""
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        ndcg_score,
        roc_auc_score,
    )

    y_true = list(y_true)
    y_score = list(y_score)
    has_both_classes = len(set(y_true)) > 1

    metrics: dict[str, float] = {}
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_score)) if has_both_classes else 0.0
    except Exception:
        metrics["auc"] = 0.0
    try:
        metrics["map"] = float(average_precision_score(y_true, y_score)) if has_both_classes else 0.0
    except Exception:
        metrics["map"] = 0.0

    metrics[f"ndcg@{k}"] = _ndcg_at_k(np, ndcg_score, y_true, y_score, groups, k)
    return metrics


def _ndcg_at_k(np, ndcg_score, y_true, y_score, groups, k) -> float:
    if groups is None:
        groups = [0] * len(y_true)

    by_group: dict[Any, tuple[list, list]] = defaultdict(lambda: ([], []))
    for true, score, group in zip(y_true, y_score, groups):
        by_group[group][0].append(true)
        by_group[group][1].append(score)

    per_group: list[float] = []
    for trues, scores in by_group.values():
        if len(trues) < 2 or sum(trues) == 0:
            continue  # NDCG undefined for single-doc or all-negative queries
        try:
            per_group.append(float(ndcg_score([trues], [scores], k=k)))
        except Exception:
            continue
    return float(np.mean(per_group)) if per_group else 0.0
