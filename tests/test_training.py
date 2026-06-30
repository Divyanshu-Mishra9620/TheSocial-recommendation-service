"""Unit tests for the offline training pipeline (pure helpers only).

These import only the stdlib-top modules (build_dataset, registry), so they run
without the heavy ML stack (pandas/numpy/scikit-learn/xgboost). The train/
evaluate functions that need those libs are syntax-checked separately.
"""

from datetime import datetime, timezone

from training.build_dataset import (
    FEATURE_COLUMNS,
    InteractionAggregate,
    aggregate_events,
    compute_binary_label,
    compute_label_score,
    time_split,
)
from training.registry import build_version_document


def _ts(year: int) -> datetime:
    return datetime(year, 1, 1, tzinfo=timezone.utc)


def test_aggregate_events_sets_flags_and_latest_ts():
    events = [
        {"user_id": "u", "reel_id": "r", "event_type": "like", "ts": _ts(2024)},
        {
            "user_id": "u",
            "reel_id": "r",
            "event_type": "completed",
            "completion_rate": 0.9,
            "ts": _ts(2025),
        },
        {"user_id": "u", "reel_id": "r2", "event_type": "skip", "ts": _ts(2024)},
    ]
    by_key = {(a.user_id, a.reel_id): a for a in aggregate_events(events)}

    first = by_key[("u", "r")]
    assert first.like == 1
    assert first.completed == 1
    assert first.skip == 0
    assert first.watch_ratio == 0.9
    assert first.ts == _ts(2025)  # latest interaction time wins

    second = by_key[("u", "r2")]
    assert second.skip == 1
    assert second.like == 0


def test_labels_positive_and_negative():
    positive = InteractionAggregate("u", "r", _ts(2024), like=1, watch_ratio=0.8)
    negative = InteractionAggregate("u", "r2", _ts(2024), skip=1)

    assert compute_label_score(positive) > 0
    assert compute_binary_label(positive) == 1
    assert compute_label_score(negative) < 0
    assert compute_binary_label(negative) == 0


def test_time_split_prevents_leakage():
    aggs = [InteractionAggregate("u", f"r{i}", _ts(2010 + i)) for i in range(10)]
    aggs = list(reversed(aggs))  # unsorted input

    train, val = time_split(aggs, train_ratio=0.8)

    assert len(train) == 8
    assert len(val) == 2
    assert max(a.ts for a in train) <= min(a.ts for a in val)  # val strictly later


def test_feature_columns_present():
    assert "freshness" in FEATURE_COLUMNS
    assert "like_rate" in FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_build_version_document():
    doc = build_version_document(
        version="v1",
        metrics={"auc": 0.81, "ndcg@10": 0.5},
        artifact_path="models_artifacts/v1.json",
        feature_columns=FEATURE_COLUMNS,
    )
    assert doc["version"] == "v1"
    assert doc["status"] == "candidate"  # never auto-activated
    assert doc["metrics"]["auc"] == 0.81
    assert doc["artifact_path"].endswith("v1.json")
    assert "trained_at" in doc
    assert doc["feature_columns"] == FEATURE_COLUMNS
