"""Offline dataset builder for the XGBoost ranker.

Reads raw events from the ReelEvent collection, aggregates per (user, reel),
derives an engagement label, joins the SAME user/reel features used at serving
time, and produces a time-based train/validation split (validation is strictly
later in time → no leakage).

OFFLINE ONLY. Module top imports only stdlib so the pure helpers
(aggregate_events / compute_label_score / time_split) are unit-testable without
pandas/motor/redis; pandas and the app feature getters are imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

# Numeric feature columns for the model — the same signals the serving feature
# vector exposes (user engagement rates + reel features). Kept in sync with
# app.ranking.scorer.FEATURE_KEYS plus the extra reel counters the model can use.
FEATURE_COLUMNS: list[str] = [
    # user features
    "like_rate",
    "share_rate",
    "comment_rate",
    "skip_rate",
    "completion_rate",
    "save_rate",
    "avg_watch_ratio",
    # reel features
    "viewCount",
    "likeCount",
    "shareCount",
    "commentCount",
    "freshness",
    "velocity",
]

# Label weights mirror the heuristic prior so the learned model and the
# cold-start prior are aligned.
LABEL_WEIGHTS: dict[str, float] = {
    "watch_ratio": 0.4,
    "like": 3.0,
    "share": 5.0,
    "comment": 4.0,
    "save": 4.0,
    "completed": 1.0,
    "skip": -2.0,
}

# Binary relevance threshold (label = 1 when the engagement score clears this).
POSITIVE_THRESHOLD = 1.0

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass
class InteractionAggregate:
    user_id: str
    reel_id: str
    ts: datetime
    watch_ratio: float = 0.0
    like: int = 0
    share: int = 0
    comment: int = 0
    save: int = 0
    completed: int = 0
    skip: int = 0


def aggregate_events(events: Iterable[dict[str, Any]]) -> list[InteractionAggregate]:
    """Reduce raw events to one aggregate per (user, reel). Pure."""
    grouped: dict[tuple[str, str], InteractionAggregate] = {}
    for event in events:
        user_id = str(event.get("user_id"))
        reel_id = str(event.get("reel_id"))
        key = (user_id, reel_id)

        ts = event.get("ts")
        if not isinstance(ts, datetime):
            ts = _EPOCH

        agg = grouped.get(key)
        if agg is None:
            agg = InteractionAggregate(user_id=user_id, reel_id=reel_id, ts=ts)
            grouped[key] = agg
        elif ts > agg.ts:
            agg.ts = ts  # keep the latest interaction time

        event_type = event.get("event_type")
        if event_type == "like":
            agg.like = 1
        elif event_type == "share":
            agg.share = 1
        elif event_type == "comment":
            agg.comment = 1
        elif event_type == "save":
            agg.save = 1
        elif event_type == "completed":
            agg.completed = 1
        elif event_type == "skip":
            agg.skip = 1

        completion_rate = event.get("completion_rate")
        if isinstance(completion_rate, (int, float)):
            agg.watch_ratio = max(agg.watch_ratio, min(1.0, float(completion_rate)))

    return list(grouped.values())


def compute_label_score(agg: InteractionAggregate) -> float:
    """Weighted engagement score (regression target / threshold source). Pure."""
    return (
        LABEL_WEIGHTS["watch_ratio"] * agg.watch_ratio
        + LABEL_WEIGHTS["like"] * agg.like
        + LABEL_WEIGHTS["share"] * agg.share
        + LABEL_WEIGHTS["comment"] * agg.comment
        + LABEL_WEIGHTS["save"] * agg.save
        + LABEL_WEIGHTS["completed"] * agg.completed
        + LABEL_WEIGHTS["skip"] * agg.skip
    )


def compute_binary_label(agg: InteractionAggregate) -> int:
    """Binary relevance label for classification framing. Pure."""
    return 1 if compute_label_score(agg) >= POSITIVE_THRESHOLD else 0


def time_split(
    aggregates: list[InteractionAggregate], train_ratio: float = 0.8
) -> tuple[list[InteractionAggregate], list[InteractionAggregate]]:
    """Chronological split: earliest `train_ratio` is train, the rest is
    validation. Validation is strictly later in time → prevents leakage. Pure."""
    ordered = sorted(aggregates, key=lambda a: a.ts)
    cut = int(len(ordered) * train_ratio)
    return ordered[:cut], ordered[cut:]


def _feature_value(user_features: dict, reel_features: dict, column: str) -> float:
    raw = reel_features[column] if column in reel_features else user_features.get(column, 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


async def _to_frame(pd, aggregates, get_user_features, get_reel_features):
    """Build a feature DataFrame from aggregates (lazy — `pd` passed in)."""
    user_cache: dict[str, dict] = {}
    rows: list[dict[str, Any]] = []
    for agg in aggregates:
        uf = user_cache.get(agg.user_id)
        if uf is None:
            uf = await get_user_features(agg.user_id)
            user_cache[agg.user_id] = uf
        rf = await get_reel_features(agg.reel_id)

        row: dict[str, Any] = {c: _feature_value(uf, rf, c) for c in FEATURE_COLUMNS}
        row["label"] = compute_binary_label(agg)
        row["score"] = compute_label_score(agg)
        row["user_id"] = agg.user_id
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["label", "score", "user_id"])


async def build_dataset(
    limit: int | None = None,
    train_ratio: float = 0.8,
    since: datetime | None = None,
):
    """Read ReelEvent → aggregate → time-split → feature frames (train, val).

    `since` (optional) restricts to events at/after that time — used by
    incremental retraining.
    """
    import pandas as pd

    from app.config import get_settings
    from app.core.mongo import get_collection
    from app.features.reel_features import get_reel_features
    from app.features.user_features import get_user_features

    settings = get_settings()
    collection = get_collection(settings.reel_events_collection)
    query: dict[str, Any] = {} if since is None else {"ts": {"$gte": since}}
    cursor = collection.find(query).sort("ts", 1)
    if limit:
        cursor = cursor.limit(limit)
    events = await cursor.to_list(length=limit)

    aggregates = aggregate_events(events)
    train_aggs, val_aggs = time_split(aggregates, train_ratio)
    train_df = await _to_frame(pd, train_aggs, get_user_features, get_reel_features)
    val_df = await _to_frame(pd, val_aggs, get_user_features, get_reel_features)
    return train_df, val_df


def main() -> None:
    import asyncio
    import os

    from app.config import get_settings

    settings = get_settings()
    train_df, val_df = asyncio.run(build_dataset())
    os.makedirs(settings.model_artifacts_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(settings.model_artifacts_dir, "train.parquet"))
    val_df.to_parquet(os.path.join(settings.model_artifacts_dir, "val.parquet"))
    print(f"dataset built: train={len(train_df)} val={len(val_df)}")


if __name__ == "__main__":
    main()
