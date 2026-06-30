"""Persistence for raw reel events (the `reelevents` collection).

Idempotent: each event is keyed by its Redis Stream entry id (`event_id`) with
a unique index, so re-delivering the same stream entry never creates a
duplicate. No feature engineering or aggregation happens here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.config import get_settings
from app.core.mongo import get_reel_events_collection

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.repository")


def _to_object_id(value: Any) -> ObjectId | Any:
    """Best-effort convert a hex string to ObjectId; keep original on failure."""
    try:
        return ObjectId(str(value))
    except Exception:
        return value


def _build_document(payload: dict[str, Any]) -> dict[str, Any]:
    """Map a stream payload to a ReelEvent document (without event_id, which is
    applied from the upsert filter)."""
    ts_raw = payload.get("ts")
    if isinstance(ts_raw, (int, float)):
        ts = datetime.fromtimestamp(ts_raw / 1000, tz=timezone.utc)
    else:
        ts = datetime.now(tz=timezone.utc)

    doc: dict[str, Any] = {
        "user_id": _to_object_id(payload.get("user_id")),
        "reel_id": _to_object_id(payload.get("reel_id")),
        "event_type": payload.get("event_type"),
        "ts": ts,
    }
    for opt in ("watch_time", "completion_rate", "session_id", "source"):
        if payload.get(opt) is not None:
            doc[opt] = payload[opt]
    return doc


async def persist_event(event_id: str, payload: dict[str, Any]) -> bool:
    """Idempotently persist one event.

    Returns True if newly inserted, False if it already existed (duplicate
    stream redelivery). event_id is set from the upsert filter, not the body.
    """
    collection = get_reel_events_collection()
    doc = _build_document(payload)
    try:
        result = await collection.update_one(
            {"event_id": event_id},
            {"$setOnInsert": doc},
            upsert=True,
        )
        return result.upserted_id is not None
    except DuplicateKeyError:
        # Concurrent insert of the same event_id — already persisted.
        return False
