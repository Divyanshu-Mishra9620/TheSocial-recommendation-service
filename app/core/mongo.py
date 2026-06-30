"""Async MongoDB client for the Recommendation Service.

Lazy singleton over motor, sharing the same MongoDB as the Hono backend (via
MONGO_URI). Used by the event-ingestion consumer to persist ReelEvent docs. No
connection is opened at import time, and the FastAPI app does not import this
module — the web process needs no MongoDB connection to serve health checks.
"""

from __future__ import annotations

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)

from app.config import get_settings

settings = get_settings()

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    """Return a lazily-created shared async Mongo client."""
    global _client
    if _client is None:
        if not settings.mongo_uri:
            raise RuntimeError("MONGO_URI is not configured")
        _client = AsyncIOMotorClient(settings.mongo_uri)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    """Return the target database (MONGO_DB override, else the URI default)."""
    client = get_client()
    if settings.mongo_db:
        return client[settings.mongo_db]
    return client.get_default_database()


def get_collection(name: str) -> AsyncIOMotorCollection:
    """Return an arbitrary collection by name."""
    return get_db()[name]


def get_reel_events_collection() -> AsyncIOMotorCollection:
    return get_db()[settings.reel_events_collection]


async def ensure_indexes() -> None:
    """Create the idempotency index (safe to call repeatedly)."""
    collection = get_reel_events_collection()
    await collection.create_index(
        "event_id", unique=True, sparse=True, name="uniq_event_id"
    )


async def close_mongo() -> None:
    """Close the shared client (call on shutdown)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
