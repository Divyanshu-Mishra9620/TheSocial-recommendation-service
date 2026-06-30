"""Reel event stream consumer.

Pipeline:  trackReelEvent → Redis Stream (reel:events) → THIS consumer →
ReelEvent Mongo collection.

Consumes raw reel engagement events from the Redis Stream the gateway
dual-writes and persists each into the ReelEvent collection. It uses a Redis
consumer group (at-least-once delivery), is idempotent (each event is keyed by
its stream entry id with a unique index), reclaims entries left pending by
crashed/slow consumers (XAUTOCLAIM), and drops poison-pill entries after a
bounded number of retries.

Run as a standalone worker (it is NOT started by the FastAPI app):

    python -m app.ingest.event_consumer

It stays inert and exits immediately unless EVENT_STREAM_ENABLED is true. No
feature engineering, ranking, or feed logic lives here.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from app.config import get_settings
from app.core.mongo import close_mongo, ensure_indexes
from app.core.redis import close_redis, get_redis
from app.ingest.repository import persist_event

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.event_consumer")


async def ensure_group() -> None:
    """Create the consumer group if it does not already exist."""
    client = get_redis()
    try:
        await client.xgroup_create(
            name=settings.reel_events_stream,
            groupname=settings.reel_events_group,
            id="0",
            mkstream=True,
        )
        logger.info(
            "created consumer group %s on %s",
            settings.reel_events_group,
            settings.reel_events_stream,
        )
    except ResponseError as exc:
        if "BUSYGROUP" in str(exc):
            logger.info("consumer group %s already exists", settings.reel_events_group)
        else:
            raise


def _parse_payload(fields: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the JSON payload published by the gateway (single 'data' field)."""
    raw = fields.get("data")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


async def _handle_entry(client: Redis, entry_id: str, fields: dict[str, Any]) -> None:
    """Persist one entry, then acknowledge it.

    A poison-pill (unparseable) payload is acknowledged and dropped to avoid an
    infinite redelivery loop. Persistence errors propagate so the caller can
    leave the entry pending for retry/reclaim.
    """
    payload = _parse_payload(fields)
    if payload is None:
        logger.warning("dropping unparseable reel event %s", entry_id)
        await client.xack(
            settings.reel_events_stream, settings.reel_events_group, entry_id
        )
        return

    inserted = await persist_event(entry_id, payload)
    await client.xack(
        settings.reel_events_stream, settings.reel_events_group, entry_id
    )
    if inserted:
        logger.debug("persisted reel event %s", entry_id)
    else:
        logger.debug("reel event %s already persisted (idempotent skip)", entry_id)


async def _process(client: Redis, entries: list[tuple[str, dict[str, Any]]]) -> int:
    handled = 0
    for entry_id, fields in entries:
        try:
            await _handle_entry(client, entry_id, fields)
            handled += 1
        except Exception:
            # Do NOT ack: leave the entry pending so it is retried/reclaimed.
            logger.exception("failed to process reel event %s; will retry", entry_id)
    return handled


async def consume_new(client: Redis) -> int:
    """Read + process a batch of new entries (group delivery '>')."""
    response = await client.xreadgroup(
        groupname=settings.reel_events_group,
        consumername=settings.reel_events_consumer,
        streams={settings.reel_events_stream: ">"},
        count=settings.consumer_batch_size,
        block=settings.consumer_block_ms,
    )
    if not response:
        return 0
    total = 0
    for _stream, entries in response:
        total += await _process(client, entries)
    return total


async def reclaim_pending(client: Redis) -> None:
    """Reclaim entries left pending by crashed/slow consumers and retry them.

    Entries exceeding the retry budget are dropped (acked) to prevent a poison
    pill from blocking the group forever.
    """
    try:
        # XAUTOCLAIM → (next_cursor, claimed_entries, deleted_ids).
        result = await client.xautoclaim(
            name=settings.reel_events_stream,
            groupname=settings.reel_events_group,
            consumername=settings.reel_events_consumer,
            min_idle_time=settings.consumer_claim_min_idle_ms,
            start_id="0-0",
            count=settings.consumer_batch_size,
        )
    except ResponseError:
        # Stream/group missing or server lacks XAUTOCLAIM — nothing to reclaim.
        return

    claimed = result[1] if len(result) > 1 else []
    if not claimed:
        return

    over_budget: set[str] = set()
    try:
        pending = await client.xpending_range(
            name=settings.reel_events_stream,
            groupname=settings.reel_events_group,
            min="-",
            max="+",
            count=settings.consumer_batch_size,
        )
        for item in pending:
            if int(item.get("times_delivered", 0)) > settings.consumer_max_retries:
                over_budget.add(item["message_id"])
    except ResponseError:
        pass

    for entry_id, fields in claimed:
        if entry_id in over_budget:
            logger.error(
                "dropping reel event %s after exceeding retry budget", entry_id
            )
            await client.xack(
                settings.reel_events_stream, settings.reel_events_group, entry_id
            )
            continue
        try:
            await _handle_entry(client, entry_id, fields)
        except Exception:
            logger.exception("retry failed for reclaimed reel event %s", entry_id)


async def run() -> None:
    """Main consumer loop. Inert unless EVENT_STREAM_ENABLED is set."""
    if not settings.event_stream_enabled:
        logger.info(
            "event stream consumer disabled (EVENT_STREAM_ENABLED=false); exiting"
        )
        return

    await ensure_group()
    await ensure_indexes()
    client = get_redis()
    logger.info("event consumer started on %s", settings.reel_events_stream)
    try:
        while True:
            try:
                await reclaim_pending(client)
                await consume_new(client)
            except Exception:
                logger.exception("error in consumer loop")
                await asyncio.sleep(1)
    finally:
        await close_redis()
        await close_mongo()


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO)
    )
    asyncio.run(run())


if __name__ == "__main__":
    main()
