"""Unit tests for the event ingestion pipeline (repository + consumer)."""

from datetime import datetime

from app.ingest import event_consumer, repository


# ---- repository ----------------------------------------------------------

def test_build_document_converts_ts_and_optionals():
    payload = {
        "user_id": "507f1f77bcf86cd799439011",
        "reel_id": "507f1f77bcf86cd799439012",
        "event_type": "like",
        "watch_time": 5,
        "ts": 1_700_000_000_000,
    }
    doc = repository._build_document(payload)
    assert doc["event_type"] == "like"
    assert isinstance(doc["ts"], datetime)
    assert doc["ts"].tzinfo is not None
    assert doc["watch_time"] == 5
    assert "completion_rate" not in doc  # omitted when absent
    assert "event_id" not in doc  # set from the upsert filter, not the body


def test_build_document_defaults_ts_when_missing():
    doc = repository._build_document(
        {"user_id": "a", "reel_id": "b", "event_type": "view"}
    )
    assert isinstance(doc["ts"], datetime)


def test_build_document_persists_completion_rate_when_present():
    doc = repository._build_document(
        {"user_id": "a", "reel_id": "b", "event_type": "completed", "completion_rate": 0.75}
    )
    assert doc["completion_rate"] == 0.75


def test_build_document_skips_null_completion_rate():
    doc = repository._build_document(
        {"user_id": "a", "reel_id": "b", "event_type": "like", "completion_rate": None}
    )
    assert "completion_rate" not in doc  # null → stays null (absent) in ReelEvent


async def test_persist_event_inserts(monkeypatch):
    captured = {}

    class _Result:
        upserted_id = "new-id"

    class _Collection:
        async def update_one(self, flt, update, upsert):
            captured["filter"] = flt
            captured["update"] = update
            captured["upsert"] = upsert
            return _Result()

    monkeypatch.setattr(repository, "get_reel_events_collection", lambda: _Collection())
    inserted = await repository.persist_event(
        "1-0", {"user_id": "a", "reel_id": "b", "event_type": "like", "ts": 1_700_000_000_000}
    )
    assert inserted is True
    assert captured["filter"] == {"event_id": "1-0"}
    assert "$setOnInsert" in captured["update"]
    assert captured["upsert"] is True


async def test_persist_event_duplicate_is_idempotent(monkeypatch):
    class _Result:
        upserted_id = None  # already existed

    class _Collection:
        async def update_one(self, *args, **kwargs):
            return _Result()

    monkeypatch.setattr(repository, "get_reel_events_collection", lambda: _Collection())
    inserted = await repository.persist_event(
        "1-0", {"user_id": "a", "reel_id": "b", "event_type": "like"}
    )
    assert inserted is False


# ---- consumer -------------------------------------------------------------

def test_parse_payload():
    assert event_consumer._parse_payload({"data": '{"a": 1}'}) == {"a": 1}
    assert event_consumer._parse_payload({}) is None
    assert event_consumer._parse_payload({"data": "not-json"}) is None


async def test_handle_entry_persists_and_acks(monkeypatch):
    acked = {}

    class _Client:
        async def xack(self, stream, group, entry_id):
            acked["id"] = entry_id

    async def fake_persist(event_id, payload):
        return True

    monkeypatch.setattr(event_consumer, "persist_event", fake_persist)
    await event_consumer._handle_entry(
        _Client(),
        "5-0",
        {"data": '{"user_id": "a", "reel_id": "b", "event_type": "like"}'},
    )
    assert acked["id"] == "5-0"


async def test_handle_entry_drops_poison_pill(monkeypatch):
    acked = {}
    called = {"persist": False}

    class _Client:
        async def xack(self, stream, group, entry_id):
            acked["id"] = entry_id

    async def fake_persist(*args):
        called["persist"] = True
        return True

    monkeypatch.setattr(event_consumer, "persist_event", fake_persist)
    await event_consumer._handle_entry(_Client(), "6-0", {"data": "garbage"})
    assert acked["id"] == "6-0"  # dropped (acked) so it is not redelivered forever
    assert called["persist"] is False
