"""Integration tests for POST /internal/v1/feed via FastAPI TestClient.

The pipeline (rerank_feed) is mocked so these exercise the endpoint contract:
auth guard, valid/empty/invalid responses, and graceful error handling.
"""

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

client = TestClient(app)
TOKEN = get_settings().internal_service_token
HEADERS = {"X-Internal-Token": TOKEN}


def test_feed_requires_internal_token():
    resp = client.post("/internal/v1/feed", json={"user_id": "u1"})
    assert resp.status_code == 401


def test_feed_valid(monkeypatch):
    async def fake_rerank(user_id, limit=20, seen=None):
        return {"final_reel_ids": ["a", "b"], "metadata": {"reranked_out": 2}}

    monkeypatch.setattr("app.api.internal.rerank_feed", fake_rerank)
    resp = client.post(
        "/internal/v1/feed",
        json={"user_id": "u1", "limit": 2},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == ["a", "b"]
    assert body["metadata"]["reranked_out"] == 2


def test_feed_empty(monkeypatch):
    async def fake_rerank(user_id, limit=20, seen=None):
        return {"final_reel_ids": [], "metadata": {}}

    monkeypatch.setattr("app.api.internal.rerank_feed", fake_rerank)
    resp = client.post("/internal/v1/feed", json={"user_id": "u1"}, headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["items"] == []


def test_feed_invalid_body_is_422():
    resp = client.post("/internal/v1/feed", json={}, headers=HEADERS)  # missing user_id
    assert resp.status_code == 422


def test_feed_pipeline_error_is_500(monkeypatch):
    async def boom(user_id, limit=20, seen=None):
        raise RuntimeError("pipeline failed")

    monkeypatch.setattr("app.api.internal.rerank_feed", boom)
    resp = client.post("/internal/v1/feed", json={"user_id": "u1"}, headers=HEADERS)
    assert resp.status_code == 500


def test_model_active_requires_token():
    resp = client.get("/internal/v1/model/active")
    assert resp.status_code == 401


def test_model_active_returns_doc(monkeypatch):
    class _Collection:
        async def find_one(self, *args, **kwargs):
            return {
                "version": "reel_ranker_v1",
                "status": "candidate",
                "metrics": {"auc": 0.8},
                "trained_at": "2026-01-01T00:00:00+00:00",
            }

    monkeypatch.setattr("app.core.mongo.get_collection", lambda _name: _Collection())
    resp = client.get("/internal/v1/model/active", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "reel_ranker_v1"
    assert body["status"] == "candidate"
    assert body["metrics"]["auc"] == 0.8
    assert body["trained_at"].startswith("2026-01-01")


def test_model_active_empty(monkeypatch):
    class _Collection:
        async def find_one(self, *args, **kwargs):
            return None

    monkeypatch.setattr("app.core.mongo.get_collection", lambda _name: _Collection())
    resp = client.get("/internal/v1/model/active", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["version"] is None
