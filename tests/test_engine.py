"""Unit tests for rank_candidates' source_breakdown propagation.

generate_candidates() already computes a per-source recall breakdown, but
rank_candidates() previously dropped it — this is the fix, and its tests.
"""

from app.ranking import engine


async def test_rank_candidates_propagates_source_breakdown(monkeypatch):
    breakdown = {"tag_affinity": 2, "creator_affinity": 1, "trending": 3}

    async def fake_generate_candidates(user_id, limit=500, seen=None):
        return {"reel_ids": ["r1", "r2"], "source_breakdown": breakdown}

    async def fake_user_features(_uid):
        return {}

    async def fake_reel_features(_rid):
        return {}

    monkeypatch.setattr(engine, "generate_candidates", fake_generate_candidates)
    monkeypatch.setattr(engine, "get_user_features", fake_user_features)
    monkeypatch.setattr(engine, "get_reel_features", fake_reel_features)

    result = await engine.rank_candidates("u1")

    assert result["source_breakdown"] == breakdown
    assert set(result["ranked_reel_ids"]) == {"r1", "r2"}


async def test_rank_candidates_empty_reel_ids_still_returns_source_breakdown(monkeypatch):
    async def fake_generate_candidates(user_id, limit=500, seen=None):
        return {"reel_ids": [], "source_breakdown": {"trending": 0}}

    monkeypatch.setattr(engine, "generate_candidates", fake_generate_candidates)

    result = await engine.rank_candidates("u1")

    assert result == {
        "ranked_reel_ids": [],
        "scores": {},
        "source_breakdown": {"trending": 0},
    }


async def test_rank_candidates_defaults_to_empty_breakdown_when_missing(monkeypatch):
    """generate_candidates() always includes source_breakdown in practice, but
    rank_candidates() must not crash if some future caller's stub omits it."""

    async def fake_generate_candidates(user_id, limit=500, seen=None):
        return {"reel_ids": []}

    monkeypatch.setattr(engine, "generate_candidates", fake_generate_candidates)

    result = await engine.rank_candidates("u1")

    assert result["source_breakdown"] == {}
