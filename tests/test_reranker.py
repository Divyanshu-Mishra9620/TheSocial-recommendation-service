"""Unit tests for the diversified reranker (pure helpers + orchestration)."""

from app.ranking import reranker


# ---- pure helpers ---------------------------------------------------------

def test_stable_unit_is_deterministic_and_bounded():
    a = reranker._stable_unit("reel-abc")
    b = reranker._stable_unit("reel-abc")
    assert a == b
    assert 0.0 <= a < 1.0


def test_normalize():
    assert reranker._normalize({}) == {}
    assert reranker._normalize({"a": 5, "b": 5}) == {"a": 0.5, "b": 0.5}
    normalized = reranker._normalize({"a": 0, "b": 10})
    assert normalized["a"] == 0.0
    assert normalized["b"] == 1.0


def test_similarity_creator_and_tags():
    a = {"creator_id": "X", "tags": ["t1", "t2"]}
    same = {"creator_id": "X", "tags": ["t1", "t2"]}
    different = {"creator_id": "Y", "tags": ["t3"]}
    assert reranker._similarity(a, same) == 1.0
    assert reranker._similarity(a, different) == 0.0


def test_violates_consecutive_creator_cap():
    selected = [
        {"creator_id": "A", "tag_sig": ("x",)},
        {"creator_id": "A", "tag_sig": ("x",)},
    ]
    assert reranker._violates_consecutive(selected, {"creator_id": "A", "tag_sig": ("z",)}) is True
    assert reranker._violates_consecutive(selected, {"creator_id": "B", "tag_sig": ("z",)}) is False


# ---- orchestration (mocked rank_candidates + metadata) --------------------

async def test_rerank_is_deterministic_and_caps_consecutive_creators(monkeypatch):
    ids = ["r1", "r2", "r3", "r4", "r5", "r6"]
    scores = {rid: 1.0 - i * 0.1 for i, rid in enumerate(ids)}

    async def fake_rank(user_id, limit=500, seen=None):
        return {"ranked_reel_ids": ids, "scores": scores}

    # creators: r1-r3 = A, r4-r6 = B
    meta = {
        rid: {
            "creator_id": "A" if rid in ("r1", "r2", "r3") else "B",
            "tags": ["x"] if rid in ("r1", "r2", "r3") else ["y"],
            "tag_sig": ("x",) if rid in ("r1", "r2", "r3") else ("y",),
            "freshness": 0.5,
        }
        for rid in ids
    }

    async def fake_meta(reel_ids):
        return meta

    monkeypatch.setattr(reranker, "rank_candidates", fake_rank)
    monkeypatch.setattr(reranker, "_fetch_metadata", fake_meta)

    first = await reranker.rerank_feed("u", limit=6)
    second = await reranker.rerank_feed("u", limit=6)

    assert first["final_reel_ids"] == second["final_reel_ids"]  # deterministic
    assert sorted(first["final_reel_ids"]) == sorted(ids)  # all present, no loss

    order = first["final_reel_ids"]
    run = 1
    for i in range(1, len(order)):
        if meta[order[i]]["creator_id"] == meta[order[i - 1]]["creator_id"]:
            run += 1
            assert run <= 2, "more than 2 consecutive reels from the same creator"
        else:
            run = 1


async def test_rerank_empty_input(monkeypatch):
    async def fake_rank(user_id, limit=500, seen=None):
        return {"ranked_reel_ids": [], "scores": {}}

    monkeypatch.setattr(reranker, "rank_candidates", fake_rank)
    result = await reranker.rerank_feed("u", limit=10)
    assert result["final_reel_ids"] == []


async def test_rerank_propagates_source_breakdown_into_metadata(monkeypatch):
    """rank_candidates' per-source recall counts must reach the final
    metadata dict — previously computed by generate_candidates() but
    silently dropped by rank_candidates(), leaving no way to see why a given
    user got a thin/generic feed."""
    breakdown = {"tag_affinity": 3, "creator_affinity": 5, "trending": 2}

    async def fake_rank(user_id, limit=500, seen=None):
        return {
            "ranked_reel_ids": ["r1"],
            "scores": {"r1": 1.0},
            "source_breakdown": breakdown,
        }

    async def fake_meta(reel_ids):
        return {"r1": {"creator_id": "A", "tags": ["x"], "tag_sig": ("x",), "freshness": 0.5}}

    monkeypatch.setattr(reranker, "rank_candidates", fake_rank)
    monkeypatch.setattr(reranker, "_fetch_metadata", fake_meta)

    result = await reranker.rerank_feed("u", limit=10)
    assert result["metadata"]["source_breakdown"] == breakdown


async def test_rerank_empty_input_still_includes_source_breakdown(monkeypatch):
    async def fake_rank(user_id, limit=500, seen=None):
        return {"ranked_reel_ids": [], "scores": {}, "source_breakdown": {"trending": 0}}

    monkeypatch.setattr(reranker, "rank_candidates", fake_rank)
    result = await reranker.rerank_feed("u", limit=10)
    assert result["metadata"]["source_breakdown"] == {"trending": 0}
