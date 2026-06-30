"""Unit tests for candidate generation (dedup, breakdown, cap)."""

from app.candidates import generator


class _Cursor:
    def __init__(self, docs):
        self._docs = docs

    def limit(self, _n):
        return self

    async def to_list(self, length=None):
        return self._docs


class _Collection:
    def __init__(self, find_one=None, docs=None):
        self._find_one = {} if find_one is None else find_one
        self._docs = docs or []

    async def find_one(self, *args, **kwargs):
        return self._find_one

    def find(self, *args, **kwargs):
        return _Cursor(self._docs)


def _stub_common(monkeypatch):
    async def fake_user_features(_uid):
        return {"tag_affinity": {}, "creator_affinity": {}}

    monkeypatch.setattr(generator, "get_user_features", fake_user_features)
    monkeypatch.setattr(generator, "get_collection", lambda _name: _Collection())


async def test_dedup_and_breakdown(monkeypatch):
    _stub_common(monkeypatch)

    async def tag(*a, **k):
        return ["r1", "r2", "r3"]

    async def creator(*a, **k):
        return ["r2", "r4"]  # r2 duplicate

    async def audio(*a, **k):
        return ["r5"]

    async def trending(*a, **k):
        return ["r3", "r6"]  # r3 duplicate

    async def collab(*a, **k):
        return []

    monkeypatch.setattr(generator.sources, "from_tag_affinity", tag)
    monkeypatch.setattr(generator.sources, "from_creator_affinity", creator)
    monkeypatch.setattr(generator.sources, "from_preferred_audio", audio)
    monkeypatch.setattr(generator.sources, "from_trending", trending)
    monkeypatch.setattr(generator.sources, "collaborative_filtering_stub", collab)

    result = await generator.generate_candidates("u", limit=500)
    ids = result["reel_ids"]

    assert ids == ["r1", "r2", "r3", "r4", "r5", "r6"]  # priority order, deduped
    assert len(ids) == len(set(ids))

    breakdown = result["source_breakdown"]
    assert breakdown["tag_affinity"] == 3
    assert breakdown["creator_affinity"] == 1  # only r4 is new
    assert breakdown["preferred_audio"] == 1
    assert breakdown["trending"] == 1  # only r6 is new
    assert breakdown["collaborative"] == 0


async def test_caps_at_limit(monkeypatch):
    _stub_common(monkeypatch)

    async def many(*a, **k):
        return [f"r{i}" for i in range(1000)]

    async def empty(*a, **k):
        return []

    monkeypatch.setattr(generator.sources, "from_tag_affinity", many)
    monkeypatch.setattr(generator.sources, "from_creator_affinity", empty)
    monkeypatch.setattr(generator.sources, "from_preferred_audio", empty)
    monkeypatch.setattr(generator.sources, "from_trending", empty)
    monkeypatch.setattr(generator.sources, "collaborative_filtering_stub", empty)

    result = await generator.generate_candidates("u", limit=10)
    assert len(result["reel_ids"]) == 10
