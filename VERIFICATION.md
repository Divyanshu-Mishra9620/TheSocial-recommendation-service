# Recommendation Pipeline — Verification

How to verify the full pipeline before any ML model is added. All flags default
**off**, so none of this changes production behavior until explicitly enabled.

```
trackReelEvent → Redis Stream (reel:events) → consumer → ReelEvent (Mongo)
                                                              │
                          candidate generation → ranking → reranking
                                                              │
                              POST /internal/v1/feed  ←  Hono gateway (flagged)
```

## 1. Automated tests

### Recommendation service (Python)

```bash
cd services/recommendation-service
python -m venv .venv && . .venv/Scripts/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest                                                # runs tests/
```

Covers:
- `test_scorer.py` — heuristic prior weights, skip penalty, missing/invalid features.
- `test_reranker.py` — MMR helpers, similarity, consecutive-creator cap, determinism.
- `test_ingest.py` — `ReelEvent` document mapping, idempotent upsert, payload parse, poison-pill drop/ack.
- `test_candidates.py` — cross-source dedup, source breakdown, cap.
- `test_internal_feed.py` — `POST /internal/v1/feed`: token guard (401), valid, empty, invalid body (422), pipeline error (500).

### Gateway (Node)

```bash
cd backend
npm test                # vitest
```

Covers:
- `recommendationFeed.test.ts` — `resolvePersonalizedFeed`: flag off → heuristic; flag on → hydrated rec feed (order preserved); rec null / throw / empty-hydration → heuristic fallback.
- `aiServiceClient.test.ts` — `fetchRecommendedFeed`: valid, empty, invalid, non-OK, **timeout (250ms) → null**, disabled.

## 2. Manual end-to-end (Docker Compose)

> Requires the AI stack running. Enable flags only in a test environment.

```bash
# 0. Set a shared secret for both the gateway and the service.
export INTERNAL_SERVICE_TOKEN=$(openssl rand -base64 32)
docker compose up -d redis rec-service          # + backend, mongo

# 1. Event → stream  (enable publish on the gateway)
#    REC_EVENT_STREAM_ENABLED=true on the backend, then trigger a reel event:
#    POST /api/v1/reels/track-event {reel_id, event_type:"like"}
redis-cli XLEN reel:events                       # expect > 0

# 2. Stream → ReelEvent  (run the consumer with EVENT_STREAM_ENABLED=true)
docker compose exec rec-service \
  sh -c 'EVENT_STREAM_ENABLED=true python -m app.ingest.event_consumer'
#    In Mongo: db.reelevents.countDocuments() increases; re-running is idempotent
#    (unique event_id index → no duplicates).

# 3. Internal feed (token-guarded, internal network only)
docker compose exec rec-service sh -c \
  'curl -s -X POST localhost:8001/internal/v1/feed \
     -H "Content-Type: application/json" \
     -H "X-Internal-Token: $INTERNAL_SERVICE_TOKEN" \
     -d "{\"user_id\":\"<id>\",\"limit\":10}"'
#    → {"items":[...], "metadata":{...}}.  Without the header → 401.

# 4. Gateway integration (flag off → on)
#    REC_SERVICE_ENABLED=false  → GET /api/v1/reels/feed/personalized/<id> = heuristic.
#    REC_SERVICE_ENABLED=true   → same endpoint, same JSON shape, rec-ordered feed.

# 5. Fallback on outage
docker compose stop rec-service
#    GET /api/v1/reels/feed/personalized/<id> still returns 200 (heuristic fallback,
#    ~250ms budget, no user-visible error).
```

## 3. Build / type checks

```bash
cd backend && npm run build && npx tsc --noEmit
```
