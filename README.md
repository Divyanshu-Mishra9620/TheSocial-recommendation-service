# Recommendation Service

ML-based personalized reel feed for **The Social**. Built with FastAPI.

> **Internal service.** It is reachable only through the Hono API gateway on
> the internal network — never directly from the frontend. The existing
> heuristic recommender (`backend/src/lib/reelRecommendation.ts`) remains the
> fallback and is never removed.

## Status

**Batch 1 — skeleton.** Runnable FastAPI app with health probes, typed config,
and a Docker image. No ML logic, data clients, or gateway wiring yet — those
arrive in later batches.

## Endpoints

| Method | Path            | Purpose                                  |
|--------|-----------------|------------------------------------------|
| GET    | `/`             | Service metadata                         |
| GET    | `/health`       | Liveness probe                           |
| GET    | `/health/ready` | Readiness probe (deps stubbed for now)   |
| GET    | `/docs`         | Swagger UI (non-production only)         |

## Run locally

```bash
cd services/recommendation-service
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8001
# → http://127.0.0.1:8001/health
```

## Run with Docker

```bash
docker build -t the-social/recommendation-service .
docker run --rm -p 8001:8001 --env-file .env the-social/recommendation-service
```

In production the port is **not** published to the host; the service joins the
internal Compose network alongside the backend and shares its Redis/MongoDB.

## Configuration

All settings come from environment variables (see [`.env.example`](./.env.example)).
Field names map 1:1 to `app/config.py`.

## Layout

```
app/
├── __init__.py
├── main.py          # FastAPI app + lifespan
├── config.py        # pydantic-settings
└── api/
    ├── __init__.py
    └── health.py    # liveness + readiness
```
