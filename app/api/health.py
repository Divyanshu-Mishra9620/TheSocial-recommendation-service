"""Health and readiness probes.

`/health`        — liveness: the process is up and serving HTTP.
`/health/ready`  — readiness: safe to receive traffic.

In this skeleton batch there are no downstream clients, so readiness reports
the app as ready and explicitly marks Redis/Mongo as `not_wired`. Real
dependency checks are added when those clients are introduced.
"""

from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health")
async def health() -> dict:
    """Liveness probe — process is up and serving."""
    return {
        "status": "ok",
        "service": settings.service_name,
        "model_version": settings.model_version,
    }


@router.get("/health/ready")
async def readiness() -> dict:
    """Readiness probe.

    Batch 1 skeleton: no downstream clients are wired yet, so this reflects
    only that the application initialized. Dependency checks (Redis, Mongo)
    are added in later batches.
    """
    return {
        "status": "ready",
        "service": settings.service_name,
        "checks": {
            "app": "ok",
            "redis": "not_wired",
            "mongo": "not_wired",
        },
    }
