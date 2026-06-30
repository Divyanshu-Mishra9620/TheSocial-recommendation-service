"""FastAPI application entrypoint for the Recommendation Service.

Batch 1 is a runnable skeleton: health probes and a metadata root only.
Infrastructure clients (Redis, MongoDB), the event consumer, candidate
generation, and ranking are introduced in later batches.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.health import router as health_router
from app.api.internal import router as internal_router
from app.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(settings.service_name)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Batch 1: nothing to connect yet. Downstream clients and the Redis
    # Stream consumer are started here in later batches.
    logger.info(
        "starting %s (env=%s, model=%s)",
        settings.service_name,
        settings.environment,
        settings.model_version,
    )
    yield
    logger.info("shutting down %s", settings.service_name)


app = FastAPI(
    title="The Social — Recommendation Service",
    version=__version__,
    description=(
        "ML-based personalized reel feed. Internal service — reachable only "
        "via the Hono API gateway, never directly from the frontend."
    ),
    lifespan=lifespan,
    # Disable interactive docs in production.
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url=None,
)

app.include_router(health_router)
app.include_router(internal_router)


@app.get("/", tags=["meta"])
async def root() -> dict:
    """Service metadata."""
    return {
        "service": settings.service_name,
        "version": app.version,
        "status": "ok",
        "docs": app.docs_url,
    }
