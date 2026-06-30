"""Internal (gateway-only) endpoints.

Reachable only by the Hono API gateway, which must present the shared
`X-Internal-Token`. These endpoints are internal-only and are NOT called by
anything yet — no Hono / aiServiceClient / gateway / reelController / frontend
integration. The service publishes no host port (Compose-internal network only).
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.security import require_internal_token
from app.ranking import rerank_feed

router = APIRouter(prefix="/internal/v1", tags=["internal"])
settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.internal")


class FeedRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    limit: int = Field(default=20, ge=1, le=500)
    seen: list[str] = Field(default_factory=list)


class FeedResponse(BaseModel):
    items: list[str]
    metadata: dict[str, Any]


class ModelInfoResponse(BaseModel):
    version: str | None = None
    status: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    trained_at: str | None = None


@router.get("/ping", dependencies=[Depends(require_internal_token)])
async def ping() -> dict:
    """Authenticated reachability check for the gateway."""
    return {"status": "ok", "service": settings.service_name}


@router.post(
    "/feed",
    response_model=FeedResponse,
    dependencies=[Depends(require_internal_token)],
)
async def feed(request: FeedRequest) -> FeedResponse:
    """Internal recommendation feed.

    Runs candidate generation → ranking → diversified reranking and returns the
    final ordered reel IDs. Internal-only; not yet called by the gateway.
    """
    try:
        result = await rerank_feed(
            request.user_id, limit=request.limit, seen=request.seen
        )
    except Exception:
        logger.exception("feed generation failed for user %s", request.user_id)
        raise HTTPException(status_code=500, detail="feed generation failed")

    return FeedResponse(
        items=result.get("final_reel_ids", []),
        metadata=result.get("metadata", {}),
    )


@router.get(
    "/model/active",
    response_model=ModelInfoResponse,
    dependencies=[Depends(require_internal_token)],
)
async def model_active() -> ModelInfoResponse:
    """Observability: the active model version, else the most recent candidate.

    Read-only; this does NOT activate or change the serving scorer.
    """
    from app.core.mongo import get_collection

    collection = get_collection(settings.model_versions_collection)
    doc = await collection.find_one({"status": "active"})
    if doc is None:
        doc = await collection.find_one(sort=[("trained_at", -1)])
    if doc is None:
        return ModelInfoResponse()
    return ModelInfoResponse(
        version=doc.get("version"),
        status=doc.get("status"),
        metrics=doc.get("metrics", {}),
        trained_at=doc.get("trained_at"),
    )
