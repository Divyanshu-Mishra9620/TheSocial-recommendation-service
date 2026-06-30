"""Model registry: persist RecModelVersion records + save artifacts.

OFFLINE ONLY. Registering a model does NOT activate it — new models are stored
with status "candidate"; promotion to "active" (and serving) is a future step.
Mongo access is imported lazily; `build_version_document` is pure.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class RecModelVersion:
    version: str
    status: str  # "candidate" | "active" | "archived"
    metrics: dict[str, Any]
    trained_at: str  # ISO-8601
    artifact_path: str
    feature_columns: list[str]


def build_version_document(
    version: str,
    metrics: dict[str, Any],
    artifact_path: str,
    feature_columns: list[str],
    status: str = "candidate",
    trained_at: str | None = None,
) -> dict[str, Any]:
    """Build the registry document. Pure."""
    return asdict(
        RecModelVersion(
            version=version,
            status=status,
            metrics=metrics,
            trained_at=trained_at or datetime.now(timezone.utc).isoformat(),
            artifact_path=artifact_path,
            feature_columns=list(feature_columns),
        )
    )


def save_artifact(model, version: str, artifacts_dir: str) -> str:
    """Persist the trained model to models_artifacts/. Returns the path."""
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, f"{version}.json")
    model.save_model(path)
    return path


async def register_model(document: dict[str, Any]) -> dict[str, Any]:
    """Upsert a RecModelVersion document into the registry collection."""
    from app.config import get_settings
    from app.core.mongo import get_collection

    settings = get_settings()
    collection = get_collection(settings.model_versions_collection)
    await collection.update_one(
        {"version": document["version"]}, {"$set": document}, upsert=True
    )
    return document
