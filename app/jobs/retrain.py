"""Scheduled retraining worker (offline).

Runs the training pipeline on a schedule via APScheduler:
  - nightly incremental retraining (recent window),
  - weekly full retraining (all data).

Each run: build dataset → train → evaluate → register as a CANDIDATE. Models are
NEVER auto-promoted or activated — status stays "candidate". On metric regression
versus the current reference model, a warning is logged (the candidate is still
recorded, never promoted).

Runs as a standalone worker; the FastAPI serving app does NOT import or start it,
so serving behavior is unchanged:

    python -m app.jobs.retrain                    # start the scheduler
    python -m app.jobs.retrain --once incremental # run one job and exit

Requires the ML stack: pip install -r requirements-train.txt
The scheduler is inert unless RETRAIN_ENABLED is true.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(f"{settings.service_name}.retrain")

# Warn if the new model's AUC is worse than the reference by more than this.
_AUC_REGRESSION_TOLERANCE = 0.02


def is_regression(
    new_auc: float,
    reference_auc: float,
    tolerance: float = _AUC_REGRESSION_TOLERANCE,
) -> bool:
    """True when the new AUC regresses beyond tolerance vs. the reference. Pure."""
    return new_auc + tolerance < reference_auc


async def get_reference_model() -> dict[str, Any] | None:
    """The active model if one exists, else the most recently trained version."""
    from app.core.mongo import get_collection

    collection = get_collection(settings.model_versions_collection)
    active = await collection.find_one({"status": "active"})
    if active is not None:
        return active
    return await collection.find_one(sort=[("trained_at", -1)])


async def _warn_on_regression(new_metrics: dict[str, Any]) -> None:
    reference = await get_reference_model()
    if not reference:
        return
    ref_auc = float(reference.get("metrics", {}).get("auc", 0.0))
    new_auc = float(new_metrics.get("auc", 0.0))
    if is_regression(new_auc, ref_auc):
        logger.warning(
            "metric regression: new AUC %.4f < reference AUC %.4f; candidate "
            "registered but NOT promoted",
            new_auc,
            ref_auc,
        )


async def run_retraining(mode: str) -> dict[str, Any] | None:
    """Build → train → evaluate → register a CANDIDATE. Returns the document, or
    None when there is not enough data to train."""
    from training.build_dataset import FEATURE_COLUMNS, build_dataset
    from training.evaluate import compute_metrics
    from training.registry import (
        build_version_document,
        register_model,
        save_artifact,
    )
    from training.train import _new_version, train_model

    since = None
    if mode == "incremental":
        since = datetime.now(timezone.utc) - timedelta(
            days=settings.retrain_incremental_days
        )

    train_df, val_df = await build_dataset(since=since)
    if len(train_df) == 0 or len(val_df) == 0:
        logger.warning(
            "retraining (%s) skipped: insufficient data (train=%d val=%d)",
            mode,
            len(train_df),
            len(val_df),
        )
        return None

    model, importance = train_model(train_df, val_df)
    proba = model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    metrics = compute_metrics(
        val_df["label"].tolist(),
        [float(p) for p in proba],
        val_df["user_id"].tolist(),
    )
    metrics["feature_importance"] = importance

    await _warn_on_regression(metrics)

    version = _new_version()
    artifact_path = save_artifact(model, version, settings.model_artifacts_dir)
    document = build_version_document(
        version=version,
        metrics=metrics,
        artifact_path=artifact_path,
        feature_columns=FEATURE_COLUMNS,
        status="candidate",  # never auto-promoted / activated
    )
    await register_model(document)
    logger.info(
        "registered candidate %s (mode=%s, auc=%.4f)",
        version,
        mode,
        metrics.get("auc", 0.0),
    )
    return document


async def _serve() -> None:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        run_retraining,
        CronTrigger(hour=3, minute=0),
        args=["incremental"],
        id="nightly_incremental",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    scheduler.add_job(
        run_retraining,
        CronTrigger(day_of_week="sun", hour=4, minute=0),
        args=["full"],
        id="weekly_full",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    scheduler.start()
    logger.info(
        "retraining scheduler started (incremental nightly 03:00 UTC, full Sun 04:00 UTC)"
    )
    try:
        import asyncio

        await asyncio.Event().wait()  # run until cancelled
    finally:
        scheduler.shutdown()


def main() -> None:
    import argparse
    import asyncio

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO)
    )

    parser = argparse.ArgumentParser(description="Recommendation retraining worker")
    parser.add_argument("--once", choices=["incremental", "full"], default=None)
    args = parser.parse_args()

    if args.once:
        asyncio.run(run_retraining(args.once))
        return

    if not settings.retrain_enabled:
        logger.info("retraining scheduler disabled (RETRAIN_ENABLED=false); exiting")
        return

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
