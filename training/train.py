"""XGBoost ranker training (offline).

Trains an XGBClassifier (P(engage)) on the time-split dataset with early
stopping and class balancing, exports feature importance, evaluates on the
validation split, and registers the resulting model as a CANDIDATE version.

OFFLINE ONLY. This does NOT replace the serving heuristic prior
(`score_features`); activation is a separate, future step. Heavy deps are
imported lazily. Run with the ML stack installed:

    pip install -r requirements-train.txt
    python -m training.build_dataset      # writes train/val parquet
    python -m training.train              # trains + registers a candidate
"""

from __future__ import annotations

from typing import Any

from training.build_dataset import FEATURE_COLUMNS

_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 20,
}


def train_model(train_df, val_df, params: dict[str, Any] | None = None):
    """Fit an XGBClassifier with early stopping + class balancing.

    Returns (model, feature_importance). `train_df`/`val_df` are pandas frames
    with FEATURE_COLUMNS + a "label" column.
    """
    import xgboost as xgb

    cfg = {**_DEFAULT_PARAMS, **(params or {})}

    x_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    x_val, y_val = val_df[FEATURE_COLUMNS], val_df["label"]

    # Class balancing for skewed positives.
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = (negatives / positives) if positives > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=cfg["early_stopping_rounds"],
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    importance = dict(zip(FEATURE_COLUMNS, [float(v) for v in model.feature_importances_]))
    return model, importance


def _new_version() -> str:
    from datetime import datetime, timezone

    return "reel_ranker_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main() -> None:
    import asyncio
    import os

    import pandas as pd

    from app.config import get_settings
    from training.evaluate import compute_metrics
    from training.registry import (
        build_version_document,
        register_model,
        save_artifact,
    )

    settings = get_settings()
    artifacts_dir = settings.model_artifacts_dir
    train_df = pd.read_parquet(os.path.join(artifacts_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(artifacts_dir, "val.parquet"))

    model, importance = train_model(train_df, val_df)

    proba = model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    metrics = compute_metrics(
        val_df["label"].tolist(),
        [float(p) for p in proba],
        val_df["user_id"].tolist(),
    )
    metrics["feature_importance"] = importance

    version = _new_version()
    artifact_path = save_artifact(model, version, artifacts_dir)
    document = build_version_document(
        version=version,
        metrics=metrics,
        artifact_path=artifact_path,
        feature_columns=FEATURE_COLUMNS,
        status="candidate",
    )
    asyncio.run(register_model(document))
    print(f"registered candidate {version}: {metrics}")


if __name__ == "__main__":
    main()
