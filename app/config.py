"""Typed application configuration.

Settings are read from environment variables (case-insensitive) and, when
present, a local `.env` file. See `.env.example` for the full list. Values
are cached so the settings object is constructed once per process.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_version` would otherwise collide with pydantic's protected
        # "model_" namespace; disable that guard since we own these names.
        protected_namespaces=(),
    )

    # Runtime
    environment: str = "development"
    log_level: str = "info"
    service_name: str = "recommendation-service"
    host: str = "0.0.0.0"
    port: int = 8001

    # Shared infrastructure — declared now; clients wired in later batches.
    mongo_uri: str = ""
    redis_url: str = "redis://redis:6379"

    # Gateway trust — enforced in a later batch (X-Internal-Token guard).
    internal_service_token: str = ""

    # Model — the heuristic prior is the active scorer until XGBoost ships.
    model_version: str = "heuristic-v0"

    # Event ingestion (Redis Stream). The foundation is prepared but inert by
    # default: the consumer is a standalone worker (not auto-started by the API)
    # and exits immediately unless `event_stream_enabled` is true.
    event_stream_enabled: bool = False
    reel_events_stream: str = "reel:events"
    reel_events_group: str = "rec-ingest"
    reel_events_consumer: str = "rec-consumer-1"

    # MongoDB (event persistence). `mongo_db` falls back to the database in
    # MONGO_URI when empty. The collection matches Mongoose's pluralization of
    # the "ReelEvent" model ("reelevents").
    mongo_db: str = ""
    reel_events_collection: str = "reelevents"

    # Consumer tuning.
    consumer_block_ms: int = 5000
    consumer_batch_size: int = 50
    consumer_max_retries: int = 5
    consumer_claim_min_idle_ms: int = 60000

    # Feature store (Redis). CODE ONLY — not activated anywhere yet.
    feat_user_prefix: str = "rec:feat:user:"
    feat_reel_prefix: str = "rec:feat:reel:"
    feat_user_ttl: int = 3600  # 1 hour
    feat_reel_ttl: int = 600  # 10 minutes

    # Source collections for feature computation (Mongoose pluralized names).
    user_interactions_collection: str = "userreelinteractions"
    reels_collection: str = "reels"
    user_preferences_collection: str = "userreelpreferences"
    follows_collection: str = "follows"

    # Model registry + artifacts (offline training only; not used at serving yet).
    model_versions_collection: str = "rec_model_versions"
    model_artifacts_dir: str = "models_artifacts"

    # Scheduled retraining (offline worker; not started by the API).
    retrain_enabled: bool = False
    retrain_incremental_days: int = 1


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
