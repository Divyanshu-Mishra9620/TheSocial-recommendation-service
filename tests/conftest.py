"""Pytest bootstrap.

Sets the environment BEFORE any app module is imported, so the cached Settings
(and the security guard's captured token) see test values. No real Redis/Mongo
connections are opened — clients are lazy and the tests mock all I/O.
"""

import os

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("INTERNAL_SERVICE_TOKEN", "test-internal-token")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017/rec-test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
