"""Internal service authentication.

Every call from the Hono API gateway carries a shared secret in the
`X-Internal-Token` header. This guard rejects any request whose token does not
match `INTERNAL_SERVICE_TOKEN`. It is applied to internal endpoints only;
health probes stay unauthenticated so container/orchestrator checks work.

Fails CLOSED: if no token is configured, guarded calls are rejected (prevents
an accidentally unauthenticated service in a misconfiguration).
"""

import hmac

from fastapi import Header, HTTPException, status

from app.config import get_settings

settings = get_settings()

INTERNAL_TOKEN_HEADER = "X-Internal-Token"


def _constant_time_eq(a: str, b: str) -> bool:
    """Constant-time string comparison (avoids timing side channels)."""
    return hmac.compare_digest(a, b)


async def require_internal_token(
    x_internal_token: str | None = Header(default=None, alias=INTERNAL_TOKEN_HEADER),
) -> None:
    """FastAPI dependency: allow only gateway calls bearing the shared secret."""
    expected = settings.internal_service_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="internal token not configured",
        )
    if not x_internal_token or not _constant_time_eq(x_internal_token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid internal token",
        )
