import hmac
import time
from collections import defaultdict, deque

from fastapi import Header, Request

from .config import Settings
from .errors import ServiceError
from .metrics import RATE_LIMITED


def _bearer(value: str | None, expected: str) -> None:
    if not value or not value.startswith("Bearer ") or not hmac.compare_digest(value[7:], expected):
        raise ServiceError(401, "unauthorized", "invalid or missing bearer token")


class Security:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    async def service_auth(self, authorization: str | None = Header(None)) -> None:
        _bearer(authorization, self.settings.api_key.get_secret_value())

    async def admin_auth(self, authorization: str | None = Header(None)) -> None:
        _bearer(authorization, self.settings.admin_token.get_secret_value())

    async def rate_limit(self, request: Request) -> None:
        key = request.client.host if request.client else "unknown"
        now, window = time.monotonic(), self._requests[key]
        while window and window[0] < now - 60:
            window.popleft()
        if len(window) >= self.settings.rate_limit_per_minute:
            RATE_LIMITED.inc()
            raise ServiceError(429, "rate_limited", "rate limit exceeded")
        window.append(now)
