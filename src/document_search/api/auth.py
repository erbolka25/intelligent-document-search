"""Single-password login with a signed, server-tracked session cookie."""

from __future__ import annotations

import hmac
import logging
import secrets
import threading
import time

from itsdangerous import BadSignature, URLSafeTimedSerializer
from starlette.requests import Request
from starlette.responses import Response

from document_search.config import Settings

logger = logging.getLogger(__name__)

COOKIE_NAME = "ds_session"


class SessionManager:
    """Issues and checks session cookies.

    The cookie carries a random session id signed with SESSION_SECRET. Ids are also
    kept in memory, so logout revokes a session immediately; a restart logs everyone out.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._password = settings.app_password.get_secret_value() if settings.app_password else ""
        secret = settings.session_secret.get_secret_value() if settings.session_secret else ""
        if not secret:
            secret = secrets.token_urlsafe(32)
            logger.warning("SESSION_SECRET not set: using a random one, sessions end on restart")
        if not self._password:
            logger.warning("APP_PASSWORD not set: nobody can log in until it is configured")
        self._serializer = URLSafeTimedSerializer(secret, salt="document-search-session")
        self._sessions: dict[str, float] = {}  # session id -> expiry (epoch seconds)
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self._password)

    def check_password(self, candidate: str) -> bool:
        if not self._password:
            return False
        # Constant-time comparison on bytes, so timing does not leak the password.
        return hmac.compare_digest(candidate.encode(), self._password.encode())

    def create(self) -> str:
        session_id = secrets.token_urlsafe(24)
        with self._lock:
            now = time.time()
            self._sessions = {k: v for k, v in self._sessions.items() if v > now}
            self._sessions[session_id] = now + self.settings.session_max_age_s
        return self._serializer.dumps(session_id)

    def session_id(self, request: Request) -> str | None:
        """The valid session id behind the request's cookie, or None."""
        token = request.cookies.get(COOKIE_NAME)
        if not token:
            return None
        try:
            session_id = self._serializer.loads(token, max_age=self.settings.session_max_age_s)
        except BadSignature:  # includes expired signatures
            return None
        with self._lock:
            expiry = self._sessions.get(session_id)
        return session_id if expiry and expiry > time.time() else None

    def revoke(self, request: Request) -> None:
        token = request.cookies.get(COOKIE_NAME)
        if not token:
            return
        try:
            session_id = self._serializer.loads(token)
        except BadSignature:
            return
        with self._lock:
            self._sessions.pop(session_id, None)

    def _secure(self, request: Request) -> bool:
        mode = self.settings.cookie_secure
        if mode == "auto":
            forwarded = request.headers.get("x-forwarded-proto", "")
            return request.url.scheme == "https" or forwarded.split(",")[0].strip() == "https"
        return mode == "true"

    def set_cookie(self, response: Response, request: Request, token: str) -> None:
        response.set_cookie(
            COOKIE_NAME,
            token,
            max_age=self.settings.session_max_age_s,
            httponly=True,
            samesite="lax",
            secure=self._secure(request),
            path="/",
        )

    def clear_cookie(self, response: Response, request: Request) -> None:
        response.delete_cookie(
            COOKIE_NAME, path="/", httponly=True, samesite="lax", secure=self._secure(request)
        )


def client_ip(request: Request) -> str:
    # X-Forwarded-For is client-controlled, so it is not trusted; behind a reverse proxy,
    # run uvicorn with --proxy-headers so request.client is the real address.
    return request.client.host if request.client else "unknown"
