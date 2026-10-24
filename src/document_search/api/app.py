"""FastAPI application: login, upload, search, ask, health, and the demo page."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path, PurePath
from typing import Annotated

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.responses import JSONResponse as _JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from document_search.api.auth import SessionManager, client_ip
from document_search.api.ratelimit import RateLimiter
from document_search.api.schemas import (
    AskRequest,
    AskResponse,
    ChunkOut,
    DeleteResponse,
    DocumentOut,
    ErrorResponse,
    HealthResponse,
    LoginRequest,
    OkResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
    UploadResponse,
)
from document_search.config import Settings, get_settings
from document_search.i18n import request_language, translate
from document_search.ingestion import SUPPORTED_EXTENSIONS, IngestionError, file_extension
from document_search.service import (
    DocumentLimitError,
    EmbeddingError,
    RagService,
    build_service,
)

logger = logging.getLogger("document_search.api")

STATIC_DIR = Path(__file__).parent / "static"
# Pages are never cached, so after sign-out the browser asks the server (and gets redirected).
NO_STORE = {"Cache-Control": "no-store"}
_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w\-. ()]+")
_DOC_ID = re.compile(r"^[0-9a-f]{16}$")

# Reachable without a session: the login flow, the health probe and static assets.
PUBLIC_PATHS = {"/login", "/auth/login", "/auth/logout", "/health", "/favicon.ico"}
PUBLIC_PREFIXES = ("/static/",)

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
}


class JSONResponse(_JSONResponse):
    """JSON with an explicit charset: Windows PowerShell 5.1 otherwise decodes it as Latin-1."""

    media_type = "application/json; charset=utf-8"


class ApiError(Exception):
    """An expected failure with a stable code; the message is translated per request."""

    def __init__(self, status: int, code: str, **params: object) -> None:
        super().__init__(code)
        self.status, self.code, self.params = status, code, params


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra fields passed via `extra={"fields": {...}}`."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpx2", "urllib3", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def sanitize_filename(raw: str | None) -> str:
    """Keep only the base name and a safe character set; never used as a disk path."""
    name = PurePath((raw or "").replace("\\", "/")).name
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")
    stem, ext = PurePath(name).stem[:100], PurePath(name).suffix.lower()[:10]
    return f"{stem or 'document'}{ext}"


def check_content(data: bytes, ext: str) -> None:
    """Reject files whose bytes do not match their extension."""
    if ext == ".pdf" and b"%PDF-" not in data[:1024]:
        raise ApiError(400, "not_a_pdf")
    if ext == ".txt" and b"\x00" in data[:8192]:
        raise ApiError(400, "not_text")


def error_response(request: Request, status: int, code: str, **params: object) -> JSONResponse:
    lang = request_language(request)
    content: dict = {"error": code, "message": translate(lang, f"errors.{code}", **params)}
    headers = {}
    if code == "rate_limited":
        headers["Retry-After"] = str(params.get("seconds", 60))
    return JSONResponse(status_code=status, content=content, headers=headers)


def _is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


def create_app(service: RagService | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or (service.settings if service else get_settings())
    sessions = SessionManager(settings)
    login_limiter = RateLimiter(settings.login_rate_per_min)
    ask_limiter = RateLimiter(settings.ask_rate_per_min)
    upload_limiter = RateLimiter(settings.upload_rate_per_min)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            configure_logging(settings.log_level)
            app.state.service = build_service(settings)
            app.state.service.embedder.embed_query("warm-up")  # load the model before traffic
            logger.info("Service ready", extra={"fields": app.state.service.stats()})
        yield

    app = FastAPI(
        title="Document Search",
        version="0.2.0",
        lifespan=lifespan,
        default_response_class=JSONResponse,
        responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
    )
    app.state.service = service
    app.state.sessions = sessions
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def get_service(request: Request) -> RagService:
        return request.app.state.service

    def limit(limiter: RateLimiter, key: str) -> None:
        wait = limiter.hit(key)
        if wait:
            raise ApiError(429, "rate_limited", seconds=wait)

    # --- middleware (the last one registered runs first) ----------------------
    @app.middleware("http")
    async def require_session(request: Request, call_next):
        path = request.url.path
        session_id = sessions.session_id(request)
        request.state.session_id = session_id
        if session_id is None and not _is_public(path):
            if request.method == "GET" and path in ("/", "/docs", "/redoc"):
                lang = request.query_params.get("lang")
                return RedirectResponse(f"/login?lang={lang}" if lang else "/login", 303)
            return error_response(request, 401, "unauthorized")
        return await call_next(request)

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        if request.url.path == "/documents" and request.method == "POST":
            length = request.headers.get("content-length")
            # Reject oversized uploads before the body is read (64 KB multipart overhead).
            if length and length.isdigit() and int(length) > settings.max_upload_bytes + 65536:
                response = error_response(
                    request, 413, "file_too_large", max_mb=settings.max_upload_mb
                )
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        for name, value in SECURITY_HEADERS.items():
            if name == "Content-Security-Policy" and request.url.path in ("/docs", "/redoc"):
                continue  # Swagger UI loads its assets from a CDN
            response.headers.setdefault(name, value)
        logger.info(
            "request",
            extra={
                "fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                }
            },
        )
        return response

    # --- exception handlers ---------------------------------------------------
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return error_response(request, exc.status, exc.code, **exc.params)

    @app.exception_handler(IngestionError)
    async def ingestion_error(request: Request, exc: IngestionError):
        return error_response(request, 400, exc.code)

    @app.exception_handler(DocumentLimitError)
    async def document_limit(request: Request, exc: DocumentLimitError):
        return error_response(
            request, 409, "too_many_documents", max_documents=settings.max_documents
        )

    @app.exception_handler(EmbeddingError)
    async def embedding_unavailable(request: Request, exc: EmbeddingError):
        # /ask degrades on its own; /search and uploads have nothing to fall back to.
        return error_response(request, 503, "embedding_unavailable")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        response = error_response(request, 422, "validation_error")
        fields = sorted({".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors()})
        body = json.loads(response.body)
        body["fields"] = fields
        return JSONResponse(status_code=422, content=body)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        # e.g. a body that is not valid UTF-8 JSON: the client's fault, not ours.
        fallback = "bad_request" if 400 <= exc.status_code < 500 else "internal_error"
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, fallback)
        return error_response(
            request, exc.status_code, code, request_id=getattr(request.state, "request_id", "")
        )

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled error", extra={"fields": {"request_id": request_id}})
        return error_response(request, 500, "internal_error", request_id=request_id)

    # --- pages and auth -----------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def index_page():
        return FileResponse(STATIC_DIR / "index.html", headers=NO_STORE)

    @app.get("/login", include_in_schema=False)
    async def login_page(request: Request):
        if request.state.session_id:
            return RedirectResponse("/", 303)
        return FileResponse(STATIC_DIR / "login.html", headers=NO_STORE)

    @app.post("/auth/login", response_model=OkResponse)
    def login(request: Request, body: LoginRequest):
        limit(login_limiter, client_ip(request))
        if not sessions.configured:
            raise ApiError(503, "login_not_configured")
        if not sessions.check_password(body.password):
            logger.warning("Failed login", extra={"fields": {"ip": client_ip(request)}})
            raise ApiError(401, "invalid_password")
        response = JSONResponse({"ok": True})
        sessions.set_cookie(response, request, sessions.create())
        return response

    @app.post("/auth/logout", response_model=OkResponse)
    def logout(request: Request):
        sessions.revoke(request)
        response = JSONResponse({"ok": True})
        sessions.clear_cookie(response, request)
        return response

    # --- API ------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok")

    @app.get("/stats", response_model=StatsResponse)
    def stats(request: Request):
        return StatsResponse(**get_service(request).stats(), max_documents=settings.max_documents)

    @app.get("/documents", response_model=list[DocumentOut])
    def list_documents(request: Request):
        return get_service(request).documents()

    @app.post("/documents", response_model=UploadResponse, status_code=201)
    def upload_document(request: Request, file: Annotated[UploadFile, File()]):
        limit(upload_limiter, request.state.session_id)
        filename = sanitize_filename(file.filename)
        ext = file_extension(filename)
        if ext not in SUPPORTED_EXTENSIONS:
            raise ApiError(415, "unsupported_type")
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise ApiError(413, "file_too_large", max_mb=settings.max_upload_mb)
        if not data:
            raise ApiError(400, "empty_file")
        check_content(data, ext)
        result = get_service(request).add_document(data, filename)
        return UploadResponse(**vars(result))

    @app.delete("/documents/{doc_id}", response_model=DeleteResponse)
    def delete_document(request: Request, doc_id: str):
        limit(upload_limiter, request.state.session_id)
        if not _DOC_ID.match(doc_id) or not get_service(request).delete_document(doc_id):
            raise ApiError(404, "not_found")
        return DeleteResponse(deleted=1)

    @app.delete("/documents", response_model=DeleteResponse)
    def clear_documents(request: Request):
        limit(upload_limiter, request.state.session_id)
        return DeleteResponse(deleted=get_service(request).clear_documents())

    @app.post("/search", response_model=SearchResponse)
    def search(request: Request, body: SearchRequest):
        limit(ask_limiter, request.state.session_id)
        results = get_service(request).search(body.query, body.top_k)
        return SearchResponse(query=body.query, results=[ChunkOut.from_result(r) for r in results])

    @app.post("/ask", response_model=AskResponse)
    def ask(request: Request, body: AskRequest):
        limit(ask_limiter, request.state.session_id)
        result = get_service(request).ask(body.question, body.top_k)
        lang = request_language(request)
        return AskResponse(
            question=body.question,
            mode=result.mode,
            answer=result.answer,
            answer_found=result.answer_found,
            citations=[ChunkOut.from_result(r) for r in result.citations],
            results=[ChunkOut.from_result(r) for r in result.results],
            notice=result.notice,
            notice_message=translate(lang, f"notices.{result.notice}") if result.notice else None,
        )

    return app


# Entry point for `uvicorn document_search.api.app:app`; heavy setup happens in lifespan.
app = create_app()
