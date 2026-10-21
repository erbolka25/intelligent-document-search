import pytest

from document_search.api.auth import COOKIE_NAME
from document_search.api.ratelimit import RateLimiter
from tests.test_api import PASSWORD, make_client, upload

PROTECTED = [
    ("get", "/stats"),
    ("get", "/documents"),
    ("post", "/documents"),
    ("delete", "/documents"),
    ("delete", "/documents/0123456789abcdef"),
    ("post", "/search"),
    ("post", "/ask"),
    ("get", "/openapi.json"),
]


@pytest.fixture
def anon(tmp_path, embedder):
    return make_client(tmp_path, embedder, login=False)


@pytest.mark.parametrize("method, path", PROTECTED)
def test_api_requires_session(anon, method, path):
    resp = getattr(anon, method)(path)
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized" and resp.json()["message"]


def test_pages_redirect_to_login(anon):
    resp = anon.get("/?lang=ky", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/login?lang=ky"
    assert anon.get("/login").status_code == 200
    assert anon.get("/static/app.css").status_code == 200


@pytest.mark.parametrize("body", [{"password": "wrong"}, {"password": PASSWORD + " "}])
def test_wrong_password(anon, body):
    resp = anon.post("/auth/login", json=body)
    assert resp.status_code == 401 and resp.json()["error"] == "invalid_password"
    assert COOKIE_NAME not in resp.cookies
    assert anon.get("/documents").status_code == 401


@pytest.mark.parametrize("body", [{}, {"password": ""}])
def test_missing_password(anon, body):
    assert anon.post("/auth/login", json=body).status_code == 422


def test_login_sets_hardened_cookie_and_logout_revokes_it(anon):
    resp = anon.post("/auth/login", json={"password": PASSWORD})
    assert resp.status_code == 200
    cookie = resp.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "secure" not in cookie
    stolen = anon.cookies.get(COOKIE_NAME)
    assert anon.get("/documents").status_code == 200

    assert anon.post("/auth/logout").status_code == 200
    assert anon.get("/documents").status_code == 401
    anon.cookies.set(COOKIE_NAME, stolen)  # replaying the old cookie must not work
    assert anon.get("/documents").status_code == 401


def test_cookie_is_secure_behind_https_proxy(anon):
    resp = anon.post(
        "/auth/login", json={"password": PASSWORD}, headers={"X-Forwarded-Proto": "https"}
    )
    assert "secure" in resp.headers["set-cookie"].lower()


def test_forged_cookie_is_rejected(anon):
    anon.cookies.set(COOKIE_NAME, "forged.value.signature")
    assert anon.get("/documents").status_code == 401


def test_login_not_configured(tmp_path, embedder):
    client = make_client(tmp_path, embedder, login=False, app_password=None)
    resp = client.post("/auth/login", json={"password": "anything"})
    assert resp.status_code == 503 and resp.json()["error"] == "login_not_configured"


def test_login_rate_limit_per_ip(tmp_path, embedder):
    client = make_client(tmp_path, embedder, login=False, login_rate_per_min=3)
    for _ in range(3):
        assert client.post("/auth/login", json={"password": "x"}).status_code == 401
    resp = client.post("/auth/login", json={"password": PASSWORD})
    assert resp.status_code == 429
    assert resp.json()["error"] == "rate_limited" and int(resp.headers["Retry-After"]) >= 1


def test_ask_and_upload_rate_limits_per_session(tmp_path, embedder):
    client = make_client(tmp_path, embedder, ask_rate_per_min=2, upload_rate_per_min=1)
    assert upload(client).status_code == 201
    assert upload(client, "b.txt", b"Another document text.").status_code == 429
    for _ in range(2):
        assert client.post("/ask", json={"question": "отпуск"}).status_code == 200
    assert client.post("/ask", json={"question": "отпуск"}).status_code == 429


def test_rate_limiter_window_slides():
    now = [0.0]
    limiter = RateLimiter(2, window_s=10, clock=lambda: now[0])
    assert limiter.hit("a") == 0 and limiter.hit("a") == 0
    assert limiter.hit("a") == 10
    assert limiter.hit("b") == 0  # keys are independent
    now[0] = 10.5
    assert limiter.hit("a") == 0
