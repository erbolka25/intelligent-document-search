import pytest
from fastapi.testclient import TestClient

from document_search.api.app import create_app, sanitize_filename
from document_search.config import Settings
from document_search.generation import LLMError
from document_search.retrieval import create_index
from document_search.service import RagService
from tests.test_generation import FakeLLM

PASSWORD = "correct horse battery staple"

HR_TEXT = (
    "Ежегодный оплачиваемый отпуск составляет 28 календарных дней. "
    "Пароль от корпоративной почты нужно менять каждые 90 дней. "
    "Командировочные расходы возмещаются по авансовому отчёту."
)


def make_client(tmp_path, embedder, llm=None, *, login=True, **overrides) -> TestClient:
    values = {"app_password": PASSWORD, "session_secret": "test-secret", **overrides}
    settings = Settings(data_dir=tmp_path, index_backend="numpy", _env_file=None, **values)
    service = RagService(settings, embedder, create_index("numpy"), llm)
    client = TestClient(create_app(service), raise_server_exceptions=False)
    if login:
        assert client.post("/auth/login", json={"password": PASSWORD}).status_code == 200
    return client


@pytest.fixture
def client(tmp_path, embedder):
    return make_client(tmp_path, embedder)


def upload(client, name="hr.txt", data: bytes | None = None):
    data = HR_TEXT.encode() if data is None else data
    return client.post("/documents", files={"file": (name, data, "application/octet-stream")})


def test_health_is_public_and_minimal(tmp_path, embedder):
    resp = make_client(tmp_path, embedder, login=False).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert "X-Request-ID" in resp.headers
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in resp.headers["Content-Security-Policy"]


def test_stats_on_empty_index(client):
    assert client.get("/stats").json() == {
        "documents": 0,
        "max_documents": 50,
        "chunks": 0,
        "index_backend": "numpy",
        "llm_enabled": False,
        "llm_provider": None,
        "llm_model": None,
    }


def test_upload_search_and_list(client):
    resp = upload(client)
    assert resp.status_code == 201
    body = resp.json()
    assert body["filename"] == "hr.txt" and body["chunks"] >= 1 and not body["duplicate"]
    assert body["skipped_pages"] == []

    resp = client.post("/search", json={"query": "пароль от почты", "top_k": 3})
    assert resp.status_code == 200
    top = resp.json()["results"][0]
    assert top["filename"] == "hr.txt" and top["rank"] == 1
    assert "Пароль" in top["text"]
    assert {"score", "doc_id", "page", "chunk_index", "char_start", "char_end"} <= top.keys()

    assert client.get("/documents").json() == [
        {"doc_id": body["doc_id"], "filename": "hr.txt", "chunks": body["chunks"]}
    ]


def test_upload_pdf(client, make_pdf):
    resp = upload(client, "report.pdf", make_pdf(["Quarterly revenue grew by ten percent."]))
    assert resp.status_code == 201
    hit = client.post("/search", json={"query": "revenue"}).json()["results"][0]
    assert hit["filename"] == "report.pdf" and hit["page"] == 1


def test_upload_cyrillic_pdf_reports_skipped_pages(client, make_unicode_pdf):
    data = make_unicode_pdf(
        ["Отпуск составляет 28 календарных дней.", "IIIII IIII IIIIIII IIIII IIII"]
    )
    body = upload(client, "политика.pdf", data).json()
    assert body["skipped_pages"] == [2]
    hit = client.post("/search", json={"query": "отпуск"}).json()["results"][0]
    assert hit["text"] == "Отпуск составляет 28 календарных дней." and hit["page"] == 1


def test_duplicate_upload(client):
    first = upload(client).json()
    second = upload(client, "copy.txt").json()
    assert second["duplicate"] and second["doc_id"] == first["doc_id"]
    assert client.get("/stats").json()["documents"] == 1


def test_search_on_empty_index_returns_empty_list(client):
    resp = client.post("/search", json={"query": "что угодно"})
    assert resp.status_code == 200 and resp.json()["results"] == []


def test_ask_retrieval_only_without_llm(client):
    upload(client)
    body = client.post("/ask", json={"question": "Сколько дней отпуска?"}).json()
    assert body["mode"] == "retrieval_only"
    assert body["answer"] is None
    assert body["results"] and body["notice"] == "llm_disabled"
    assert body["notice_message"] and "API_KEY" not in body["notice_message"]


def test_ask_with_mocked_llm(tmp_path, embedder):
    client = make_client(tmp_path, embedder, FakeLLM("Отпуск составляет 28 дней [1]."))
    upload(client)
    body = client.post("/ask", json={"question": "Сколько дней отпуска?"}).json()
    assert body["mode"] == "rag"
    assert body["answer"] == "Отпуск составляет 28 дней [1]."
    assert body["citations"][0]["filename"] == "hr.txt"
    assert body["citations"][0]["rank"] == 1
    assert body["notice"] is None


@pytest.mark.parametrize("code", ["auth", "quota", "rate_limit", "network", "api"])
def test_ask_llm_failure_falls_back_with_friendly_message(tmp_path, embedder, code):
    llm = FakeLLM(error=LLMError(code, "secret detail sk-123 OPENAI_API_KEY"))
    client = make_client(tmp_path, embedder, llm)
    upload(client)
    resp = client.post("/ask", json={"question": "Сколько дней отпуска?"})
    body = resp.json()
    assert resp.status_code == 200
    assert body["mode"] == "retrieval_only" and body["results"]
    assert body["notice"] == code
    assert "sk-123" not in resp.text and "API_KEY" not in resp.text


def test_ask_no_relevant_result(tmp_path, embedder):
    client = make_client(tmp_path, embedder, FakeLLM("should not be called"), min_score=0.5)
    upload(client)
    body = client.post("/ask", json={"question": "квантовая хромодинамика"}).json()
    assert body["results"] == [] and body["answer_found"] is False


@pytest.mark.parametrize(
    "name, data, status, error",
    [
        ("empty.txt", b"", 400, "empty_file"),
        ("blank.txt", b"   \n  ", 400, "no_text"),
        ("virus.exe", b"MZ...", 415, "unsupported_type"),
        ("noext", b"text", 415, "unsupported_type"),
        ("script.pdf.exe", b"%PDF-1.4", 415, "unsupported_type"),
        ("fake.pdf", b"MZ not a pdf at all", 400, "not_a_pdf"),
        ("broken.pdf", b"%PDF-1.4 garbage", 400, "corrupted_pdf"),
        ("binary.txt", b"abc\x00\x01\x02", 400, "not_text"),
    ],
)
def test_upload_invalid(client, name, data, status, error):
    resp = upload(client, name, data)
    assert resp.status_code == status
    body = resp.json()
    assert body["error"] == error and body["message"]


def test_upload_too_large(tmp_path, embedder):
    client = make_client(tmp_path, embedder, max_upload_mb=1)
    resp = upload(client, "big.txt", b"a" * (1024 * 1024 + 1))
    assert resp.status_code == 413 and resp.json()["error"] == "file_too_large"


def test_document_limit(tmp_path, embedder):
    client = make_client(tmp_path, embedder, max_documents=1)
    assert upload(client).status_code == 201
    resp = upload(client, "other.txt", b"Completely different text about travel.")
    assert resp.status_code == 409 and resp.json()["error"] == "too_many_documents"


def test_upload_without_file(client):
    resp = client.post("/documents")
    assert resp.status_code == 422
    assert resp.json()["error"] == "validation_error" and resp.json()["fields"] == ["file"]


@pytest.mark.parametrize(
    "path, payload",
    [
        ("/search", {"query": ""}),
        ("/search", {"query": "   "}),
        ("/search", {}),
        ("/search", {"query": "ok", "top_k": 0}),
        ("/ask", {"question": ""}),
        ("/ask", {"question": "   "}),
        ("/ask", {"question": "x" * 2001}),
    ],
)
def test_invalid_queries(client, path, payload):
    resp = client.post(path, json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "validation_error" and body["message"] and body["fields"]


def test_internal_error_hides_details(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret path C:/internal/stuff")

    monkeypatch.setattr(client.app.state.service, "search", boom)
    resp = client.post("/search", json={"query": "x"})
    assert resp.status_code == 500
    assert resp.json()["error"] == "internal_error"
    assert "secret" not in resp.text and "Traceback" not in resp.text


def test_unknown_route_is_json_404(client):
    resp = client.get("/nope")
    assert resp.status_code == 404 and resp.json()["error"] == "not_found"


def test_delete_document(client):
    doc_id = upload(client).json()["doc_id"]
    other = upload(client, "it.txt", b"Password rotation every ninety days.").json()["doc_id"]
    assert client.delete(f"/documents/{doc_id}").json() == {"deleted": 1}
    assert [d["doc_id"] for d in client.get("/documents").json()] == [other]
    hits = client.post("/search", json={"query": "отпуск", "top_k": 10}).json()["results"]
    assert all(h["doc_id"] == other for h in hits)
    assert client.delete(f"/documents/{doc_id}").status_code == 404


@pytest.mark.parametrize("doc_id", ["..%2F..%2Fetc", "zzzzzzzzzzzzzzzz", "abc"])
def test_delete_rejects_bad_ids(client, doc_id):
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_clear_documents(client, tmp_path):
    upload(client)
    upload(client, "it.txt", b"Password rotation every ninety days.")
    assert client.delete("/documents").json() == {"deleted": 2}
    assert client.get("/documents").json() == []
    assert list((tmp_path / "uploads").iterdir()) == []
    assert client.get("/stats").json()["chunks"] == 0


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("..\\..\\windows\\evil.TXT", "evil.txt"),
        ("Отчёт за 2026 (финал).pdf", "Отчёт за 2026 (финал).pdf"),
        ("a<b>|c?.txt", "a_b_c_.txt"),
        ("", "document"),
        ("....", "document"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_uploaded_file_stored_under_content_hash(client, tmp_path):
    doc_id = upload(client, "../../escape.txt").json()["doc_id"]
    stored = list((tmp_path / "uploads").iterdir())
    assert [p.name for p in stored] == [f"{doc_id}.txt"]
