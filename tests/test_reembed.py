"""Index metadata and re-embedding when the configured embedding model changes."""

import json

import numpy as np
import pytest

from document_search.config import Settings
from document_search.retrieval import load_index
from document_search.retrieval.index import META_FILE
from document_search.service import EmbeddingError, RagService
from tests.conftest import HashingEmbedder

BACKENDS = ["numpy", pytest.param("faiss", marks=pytest.mark.faiss)]

HR = "Отпуск составляет 28 календарных дней. Пароль меняется раз в 90 дней.".encode()
TRAVEL = "Командировочные расходы возмещаются по авансовому отчёту.".encode()


class CountingEmbedder(HashingEmbedder):
    """Counts embed_documents calls; ``fail`` makes them raise."""

    def __init__(self, dim: int = 256, fail: bool = False) -> None:
        super().__init__(dim)
        self.document_calls = 0
        self.fail = fail

    def embed_documents(self, texts):
        self.document_calls += 1
        if self.fail:
            raise RuntimeError("model unavailable")
        return super().embed_documents(texts)


class PrefixedEmbedder(HashingEmbedder):
    """Same model and dimension as HashingEmbedder(256), different prefixes."""

    @property
    def identity(self):
        return {**super().identity, "query_prefix": "query: ", "passage_prefix": "passage: "}


@pytest.fixture(params=BACKENDS)
def backend(request):
    if request.param == "faiss":
        pytest.importorskip("faiss")
    return request.param


def _settings(tmp_path, backend) -> Settings:
    return Settings(data_dir=tmp_path, index_backend=backend, min_score=0.0, _env_file=None)


def _build_index(tmp_path, backend, embedder) -> Settings:
    """Index two documents with ``embedder`` and save them to disk."""
    settings = _settings(tmp_path, backend)
    service = RagService(settings, embedder, load_index(backend, settings.index_dir))
    service.add_document(HR, "hr.txt")
    service.add_document(TRAVEL, "travel.txt")
    return settings


def _reopen(settings, embedder) -> RagService:
    """What build_service does at startup: load the saved index, then wire the service."""
    return RagService(settings, embedder, load_index(settings.index_backend, settings.index_dir))


def _meta(settings) -> dict:
    return json.loads((settings.index_dir / META_FILE).read_text(encoding="utf-8"))


def test_saved_index_records_the_embedder(tmp_path, backend):
    settings = _build_index(tmp_path, backend, HashingEmbedder(256))
    assert _meta(settings) == {
        "embedding_model": "hashing-256",
        "dimension": 256,
        "query_prefix": "",
        "passage_prefix": "",
    }


def test_model_change_re_embeds_and_saves(tmp_path, backend, caplog):
    settings = _build_index(tmp_path, backend, HashingEmbedder(256))
    chunks = len(load_index(backend, settings.index_dir))
    new = CountingEmbedder(768)

    with caplog.at_level("WARNING", logger="document_search.service"):
        service = _reopen(settings, new)

    assert new.document_calls == 1
    assert f"re-embedding {chunks} chunks" in caplog.text
    assert service.index.dim == 768
    assert len(service.index) == chunks
    assert _meta(settings)["embedding_model"] == "hashing-768"
    assert _meta(settings)["dimension"] == 768
    # The re-embedded index was saved: a restart with the same model loads it as is.
    again = CountingEmbedder(768)
    restarted = _reopen(settings, again)
    assert again.document_calls == 0
    assert restarted.index.dim == 768


def test_search_and_upload_work_after_re_embed(tmp_path, backend):
    settings = _build_index(tmp_path, backend, HashingEmbedder(256))
    service = _reopen(settings, HashingEmbedder(768))

    results = service.search("Сколько дней отпуск?")
    assert results and results[0].chunk.filename == "hr.txt"
    # New uploads use the same model, so they join the re-embedded index.
    service.add_document("Офис открыт с 9 до 18 часов.".encode(), "office.txt")
    assert service.search("Когда открыт офис?")[0].chunk.filename == "office.txt"


def test_matching_metadata_does_not_re_embed(tmp_path, backend):
    settings = _build_index(tmp_path, backend, HashingEmbedder(256))
    before = np.load(settings.index_dir / "vectors.npy")
    same = CountingEmbedder(256)

    service = _reopen(settings, same)

    assert same.document_calls == 0
    assert service.index.dim == 256
    np.testing.assert_array_equal(np.load(settings.index_dir / "vectors.npy"), before)


def test_index_without_metadata_is_re_embedded_even_at_same_dim(tmp_path, backend):
    # An index saved before metadata existed: its model is unknown, so it is not trusted
    # even when the dimension happens to match the configured model.
    settings = _build_index(tmp_path, backend, HashingEmbedder(256))
    (settings.index_dir / META_FILE).unlink()
    loaded = load_index(backend, settings.index_dir)
    assert loaded.embedding is None
    assert len(loaded) > 0

    embedder = CountingEmbedder(256)
    service = RagService(settings, embedder, loaded)

    assert embedder.document_calls == 1
    assert _meta(settings)["embedding_model"] == "hashing-256"
    assert service.search("Командировочные расходы")[0].chunk.filename == "travel.txt"


def test_prefix_change_re_embeds(tmp_path, backend):
    settings = _build_index(tmp_path, backend, HashingEmbedder(256))
    _reopen(settings, PrefixedEmbedder(256))
    assert _meta(settings)["passage_prefix"] == "passage: "


@pytest.mark.parametrize(
    "meta",
    [
        "{not json",
        json.dumps({"embedding_model": "hashing-256"}),  # keys missing
        # Claims the right model but a dimension the vectors do not have.
        json.dumps(
            {
                "embedding_model": "hashing-256",
                "dimension": 768,
                "query_prefix": "",
                "passage_prefix": "",
            }
        ),
    ],
)
def test_bad_metadata_is_treated_as_unknown_model(tmp_path, backend, meta):
    settings = _build_index(tmp_path, backend, HashingEmbedder(256))
    (settings.index_dir / META_FILE).write_text(meta, encoding="utf-8")

    assert load_index(backend, settings.index_dir).embedding is None
    embedder = CountingEmbedder(256)
    _reopen(settings, embedder)
    assert embedder.document_calls == 1
    assert _meta(settings)["dimension"] == 256


def test_failed_re_embed_keeps_the_old_files(tmp_path, backend):
    settings = _build_index(tmp_path, backend, HashingEmbedder(256))
    files = {p.name: p.read_bytes() for p in settings.index_dir.iterdir()}

    with pytest.raises(EmbeddingError):
        _reopen(settings, CountingEmbedder(768, fail=True))

    assert {p.name: p.read_bytes() for p in settings.index_dir.iterdir()} == files


def test_empty_index_takes_the_configured_embedder_without_embedding(tmp_path, backend):
    settings = _settings(tmp_path, backend)
    embedder = CountingEmbedder(768)
    service = _reopen(settings, embedder)
    assert embedder.document_calls == 0
    assert service.index.embedding == embedder.identity
