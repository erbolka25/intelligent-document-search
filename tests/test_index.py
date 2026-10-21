import numpy as np
import pytest

from document_search.ingestion import Chunk
from document_search.retrieval import create_index, load_index

BACKENDS = ["numpy", pytest.param("faiss", marks=pytest.mark.faiss)]

TEXTS = [
    "Отпуск сотрудника составляет 28 календарных дней в году.",
    "Пароль от рабочей почты нужно менять каждые 90 дней.",
    "Командировочные расходы возмещаются по авансовому отчёту.",
]


def _chunks(texts: list[str], doc_id: str = "doc1") -> list[Chunk]:
    return [
        Chunk(doc_id, f"{doc_id}.txt", None, i, 0, len(t), t) for i, t in enumerate(texts)
    ]


@pytest.fixture(params=BACKENDS)
def index(request):
    if request.param == "faiss":
        pytest.importorskip("faiss")
    return create_index(request.param)


def test_empty_index_returns_nothing(index, embedder):
    assert len(index) == 0
    assert index.search(embedder.embed_query("отпуск"), top_k=5) == []


def test_finds_most_relevant_chunk(index, embedder):
    index.add(embedder.embed_documents(TEXTS), _chunks(TEXTS))
    results = index.search(embedder.embed_query("Пароль от почты"), top_k=2)

    assert len(results) == 2
    assert results[0].chunk.text == TEXTS[1]
    assert results[0].rank == 1 and results[1].rank == 2
    assert results[0].score >= results[1].score
    assert -1.0 - 1e-6 <= results[1].score <= results[0].score <= 1.0 + 1e-6


def test_cosine_scores_match_manual_computation(index):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(10, 8)).astype(np.float32)
    query = rng.normal(size=8).astype(np.float32)
    index.add(vectors, _chunks([f"t{i}" for i in range(10)]))

    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    expected = normed @ (query / np.linalg.norm(query))
    results = index.search(query, top_k=10)

    assert [r.chunk.text for r in results] == [f"t{i}" for i in np.argsort(-expected)]
    np.testing.assert_allclose([r.score for r in results], np.sort(expected)[::-1], rtol=1e-5)


def test_no_relevant_result_with_min_score(index, embedder):
    index.add(embedder.embed_documents(TEXTS), _chunks(TEXTS))
    assert index.search(embedder.embed_query("квантовая хромодинамика"), min_score=0.3) == []


def test_top_k_larger_than_index_and_non_positive(index, embedder):
    index.add(embedder.embed_documents(TEXTS), _chunks(TEXTS))
    query = embedder.embed_query("отпуск")
    assert len(index.search(query, top_k=100)) == len(TEXTS)
    assert index.search(query, top_k=0) == []


def test_dimension_mismatch_raises(index):
    index.add(np.ones((1, 4)), _chunks(["a"]))
    with pytest.raises(ValueError):
        index.add(np.ones((1, 5)), _chunks(["b"]))
    with pytest.raises(ValueError):
        index.search(np.ones(5))


def test_documents_summary(index, embedder):
    index.add(embedder.embed_documents(TEXTS), _chunks(TEXTS, "a"))
    index.add(embedder.embed_documents(TEXTS[:1]), _chunks(TEXTS[:1], "b"))
    assert index.has_document("a") and not index.has_document("zzz")
    assert index.documents() == [
        {"doc_id": "a", "filename": "a.txt", "chunks": 3},
        {"doc_id": "b", "filename": "b.txt", "chunks": 1},
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_roundtrip(tmp_path, embedder, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
    index = create_index(backend)
    index.add(embedder.embed_documents(TEXTS), _chunks(TEXTS))
    index.save(tmp_path)

    restored = load_index(backend, tmp_path)
    query = embedder.embed_query("командировочные расходы")
    assert len(restored) == len(TEXTS)
    assert restored.search(query, 3) == index.search(query, 3)


def test_load_missing_directory_gives_empty_index(tmp_path):
    assert len(load_index("numpy", tmp_path / "nothing")) == 0


def test_numpy_and_faiss_agree(embedder):
    pytest.importorskip("faiss")
    np_index, faiss_index = create_index("numpy"), create_index("faiss")
    vectors = embedder.embed_documents(TEXTS)
    np_index.add(vectors, _chunks(TEXTS))
    faiss_index.add(vectors, _chunks(TEXTS))
    query = embedder.embed_query("отпуск дней")
    a, b = np_index.search(query, 3), faiss_index.search(query, 3)
    assert [r.chunk for r in a] == [r.chunk for r in b]
    np.testing.assert_allclose([r.score for r in a], [r.score for r in b], rtol=1e-5)


def test_unknown_backend():
    with pytest.raises(ValueError):
        create_index("annoy")


def test_remove_document_and_clear(index, embedder, tmp_path):
    index.add(embedder.embed_documents(TEXTS), _chunks(TEXTS, "a"))
    index.add(embedder.embed_documents(TEXTS[:1]), _chunks(TEXTS[:1], "b"))
    assert index.remove_document("a") == 3
    assert index.remove_document("missing") == 0
    assert [d["doc_id"] for d in index.documents()] == ["b"]
    assert index.search(embedder.embed_query("отпуск"), 5)[0].chunk.doc_id == "b"

    index.clear()
    assert len(index) == 0 and index.search(embedder.embed_query("отпуск"), 5) == []
    index.save(tmp_path)
    assert len(load_index(index.name, tmp_path)) == 0
    index.add(embedder.embed_documents(TEXTS), _chunks(TEXTS))  # usable after clear
    assert len(index) == 3
