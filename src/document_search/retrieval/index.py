"""Vector indexes with cosine similarity (inner product over normalized vectors)."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from document_search.ingestion import Chunk
from document_search.retrieval.embeddings import l2_normalize

logger = logging.getLogger(__name__)

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int


class BaseVectorIndex(ABC):
    """Stores normalized vectors alongside the chunks they were built from."""

    name: str

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self.dim: int | None = None

    # --- backend-specific hooks -------------------------------------------
    @abstractmethod
    def _add_vectors(self, vectors: np.ndarray) -> None: ...

    @abstractmethod
    def _search_vectors(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, positions) of the k best matches, best first."""

    @abstractmethod
    def _all_vectors(self) -> np.ndarray: ...

    @abstractmethod
    def _reset_vectors(self) -> None: ...

    # --- public API -------------------------------------------------------
    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        vectors = l2_normalize(np.atleast_2d(vectors))
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")
        if not chunks:
            return
        if self.dim is None:
            self.dim = vectors.shape[1]
        elif vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors of dim {self.dim}, got {vectors.shape[1]}")
        self._add_vectors(vectors)
        self._chunks.extend(chunks)

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, min_score: float | None = None
    ) -> list[SearchResult]:
        if not self._chunks or top_k <= 0:
            return []
        query = l2_normalize(np.asarray(query_vector).reshape(-1))
        if query.shape[0] != self.dim:
            raise ValueError(f"Expected query of dim {self.dim}, got {query.shape[0]}")
        scores, positions = self._search_vectors(query, min(top_k, len(self._chunks)))
        results = []
        for score, pos in zip(scores, positions, strict=True):
            if pos < 0 or (min_score is not None and score < min_score):
                continue
            results.append(SearchResult(self._chunks[pos], float(score), len(results) + 1))
        return results

    def remove_document(self, doc_id: str) -> int:
        """Drop all chunks of a document by rebuilding the index; returns chunks removed."""
        keep = [c.doc_id != doc_id for c in self._chunks]
        removed = keep.count(False)
        if removed:
            vectors = self._all_vectors()[np.array(keep, dtype=bool)]
            chunks = [c for c, k in zip(self._chunks, keep, strict=True) if k]
            self.clear()
            self.add(vectors, chunks)
        return removed

    def clear(self) -> None:
        self._chunks = []
        self.dim = None
        self._reset_vectors()

    def has_document(self, doc_id: str) -> bool:
        return any(c.doc_id == doc_id for c in self._chunks)

    def documents(self) -> list[dict]:
        """One entry per document: doc_id, filename and chunk count."""
        docs: dict[str, dict] = {}
        for c in self._chunks:
            entry = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "filename": c.filename, "chunks": 0})
            entry["chunks"] += 1
        return list(docs.values())

    def save(self, directory: str | Path) -> None:
        """Persist vectors and chunk metadata; files are replaced atomically."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        vectors = self._all_vectors() if self._chunks else np.zeros((0, 0), dtype=np.float32)

        tmp_vectors = directory / (VECTORS_FILE + ".tmp")
        with open(tmp_vectors, "wb") as f:
            np.save(f, vectors)
        tmp_chunks = directory / (CHUNKS_FILE + ".tmp")
        tmp_chunks.write_text(
            json.dumps([c.to_dict() for c in self._chunks], ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp_vectors, directory / VECTORS_FILE)
        os.replace(tmp_chunks, directory / CHUNKS_FILE)

    def load(self, directory: str | Path) -> None:
        """Load a previously saved index; a missing directory leaves it empty."""
        directory = Path(directory)
        vectors_path, chunks_path = directory / VECTORS_FILE, directory / CHUNKS_FILE
        if not (vectors_path.exists() and chunks_path.exists()):
            return
        vectors = np.load(vectors_path)
        chunks = [Chunk(**d) for d in json.loads(chunks_path.read_text(encoding="utf-8"))]
        self.add(vectors, chunks)
        logger.info("Loaded %s index with %d chunks from %s", self.name, len(self), directory)


class NumpyIndex(BaseVectorIndex):
    """Brute-force search: one matrix-vector product over all stored vectors."""

    name = "numpy"

    def __init__(self) -> None:
        super().__init__()
        self._matrix = np.zeros((0, 0), dtype=np.float32)

    def _add_vectors(self, vectors: np.ndarray) -> None:
        self._matrix = vectors if self._matrix.size == 0 else np.vstack([self._matrix, vectors])

    def _search_vectors(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self._matrix @ query
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return scores[top], top

    def _all_vectors(self) -> np.ndarray:
        return self._matrix

    def _reset_vectors(self) -> None:
        self._matrix = np.zeros((0, 0), dtype=np.float32)


class FaissIndex(BaseVectorIndex):
    """Exact inner-product search with FAISS IndexFlatIP."""

    name = "faiss"

    def __init__(self) -> None:
        super().__init__()
        self._index = None

    def _add_vectors(self, vectors: np.ndarray) -> None:
        import faiss

        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(np.ascontiguousarray(vectors))

    def _search_vectors(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, positions = self._index.search(np.ascontiguousarray(query[None, :]), k)
        return scores[0], positions[0]

    def _all_vectors(self) -> np.ndarray:
        return self._index.reconstruct_n(0, self._index.ntotal)

    def _reset_vectors(self) -> None:
        self._index = None


INDEX_BACKENDS: dict[str, type[BaseVectorIndex]] = {"numpy": NumpyIndex, "faiss": FaissIndex}


def create_index(backend: str) -> BaseVectorIndex:
    try:
        return INDEX_BACKENDS[backend]()
    except KeyError:
        raise ValueError(f"Unknown index backend '{backend}'. Use one of {list(INDEX_BACKENDS)}.")


def load_index(backend: str, directory: str | Path) -> BaseVectorIndex:
    index = create_index(backend)
    index.load(directory)
    return index
