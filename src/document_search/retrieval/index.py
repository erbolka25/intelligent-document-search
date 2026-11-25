"""Vector indexes with cosine similarity (inner product over normalized vectors)."""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from document_search.ingestion import Chunk
from document_search.retrieval.embeddings import l2_normalize

logger = logging.getLogger(__name__)

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"
# Which embedder produced the vectors: model, prefixes and dimension.
META_FILE = "index_meta.json"


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
        # Identity of the embedder behind the vectors (see BaseEmbedder.identity), saved
        # to META_FILE. None: unknown, e.g. an index saved before metadata existed.
        self.embedding: dict[str, str] | None = None

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

    def replace_vectors(self, vectors: np.ndarray) -> None:
        """Swap in new vectors for the same chunks, e.g. after a model change."""
        chunks = list(self._chunks)
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")
        self.clear()
        self.add(vectors, chunks)

    def clear(self) -> None:
        """Drop all chunks and vectors; the embedder identity is kept."""
        self._chunks = []
        self.dim = None
        self._reset_vectors()

    def chunks(self) -> list[Chunk]:
        """The stored chunks, in vector order."""
        return list(self._chunks)

    def has_document(self, doc_id: str) -> bool:
        return any(c.doc_id == doc_id for c in self._chunks)

    def documents(self) -> list[dict]:
        """One entry per document: doc_id, filename and chunk count."""
        docs: dict[str, dict] = {}
        for c in self._chunks:
            entry = docs.setdefault(
                c.doc_id, {"doc_id": c.doc_id, "filename": c.filename, "chunks": 0}
            )
            entry["chunks"] += 1
        return list(docs.values())

    def save(self, directory: str | Path) -> None:
        """Persist vectors, chunks and embedder metadata; each file is replaced atomically."""
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
        meta_path = directory / META_FILE
        tmp_meta = None
        if self.embedding is not None:
            tmp_meta = directory / (META_FILE + ".tmp")
            meta = {
                "embedding_model": self.embedding["model"],
                "dimension": self.dim,
                "query_prefix": self.embedding["query_prefix"],
                "passage_prefix": self.embedding["passage_prefix"],
            }
            tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_vectors, directory / VECTORS_FILE)
        os.replace(tmp_chunks, directory / CHUNKS_FILE)
        if tmp_meta is not None:
            os.replace(tmp_meta, meta_path)
        else:
            # Unknown embedder: a metadata file left from earlier would vouch for these vectors.
            meta_path.unlink(missing_ok=True)

    def load(self, directory: str | Path) -> None:
        """Load a previously saved index; a missing directory leaves it empty.

        Unreadable or mismatched files also leave it empty, so the server still starts.
        They are renamed rather than deleted, so the next save cannot overwrite them.
        """
        directory = Path(directory)
        vectors_path, chunks_path = directory / VECTORS_FILE, directory / CHUNKS_FILE
        if not (vectors_path.exists() and chunks_path.exists()):
            return
        try:
            vectors = np.load(vectors_path)
            chunks = [Chunk(**d) for d in json.loads(chunks_path.read_text(encoding="utf-8"))]
            self.add(vectors, chunks)
        except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
            self.clear()
            suffix = f".corrupt-{int(time.time())}"
            for path in (vectors_path, chunks_path, directory / META_FILE):
                if path.exists():
                    path.replace(path.with_name(path.name + suffix))
            logger.error(
                "Index files in %s are unreadable (%s: %s); starting with an empty index. "
                "The old files were renamed with suffix %s; re-upload the documents.",
                directory,
                type(exc).__name__,
                exc,
                suffix,
            )
            return
        self.embedding = self._load_meta(directory / META_FILE)
        logger.info("Loaded %s index with %d chunks from %s", self.name, len(self), directory)

    def _load_meta(self, path: Path) -> dict[str, str] | None:
        """Embedder identity from META_FILE; None if missing, unreadable or inconsistent
        with the loaded vectors, so the caller treats the vectors as from an unknown model."""
        if not path.exists():
            return None
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            identity = {
                "model": str(meta["embedding_model"]),
                "query_prefix": str(meta["query_prefix"]),
                "passage_prefix": str(meta["passage_prefix"]),
            }
            dimension = meta["dimension"]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Ignoring unreadable index metadata %s (%s)", path, exc)
            return None
        if self._chunks and dimension != self.dim:
            logger.warning(
                "Index metadata %s says dimension %s, but the vectors have %s; ignoring it",
                path,
                dimension,
                self.dim,
            )
            return None
        return identity


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
