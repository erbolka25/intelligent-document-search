"""Orchestrates ingestion, retrieval and generation behind one object."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Literal

from document_search.config import Settings
from document_search.generation import (
    BaseLLMClient,
    LLMError,
    build_llm_client,
    generate_answer,
)
from document_search.ingestion import compute_doc_id, file_extension, ingest_document
from document_search.retrieval import (
    BaseEmbedder,
    BaseVectorIndex,
    SearchResult,
    SentenceTransformerEmbedder,
    load_index,
)

logger = logging.getLogger(__name__)


class DocumentLimitError(RuntimeError):
    """The index already holds the maximum number of documents."""


class EmbeddingError(RuntimeError):
    """The embedding model failed, so text cannot be searched or indexed right now."""


@dataclass
class AddResult:
    doc_id: str
    filename: str
    chunks: int
    duplicate: bool
    skipped_pages: list[int] = field(default_factory=list)


@dataclass
class AskResult:
    mode: Literal["rag", "retrieval_only"]
    answer: str | None
    answer_found: bool
    citations: list[SearchResult] = field(default_factory=list)
    results: list[SearchResult] = field(default_factory=list)
    # Why the answer is missing, as a stable code the UI translates: llm_disabled,
    # embedding_unavailable, or an LLMError code (auth, quota, rate_limit, network, ...).
    notice: str | None = None


class RagService:
    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbedder,
        index: BaseVectorIndex,
        llm: BaseLLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.index = index
        self.llm = llm
        # One process, one writer: the lock keeps index vectors and metadata in sync.
        self._lock = threading.RLock()

    def add_document(self, data: bytes, filename: str) -> AddResult:
        doc_id = compute_doc_id(data)
        with self._lock:
            if self.index.has_document(doc_id):
                existing = next(d for d in self.index.documents() if d["doc_id"] == doc_id)
                return AddResult(doc_id, existing["filename"], existing["chunks"], duplicate=True)
            if len(self.index.documents()) >= self.settings.max_documents:
                raise DocumentLimitError(f"Limit of {self.settings.max_documents} documents reached.")

        ingested = ingest_document(
            data,
            filename,
            max_chars=self.settings.chunk_max_chars,
            overlap_chars=self.settings.chunk_overlap_chars,
        )
        chunks = ingested.chunks
        try:
            vectors = self.embedder.embed_documents([c.text for c in chunks])
        except Exception as exc:
            logger.exception("Embedding %s failed", filename)
            raise EmbeddingError("The embedding model failed.") from exc

        with self._lock:
            if self.index.has_document(doc_id):  # same file uploaded concurrently
                return AddResult(doc_id, filename, len(chunks), duplicate=True)
            # Stored under a content-derived name: user input never reaches the path.
            upload_dir = self.settings.upload_dir
            upload_dir.mkdir(parents=True, exist_ok=True)
            upload_path = upload_dir / f"{doc_id}{file_extension(filename)}"
            try:
                upload_path.write_bytes(data)
                self.index.add(vectors, chunks)
                self.index.save(self.settings.index_dir)
            except Exception:
                # Undo in memory and on disk, so a failed save never leaves a document
                # that is searchable now but gone after a restart.
                logger.exception("Indexing %s failed; rolling back", filename)
                self.index.remove_document(doc_id)
                upload_path.unlink(missing_ok=True)
                self._try_save()
                raise
        logger.info("Indexed %s (doc_id=%s, %d chunks)", filename, doc_id, len(chunks))
        return AddResult(doc_id, filename, len(chunks), False, ingested.skipped_pages)

    def _try_save(self) -> None:
        """Best effort: a save interrupted between its two files leaves them mismatched."""
        try:
            self.index.save(self.settings.index_dir)
        except Exception:
            logger.exception("Could not re-save the index after a rollback")

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Chunks scoring at least ``min_score``; raises EmbeddingError if the model fails."""
        try:
            query_vector = self.embedder.embed_query(query)
        except Exception as exc:
            logger.exception("Embedding the query failed")
            raise EmbeddingError("The embedding model failed.") from exc
        with self._lock:
            return self.index.search(
                query_vector, top_k or self.settings.top_k, min_score=self.settings.min_score
            )

    def ask(self, question: str, top_k: int | None = None) -> AskResult:
        try:
            results = self.search(question, top_k)
        except EmbeddingError:
            return AskResult("retrieval_only", None, False, notice="embedding_unavailable")
        # search() already dropped chunks below min_score, so without an LLM "found"
        # means at least one passage cleared the relevance threshold.
        if self.llm is None:
            return AskResult(
                "retrieval_only",
                None,
                bool(results),
                results=results,
                notice="llm_disabled",
            )
        try:
            answer = generate_answer(self.llm, question, results)
        except LLMError as exc:
            logger.warning(
                "LLM failed, falling back to retrieval-only",
                extra={"fields": {"llm_error": exc.code, "detail": str(exc)}},
            )
            return AskResult(
                "retrieval_only", None, bool(results), results=results, notice=exc.code
            )
        except Exception:  # never let an SDK surprise break /ask
            logger.exception("Unexpected LLM failure, falling back to retrieval-only")
            return AskResult("retrieval_only", None, bool(results), results=results, notice="api")
        return AskResult(
            "rag", answer.text or None, answer.answer_found, answer.citations, results
        )

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from the index and disk; False if it was not indexed."""
        with self._lock:
            if not self.index.remove_document(doc_id):
                return False
            self.index.save(self.settings.index_dir)
            self._delete_upload(doc_id)
        logger.info("Deleted document %s", doc_id)
        return True

    def clear_documents(self) -> int:
        """Remove every document; returns how many were removed."""
        with self._lock:
            doc_ids = [d["doc_id"] for d in self.index.documents()]
            self.index.clear()
            self.index.save(self.settings.index_dir)
            for doc_id in doc_ids:
                self._delete_upload(doc_id)
        logger.info("Cleared %d documents", len(doc_ids))
        return len(doc_ids)

    def _delete_upload(self, doc_id: str) -> None:
        # doc_id is a hex content hash, never raw user input, so the glob stays inside upload_dir.
        if self.settings.upload_dir.is_dir():
            for path in self.settings.upload_dir.glob(f"{doc_id}.*"):
                path.unlink(missing_ok=True)

    def documents(self) -> list[dict]:
        with self._lock:
            return self.index.documents()

    def stats(self) -> dict:
        with self._lock:
            return {
                "documents": len(self.index.documents()),
                "chunks": len(self.index),
                "index_backend": self.index.name,
                "llm_enabled": self.llm is not None,
                "llm_provider": self.llm.provider if self.llm else None,
                "llm_model": self.llm.model if self.llm else None,
            }


def build_service(settings: Settings) -> RagService:
    """Wire up real components from settings."""
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    index = load_index(settings.index_backend, settings.index_dir)
    llm = build_llm_client(
        settings.llm_provider,
        settings.llm_api_key(),
        settings.resolved_llm_model,
        settings.llm_effort,
        settings.llm_max_tokens,
        settings.llm_timeout_s,
    )
    return RagService(settings, embedder, index, llm)
