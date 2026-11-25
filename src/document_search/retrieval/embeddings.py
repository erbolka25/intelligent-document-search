"""Text embedding models."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero vectors are left as zeros."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class BaseEmbedder(ABC):
    """Turns text into L2-normalized float32 vectors."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Return an array of shape (dim,)."""

    @property
    def identity(self) -> dict[str, str]:
        """What determines the vectors: model and prefixes. Stored with the index, so
        vectors from different models are never mixed. Override for new embedders."""
        return {"model": type(self).__name__, "query_prefix": "", "passage_prefix": ""}


class SentenceTransformerEmbedder(BaseEmbedder):
    """sentence-transformers model, loaded lazily on first use.

    ``query_prefix`` / ``passage_prefix`` are prepended to queries and documents.
    E5 models are trained with "query: " / "passage: "; models such as bge-m3 use none.
    They come from EMBEDDING_QUERY_PREFIX / EMBEDDING_PASSAGE_PREFIX in the settings.
    """

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s", self.model_name)
                # Local cache first: no network calls, faster startup, works offline.
                try:
                    self._model = SentenceTransformer(
                        self.model_name, device="cpu", local_files_only=True
                    )
                except OSError:
                    logger.info("Model %s is not cached yet: downloading it", self.model_name)
                    self._model = SentenceTransformer(self.model_name, device="cpu")
            return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._get_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalize(vectors)

    @property
    def identity(self) -> dict[str, str]:
        return {
            "model": self.model_name,
            "query_prefix": self._query_prefix,
            "passage_prefix": self._passage_prefix,
        }

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self._passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self._query_prefix + text])[0]
