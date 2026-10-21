"""Model loading prefers the local cache, without loading the real model."""

import sys
import types

import numpy as np
import pytest

from document_search.retrieval import SentenceTransformerEmbedder


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 4), dtype=np.float32)


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Install a fake `sentence_transformers` module that records constructor calls."""
    calls: list[dict] = []
    state = {"cached": True}

    def SentenceTransformer(name, **kwargs):  # mirrors the real class name
        calls.append(kwargs)
        if kwargs.get("local_files_only") and not state["cached"]:
            raise OSError("not in the local cache")
        return FakeModel()

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return calls, state


def test_cached_model_loads_without_network(fake_sentence_transformers):
    calls, _ = fake_sentence_transformers
    SentenceTransformerEmbedder("some-e5-model").embed_query("hello")
    assert calls == [{"device": "cpu", "local_files_only": True}]


def test_missing_model_falls_back_to_download(fake_sentence_transformers):
    calls, state = fake_sentence_transformers
    state["cached"] = False
    vector = SentenceTransformerEmbedder("some-e5-model").embed_query("hello")
    assert calls == [{"device": "cpu", "local_files_only": True}, {"device": "cpu"}]
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_model_is_loaded_once(fake_sentence_transformers):
    calls, _ = fake_sentence_transformers
    embedder = SentenceTransformerEmbedder("some-e5-model")
    embedder.embed_query("a")
    embedder.embed_documents(["b", "c"])
    assert len(calls) == 1
