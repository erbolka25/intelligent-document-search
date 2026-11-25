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


class RecordingModel:
    def __init__(self):
        self.texts: list[str] = []

    def encode(self, texts, **kwargs):
        self.texts.extend(texts)
        return np.ones((len(texts), 4), dtype=np.float32)


def test_explicit_prefixes_are_prepended():
    embedder = SentenceTransformerEmbedder(
        "any-model", query_prefix="query: ", passage_prefix="passage: "
    )
    embedder._model = model = RecordingModel()
    embedder.embed_query("q")
    embedder.embed_documents(["d"])
    assert model.texts == ["query: q", "passage: d"]


def test_no_prefixes_by_default_even_for_e5_names():
    # No more guessing from the model name: prefixes come only from the settings.
    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
    embedder._model = model = RecordingModel()
    embedder.embed_query("q")
    embedder.embed_documents(["d"])
    assert model.texts == ["q", "d"]


def test_settings_default_to_e5_prefixes_and_read_them_from_env(tmp_path):
    from document_search.config import Settings

    defaults = Settings(_env_file=None)
    assert (defaults.embedding_query_prefix, defaults.embedding_passage_prefix) == (
        "query: ",
        "passage: ",
    )
    env = tmp_path / ".env"
    env.write_text('EMBEDDING_QUERY_PREFIX="q: "\nEMBEDDING_PASSAGE_PREFIX=""\n', encoding="utf-8")
    custom = Settings(_env_file=env)
    assert (custom.embedding_query_prefix, custom.embedding_passage_prefix) == ("q: ", "")
