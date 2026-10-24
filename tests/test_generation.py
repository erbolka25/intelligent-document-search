import pytest

from document_search.config import Settings
from document_search.generation import (
    NO_ANSWER,
    BaseLLMClient,
    LLMError,
    build_user_prompt,
    detect_language,
    extract_citations,
    generate_answer,
)
from document_search.ingestion import Chunk
from document_search.retrieval import SearchResult, create_index
from document_search.service import EmbeddingError, RagService


class FakeLLM(BaseLLMClient):
    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.reply


def _result(n: int, page: int | None = None) -> SearchResult:
    chunk = Chunk("d", f"file{n}.pdf", page, n, 0, 10, f"text {n}")
    return SearchResult(chunk, 0.9 - n / 10, n + 1)


RESULTS = [_result(0, page=1), _result(1), _result(2, page=7)]


def test_prompt_numbers_sources_with_location():
    prompt = build_user_prompt("Вопрос?", RESULTS)
    assert '<source id="1" file="file0.pdf" page="1">' in prompt
    assert '<source id="2" file="file1.pdf" chunk="1">' in prompt
    assert '<source id="3" file="file2.pdf" page="7">' in prompt
    assert "Question: Вопрос?\n\nThe question is in Russian." in prompt


@pytest.mark.parametrize(
    "question, language",
    [
        ("Сколько дней отпуска?", "Russian"),
        ("How often do I change my password?", "English"),
        ("Өргүү канча күн?", "Kyrgyz"),
        ("Что такое VPN?", "Russian"),
        ("2 + 2", None),
    ],
)
def test_detect_language(question, language):
    assert detect_language(question) == language


def test_prompt_names_question_language_after_question():
    prompt = build_user_prompt("How many vacation days?", RESULTS)
    assert prompt.endswith("Write the entire answer in English, "
                           "translating facts from the sources if they are in another language.")


def test_extract_citations_dedupes_and_ignores_out_of_range():
    cited = extract_citations("A [3]. B [1][3]. C [9] [0].", RESULTS)
    assert [c.chunk.chunk_index for c in cited] == [2, 0]


def test_generate_answer_returns_citations():
    llm = FakeLLM("Ответ из документа [2].")
    answer = generate_answer(llm, "Вопрос?", RESULTS)
    assert answer.answer_found
    assert answer.text == "Ответ из документа [2]."
    assert [c.chunk.filename for c in answer.citations] == ["file1.pdf"]
    assert "ONLY" in llm.calls[0][0]


@pytest.mark.parametrize(
    "reply, explanation",
    [
        (NO_ANSWER, ""),
        ("no_answer.", ""),
        (" `NO_ANSWER` ", ""),
        ("  ", ""),
        ("NO_ANSWER: В документах нет этой информации.", "В документах нет этой информации."),
        ("**NO_ANSWER** - The documents do not say.", "The documents do not say."),
    ],
)
def test_generate_answer_no_answer(reply, explanation):
    answer = generate_answer(FakeLLM(reply), "q", RESULTS)
    assert not answer.answer_found
    assert answer.text == explanation
    assert answer.citations == []


def test_answer_mentioning_sentinel_later_is_a_real_answer():
    answer = generate_answer(FakeLLM("Отпуск 28 дней [1]. NO_ANSWER"), "q", RESULTS)
    assert answer.answer_found
    assert answer.text == "Отпуск 28 дней [1]."


@pytest.mark.parametrize(
    "reply",
    [
        "Бюджет на корпоратив составляет 50 000 сом.",  # confident claim, no source
        "В документах нет этой информации. NO_ANSWER",  # sentinel at the end, nothing cited
        "Бюджет 50 000 сом [7].",  # only a marker that points at no source
        "Бюджет 50 000 сом [0].",
    ],
)
def test_answer_without_valid_citation_is_not_found(reply):
    answer = generate_answer(FakeLLM(reply), "q", RESULTS)
    assert not answer.answer_found
    assert answer.text == ""
    assert answer.citations == []


def test_out_of_range_markers_are_removed_from_the_answer():
    answer = generate_answer(FakeLLM("Отпуск 28 дней [1][9]. Пароль раз в 90 дней [4]."), "q", RESULTS)
    assert answer.answer_found
    assert answer.text == "Отпуск 28 дней [1]. Пароль раз в 90 дней."
    assert [c.chunk.chunk_index for c in answer.citations] == [0]


def test_prompt_enforces_context_only_language_and_citations():
    llm = FakeLLM("ok [1]")
    generate_answer(llm, "Канча күн?", RESULTS)
    system = llm.calls[0][0]
    assert "ONLY" in system and NO_ANSWER in system
    assert "language of the question" in system and "[1]" in system


def test_generate_answer_without_context_skips_llm():
    llm = FakeLLM("should not be called")
    answer = generate_answer(llm, "q", [])
    assert not answer.answer_found and llm.calls == []


def _service(tmp_path, embedder, llm=None, min_score=0.0) -> RagService:
    # The default MIN_SCORE is calibrated for e5; the hashing test embedder scores far lower.
    settings = Settings(data_dir=tmp_path, index_backend="numpy", min_score=min_score, _env_file=None)
    service = RagService(settings, embedder, create_index("numpy"), llm)
    service.add_document("Отпуск составляет 28 дней. Пароль меняется раз в 90 дней.".encode(), "hr.txt")
    return service


def test_ask_rag_mode(tmp_path, embedder):
    service = _service(tmp_path, embedder, FakeLLM("28 дней [1]."))
    result = service.ask("Сколько дней отпуск?")
    assert result.mode == "rag"
    assert result.answer == "28 дней [1]."
    assert result.citations[0].chunk.filename == "hr.txt"


def test_ask_below_min_score_answers_not_found_without_calling_llm(tmp_path, embedder):
    llm = FakeLLM("Бюджет 50 000 сом [1].")
    result = _service(tmp_path, embedder, llm, min_score=0.5).ask("квантовая хромодинамика")
    assert result.mode == "rag"
    assert not result.answer_found and result.answer is None
    assert result.results == [] and llm.calls == []


def test_ask_uncited_reply_is_not_found(tmp_path, embedder):
    service = _service(tmp_path, embedder, FakeLLM("Отпуск 35 дней."))
    result = service.ask("Сколько дней отпуск?")
    assert result.mode == "rag"
    assert not result.answer_found and result.answer is None and result.citations == []
    assert result.results  # the closest passages are still shown, collapsed, in the UI


def test_default_min_score_filters_off_topic_e5_scores():
    # Calibrated on data/samples with multilingual-e5-small: the weakest answerable eval
    # question scored 0.778, clearly off-topic questions 0.68-0.70.
    assert 0.70 < Settings(_env_file=None).min_score < 0.778


def test_ask_without_llm_is_retrieval_only(tmp_path, embedder):
    result = _service(tmp_path, embedder).ask("отпуск")
    assert result.mode == "retrieval_only"
    assert result.answer is None
    assert result.results and result.notice == "llm_disabled"


def test_ask_without_llm_answer_found_follows_min_score(tmp_path, embedder):
    service = _service(tmp_path, embedder, min_score=0.3)
    found = service.ask("Сколько дней отпуск?")
    assert found.mode == "retrieval_only" and found.answer_found and found.results
    missed = service.ask("квантовая хромодинамика")
    assert missed.mode == "retrieval_only"
    assert not missed.answer_found and missed.results == []


class BrokenEmbedder:
    """Indexes like the real thing, then fails on queries (e.g. the model crashed)."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.broken = False

    def embed_documents(self, texts):
        if self.broken:
            raise RuntimeError("model crashed")
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        if self.broken:
            raise RuntimeError("model crashed")
        return self.inner.embed_query(text)


def test_search_wraps_embedding_failure(tmp_path, embedder):
    broken = BrokenEmbedder(embedder)
    service = _service(tmp_path, broken)
    broken.broken = True
    with pytest.raises(EmbeddingError):
        service.search("отпуск")


def test_ask_degrades_when_embedding_fails(tmp_path, embedder):
    broken = BrokenEmbedder(embedder)
    llm = FakeLLM("should not be called")
    service = _service(tmp_path, broken, llm)
    broken.broken = True
    result = service.ask("отпуск")
    assert result.mode == "retrieval_only" and result.notice == "embedding_unavailable"
    assert not result.answer_found and result.results == [] and llm.calls == []


def test_add_document_embedding_failure_writes_nothing(tmp_path, embedder):
    broken = BrokenEmbedder(embedder)
    service = _service(tmp_path, broken)
    broken.broken = True
    with pytest.raises(EmbeddingError):
        service.add_document("Командировка оформляется приказом.".encode(), "trips.txt")
    assert len(service.documents()) == 1
    assert len(list(service.settings.upload_dir.iterdir())) == 1


def test_add_document_rolls_back_when_save_fails(tmp_path, embedder, monkeypatch):
    service = _service(tmp_path, embedder)
    saved_chunks = (service.settings.index_dir / "chunks.json").read_text(encoding="utf-8")
    real_save = service.index.save
    calls = []

    def failing_save(directory):
        calls.append(directory)
        if len(calls) == 1:
            raise OSError("disk full")
        real_save(directory)

    monkeypatch.setattr(service.index, "save", failing_save)
    data = "Командировка оформляется приказом.".encode()
    with pytest.raises(OSError):
        service.add_document(data, "trips.txt")

    assert [d["filename"] for d in service.documents()] == ["hr.txt"]
    assert len(list(service.settings.upload_dir.iterdir())) == 1
    assert len(calls) == 2  # the rollback re-saved the restored state
    assert (service.settings.index_dir / "chunks.json").read_text(encoding="utf-8") == saved_chunks
    monkeypatch.setattr(service.index, "save", real_save)
    assert not service.add_document(data, "trips.txt").duplicate  # a retry works


def test_ask_falls_back_when_llm_fails(tmp_path, embedder):
    service = _service(tmp_path, embedder, FakeLLM(error=LLMError("quota", "no credits")))
    result = service.ask("отпуск")
    assert result.mode == "retrieval_only"
    assert result.notice == "quota"
    assert result.results


def test_ask_falls_back_on_unexpected_llm_exception(tmp_path, embedder):
    service = _service(tmp_path, embedder, FakeLLM(error=KeyError("boom")))
    result = service.ask("отпуск")
    assert result.mode == "retrieval_only" and result.notice == "api"


def test_service_persists_and_dedupes(tmp_path, embedder):
    service = _service(tmp_path, embedder)
    again = service.add_document(
        "Отпуск составляет 28 дней. Пароль меняется раз в 90 дней.".encode(), "copy.txt"
    )
    assert again.duplicate and again.filename == "hr.txt"
    assert len(list(service.settings.upload_dir.iterdir())) == 1
    assert (service.settings.index_dir / "chunks.json").exists()
