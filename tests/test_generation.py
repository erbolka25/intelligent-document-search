import pytest

from document_search.config import Settings
from document_search.generation import (
    NO_ANSWER,
    BaseLLMClient,
    LLMError,
    build_user_prompt,
    extract_citations,
    generate_answer,
)
from document_search.ingestion import Chunk
from document_search.retrieval import SearchResult, create_index
from document_search.service import RagService


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
    assert prompt.endswith("Question: Вопрос?")


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


def _service(tmp_path, embedder, llm=None) -> RagService:
    settings = Settings(data_dir=tmp_path, index_backend="numpy", _env_file=None)
    service = RagService(settings, embedder, create_index("numpy"), llm)
    service.add_document("Отпуск составляет 28 дней. Пароль меняется раз в 90 дней.".encode(), "hr.txt")
    return service


def test_ask_rag_mode(tmp_path, embedder):
    service = _service(tmp_path, embedder, FakeLLM("28 дней [1]."))
    result = service.ask("Сколько дней отпуск?")
    assert result.mode == "rag"
    assert result.answer == "28 дней [1]."
    assert result.citations[0].chunk.filename == "hr.txt"


def test_ask_without_llm_is_retrieval_only(tmp_path, embedder):
    result = _service(tmp_path, embedder).ask("отпуск")
    assert result.mode == "retrieval_only"
    assert result.answer is None
    assert result.results and result.notice == "llm_disabled"


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
