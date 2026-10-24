import json

import pytest

from document_search.evaluation import (
    EvalCase,
    benchmark_latency,
    evaluate,
    evaluate_answers,
    load_cases,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from document_search.ingestion import Chunk

RETRIEVED = ["a", "b", "c", "d"]


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, {"b", "z"}, 1) == 0.0
    assert recall_at_k(RETRIEVED, {"b", "z"}, 2) == 0.5
    assert recall_at_k(RETRIEVED, {"a", "c"}, 3) == 1.0
    assert recall_at_k(RETRIEVED, set(), 3) == 0.0


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, {"a", "c"}, 1) == 1.0
    assert precision_at_k(RETRIEVED, {"a", "c"}, 4) == 0.5
    assert precision_at_k(RETRIEVED, {"a"}, 0) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, {"a"}) == 1.0
    assert reciprocal_rank(RETRIEVED, {"c", "d"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RETRIEVED, {"z"}) == 0.0


CHUNKS = [
    Chunk("1", "hr.txt", None, 0, 0, 0, "Отпуск 28 календарных дней."),
    Chunk("1", "hr.txt", None, 1, 0, 0, "Удалённая работа два дня в неделю."),
    Chunk("2", "it.txt", None, 0, 0, 0, "Пароль менять каждые 90 дней."),
]


def test_load_cases_matches_by_text(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [{"question": "q", "relevant": [{"filename": "it.txt", "contains": "90 дней"}]}]
        ),
        encoding="utf-8",
    )
    assert load_cases(path, CHUNKS)[0].relevant == {"it.txt#0"}


def test_load_cases_rejects_stale_spec(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps([{"question": "q", "relevant": [{"filename": "it.txt", "contains": "нет"}]}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_cases(path, CHUNKS)


def test_evaluate_both_backends_agree(embedder):
    pytest.importorskip("faiss")
    cases = [
        EvalCase("пароль менять 90 дней", {"it.txt#0"}),
        EvalCase("отпуск 28 дней", {"hr.txt#0"}),
    ]
    report = evaluate(embedder, CHUNKS, cases, [1, 3], ["numpy", "faiss"])
    numpy_metrics = report["backends"]["numpy"]["metrics"]
    assert numpy_metrics == report["backends"]["faiss"]["metrics"]
    assert numpy_metrics["MRR"] == 1.0 and numpy_metrics["Recall@1"] == 1.0


def test_load_cases_accepts_unanswerable_questions(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {"question": "q", "relevant": [{"filename": "it.txt", "contains": "90 дней"}]},
                {"question": "бюджет на корпоратив?", "answerable": False, "relevant": []},
            ]
        ),
        encoding="utf-8",
    )
    cases = load_cases(path, CHUNKS)
    assert [c.answerable for c in cases] == [True, False]
    assert cases[1].relevant == set()


def test_load_cases_rejects_unanswerable_with_relevant_spec(tmp_path):
    path = tmp_path / "eval.json"
    spec = [{"filename": "it.txt", "contains": "90 дней"}]
    path.write_text(
        json.dumps([{"question": "q", "answerable": False, "relevant": spec}]), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_cases(path, CHUNKS)


MIXED_CASES = [
    EvalCase("пароль менять 90 дней", {"it.txt#0"}),
    EvalCase("отпуск 28 дней", {"hr.txt#0"}),
    EvalCase("квантовая хромодинамика", set(), answerable=False),
]


def test_evaluate_scores_refusals_and_keeps_ranking_on_answerable(embedder):
    report = evaluate(embedder, CHUNKS, MIXED_CASES, [1], ["numpy"], min_score=0.3)
    data = report["backends"]["numpy"]
    assert report["questions"] == 2 and report["unanswerable"] == 1
    # the unanswerable case does not drag the ranking metrics down
    assert data["metrics"]["MRR"] == 1.0
    assert data["metrics"]["CorrectRefusal"] == 1.0
    assert data["metrics"]["FalseRefusal"] == 0.0
    assert data["not_refused"] == []


def test_evaluate_without_min_score_reports_no_refusals(embedder):
    data = evaluate(embedder, CHUNKS, MIXED_CASES, [1], ["numpy"])["backends"]["numpy"]
    assert "CorrectRefusal" not in data["metrics"] and data["not_refused"] == []


def test_evaluate_answers_counts_rag_refusals(embedder):
    from tests.test_generation import FakeLLM

    # Answers every question with a citation, so only empty retrieval can refuse.
    report = evaluate_answers(FakeLLM("Ответ [1]."), embedder, CHUNKS, MIXED_CASES, 3, 0.3)
    assert report["metrics"] == {"CorrectRefusal": 1.0, "FalseRefusal": 0.0}
    assert report["not_refused"] == [] and report["falsely_refused"] == []

    # A model that never cites anything refuses everything, answerable or not.
    report = evaluate_answers(FakeLLM("Без ссылок."), embedder, CHUNKS, MIXED_CASES, 3, 0.0)
    assert report["metrics"] == {"CorrectRefusal": 1.0, "FalseRefusal": 1.0}
    assert len(report["falsely_refused"]) == 2


def test_evaluate_answers_records_llm_errors(embedder):
    from document_search.generation import LLMError
    from tests.test_generation import FakeLLM

    llm = FakeLLM(error=LLMError("quota", "no credits"))
    report = evaluate_answers(llm, embedder, CHUNKS, MIXED_CASES, 3, 0.0)
    assert [e["error"] for e in report["errors"]] == ["quota"] * 3
    assert report["metrics"] == {}


def test_benchmark_latency_runs():
    result = benchmark_latency(["numpy"], n=100, dim=8, queries=3)
    assert result["numpy"] >= 0
