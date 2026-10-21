import json

import pytest

from document_search.evaluation import (
    EvalCase,
    benchmark_latency,
    evaluate,
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


def test_benchmark_latency_runs():
    result = benchmark_latency(["numpy"], n=100, dim=8, queries=3)
    assert result["numpy"] >= 0
